"""Microphone capture.

Spec §2: the mic stream is ALWAYS OPEN, including while TTS is playing. That is
what makes barge-in possible at M5, so nothing in this module may ever stop or
reopen the stream as part of normal turn-taking.

Frames are delivered as raw little-endian int16 PCM bytes through a queue. The
sounddevice callback runs on a PortAudio thread, so it does the absolute minimum
and never blocks: anything expensive belongs on the consumer side.
"""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from typing import Iterator

import sounddevice as sd

# Silero VAD v5 accepts exactly 512 samples per call at 16 kHz. Spec §2 says
# 20 ms frames (320 samples), which the model rejects, so 512 samples / 32 ms is
# the real frame size for the whole pipeline.
SILERO_FRAME_SAMPLES = 512


@dataclass
class AudioFormat:
    sample_rate: int = 16000
    channels: int = 1
    frame_samples: int = SILERO_FRAME_SAMPLES

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * self.channels * 2  # int16

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_samples / self.sample_rate


class MicrophoneStream:
    """An always-open 16 kHz mono int16 capture stream."""

    def __init__(
        self,
        fmt: AudioFormat | None = None,
        device: int | str | None = None,
        max_queued_frames: int = 200,
    ):
        self.fmt = fmt or AudioFormat()
        self.device = device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queued_frames)
        self._stream: sd.RawInputStream | None = None
        self.dropped_frames = 0
        self.overflow_count = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.RawInputStream(
            samplerate=self.fmt.sample_rate,
            blocksize=self.fmt.frame_samples,
            device=self.device,
            channels=self.fmt.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    # -- capture -----------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Input overflow means we lost audio. Count it; never print from the
            # PortAudio thread.
            if getattr(status, "input_overflow", False):
                self.overflow_count += 1
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            # Consumer is too slow. Drop the oldest frame so we stay near
            # real time rather than accumulating latency forever.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(indata))
            except (queue.Empty, queue.Full):
                pass
            self.dropped_frames += 1

    def frames(self, timeout: float | None = 1.0) -> Iterator[bytes]:
        """Yield captured frames forever. Blocks waiting on the microphone."""
        while True:
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                if self._stream is None:
                    return
                continue

    @property
    def backlog(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def resolve_device(spec: int | str | None) -> int | None:
    """Turn a config value into a sounddevice index.

    Accepts an index, a case-insensitive substring of the device name, or None
    for the system default.
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str) and spec.strip().isdigit():
        return int(spec.strip())

    needle = str(spec).lower()
    matches = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0 and needle in d["name"].lower()
    ]
    if not matches:
        raise ValueError(f"no input device matching {spec!r}")
    return matches[0][0]


def describe_device(index: int | None) -> str:
    info = sd.query_devices(index, kind="input") if index is None else sd.query_devices(index)
    host = sd.query_hostapis(info["hostapi"])["name"]
    idx = "default" if index is None else str(index)
    return f"[{idx}] {info['name']} ({host}, {int(info['default_samplerate'])} Hz native)"


def list_input_devices() -> None:
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        host = sd.query_hostapis(d["hostapi"])["name"]
        mark = " <- default" if i == default_in else ""
        print(
            f"  [{i:>3}] {d['name'][:52]:<52} ch={d['max_input_channels']} "
            f"{host}{mark}",
            file=sys.stdout,
        )
