"""Speaker output: a flushable, interruptible playback queue.

This is the piece spec §7.3 measures. On a confirmed barge-in the audio queue
must be cleared and the stream silenced within 60 ms, so:

* Audio is queued as many small chunks, never one blob.
* flush() clears the queue and silences the in-flight chunk immediately; it does
  not wait for anything to finish.
* The stream stays open across a flush. Stopping and restarting a PortAudio
  stream costs tens of milliseconds and would blow the budget.

It also answers "what did the user actually hear?", which §7.3 needs so that
interrupted speech enters history as only the words that were really spoken.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soxr

# Windows host APIs differ enormously in buffer depth, and that buffer is dead
# weight against the 60 ms flush budget: audio already handed to the device keeps
# playing after the queue is cleared. Measured on this machine at 24 kHz/48 kHz:
#
#     MME              ~192 ms   (unusable for barge-in)
#     WASAPI shared     ~22 ms   (only accepts the device's native rate)
#     WASAPI exclusive  ~32 ms   (locks the device from other apps)
#
# WASAPI shared at the native rate is therefore the default, with Kokoro's 24 kHz
# resampled up to meet it. soxr costs ~0.3 ms per second of audio, which is free.
PREFERRED_HOST_APIS = ("Windows WASAPI", "Windows DirectSound", "MME")


@dataclass
class _Playing:
    text: str
    samples: np.ndarray
    offset: int = 0

    @property
    def fraction_played(self) -> float:
        if self.samples.size == 0:
            return 1.0
        return min(1.0, self.offset / self.samples.size)


@dataclass
class FlushResult:
    """What a flush interrupted.

    `latency_ms` alone is NOT the §7.3 number. Clearing the queue is nearly free,
    but audio already handed to the sound device keeps playing until its buffer
    drains. `audible_stop_ms` is the honest figure and the one to judge against
    the 60 ms budget.
    """

    latency_ms: float
    spoken_text: str
    dropped_chunks: int
    dropped_text: str
    device_latency_ms: float = 0.0

    @property
    def audible_stop_ms(self) -> float:
        return self.latency_ms + self.device_latency_ms


def _prefix_by_fraction(text: str, fraction: float) -> str:
    """Approximate the words actually heard from a partially played chunk."""
    if fraction >= 0.999:
        return text
    if fraction <= 0.0:
        return ""
    words = text.split()
    if not words:
        return ""
    keep = int(round(len(words) * fraction))
    return " ".join(words[: max(0, keep)])


class SpeakerOutput:
    def __init__(
        self,
        sample_rate: int = 24000,
        *,
        device: int | str | None = None,
        blocksize: int = 256,
        channels: int = 1,
        device_sample_rate: int | None = None,
    ):
        """
        Args:
            sample_rate: rate of the audio handed to enqueue() (Kokoro is 24 kHz).
            device_sample_rate: rate the device is opened at. WASAPI shared mode
                only accepts the device's native rate, so audio is resampled to
                match. Defaults to the device's native rate.
        """
        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = blocksize
        self.channels = channels

        if device_sample_rate is None:
            try:
                info = sd.query_devices(device, kind="output") if device is None \
                    else sd.query_devices(device)
                device_sample_rate = int(info["default_samplerate"])
            except Exception:
                device_sample_rate = sample_rate
        self.device_sample_rate = device_sample_rate

        self._lock = threading.Lock()
        self._queue: list[_Playing] = []
        self._current: _Playing | None = None
        self._spoken: list[str] = []
        self._stream: sd.OutputStream | None = None
        self._idle = threading.Event()
        self._idle.set()
        self.underruns = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "SpeakerOutput":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.device_sample_rate,
            blocksize=self.blocksize,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    @property
    def latency_ms(self) -> float:
        """PortAudio's own reported output latency, part of the flush budget."""
        if self._stream is None:
            return 0.0
        return float(self._stream.latency) * 1000.0

    # -- playback ----------------------------------------------------------

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status and getattr(status, "output_underflow", False):
            self.underruns += 1

        written = 0
        with self._lock:
            while written < frames:
                if self._current is None:
                    if not self._queue:
                        break
                    self._current = self._queue.pop(0)

                remaining = self._current.samples.size - self._current.offset
                take = min(remaining, frames - written)
                if take > 0:
                    outdata[written : written + take, 0] = self._current.samples[
                        self._current.offset : self._current.offset + take
                    ]
                    self._current.offset += take
                    written += take

                if self._current.offset >= self._current.samples.size:
                    self._spoken.append(self._current.text)
                    self._current = None

            if self._current is None and not self._queue:
                self._idle.set()

        if written < frames:
            outdata[written:, 0] = 0.0

    def enqueue(self, text: str, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.device_sample_rate != self.sample_rate and samples.size:
            # Done here, on the synthesis thread. Never in the audio callback.
            samples = np.asarray(
                soxr.resample(samples, self.sample_rate, self.device_sample_rate),
                dtype=np.float32,
            )
        with self._lock:
            self._queue.append(_Playing(text=text, samples=samples))
            self._idle.clear()

    def flush(self) -> FlushResult:
        """Stop speaking immediately. This is the §7.3 60 ms path."""
        started = time.monotonic()
        with self._lock:
            spoken_partial = ""
            dropped_text_parts: list[str] = []

            if self._current is not None:
                fraction = self._current.fraction_played
                spoken_partial = _prefix_by_fraction(self._current.text, fraction)
                unheard = self._current.text[len(spoken_partial) :].strip()
                if spoken_partial:
                    self._spoken.append(spoken_partial)
                if unheard:
                    dropped_text_parts.append(unheard)
                self._current = None

            dropped = len(self._queue)
            dropped_text_parts.extend(item.text for item in self._queue)
            self._queue.clear()
            self._idle.set()
            spoken = " ".join(self._spoken).strip()

        return FlushResult(
            latency_ms=(time.monotonic() - started) * 1000.0,
            spoken_text=spoken,
            dropped_chunks=dropped,
            dropped_text=" ".join(dropped_text_parts).strip(),
            device_latency_ms=self.latency_ms,
        )

    # -- state -------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._current is not None or bool(self._queue)

    @property
    def queued_chunks(self) -> int:
        with self._lock:
            return len(self._queue) + (1 if self._current is not None else 0)

    @property
    def queued_seconds(self) -> float:
        with self._lock:
            total = sum(item.samples.size for item in self._queue)
            if self._current is not None:
                total += self._current.samples.size - self._current.offset
            return total / self.device_sample_rate

    def spoken_text(self) -> str:
        with self._lock:
            return " ".join(self._spoken).strip()

    def reset_spoken(self) -> None:
        with self._lock:
            self._spoken.clear()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout=timeout)


def describe_output_device(index: int | None) -> str:
    info = sd.query_devices(index, kind="output") if index is None else sd.query_devices(index)
    host = sd.query_hostapis(info["hostapi"])["name"]
    idx = "default" if index is None else str(index)
    return f"[{idx}] {info['name']} ({host})"


def resolve_output_device(spec: int | str | None) -> int | None:
    """Resolve a config value to a device index.

    None means "pick the lowest-latency host API offering the system default
    output", not "hand it to PortAudio", because PortAudio's default on Windows
    is MME and MME's ~192 ms buffer makes the §7.3 flush budget unreachable.
    """
    if spec is None:
        return pick_low_latency_output()
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str) and spec.strip().isdigit():
        return int(spec.strip())
    needle = str(spec).lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and needle in d["name"].lower():
            return i
    raise ValueError(f"no output device matching {spec!r}")


def _base_name(name: str) -> str:
    """Device names differ slightly across host APIs; compare on a stable prefix."""
    return name.split("(")[0].strip().lower()[:24]


def pick_low_latency_output() -> int | None:
    """Find the system default output exposed on the fastest available host API."""
    try:
        default_index = sd.default.device[1]
        default_name = _base_name(sd.query_devices(default_index)["name"])
    except Exception:
        return None

    devices = list(enumerate(sd.query_devices()))
    for api in PREFERRED_HOST_APIS:
        for index, info in devices:
            if info["max_output_channels"] <= 0:
                continue
            if sd.query_hostapis(info["hostapi"])["name"] != api:
                continue
            if _base_name(info["name"]) != default_name:
                continue
            try:
                sd.check_output_settings(
                    device=index,
                    samplerate=int(info["default_samplerate"]),
                    channels=1,
                    dtype="float32",
                )
            except Exception:
                continue
            return index
    return default_index
