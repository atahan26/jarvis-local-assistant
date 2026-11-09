"""Voice activity detection and endpointing.

Silero VAD answers "is there speech in this 32 ms frame". Endpointing answers
"has the user finished their turn", which is a different and much harder
question, and it is where spec §7.2 says previous attempts failed.

pipecat's own VADAnalyzer state machine is deliberately NOT used for the turn
decision: its ``stop_secs`` is a single fixed silence threshold, whereas §7.2
requires a 700 ms base that a turn-completeness model can extend to a 3000 ms
ceiling. SileroVADAnalyzer is used purely as the ONNX confidence source.

The SmartTurn hook is present but inert until M5.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


class TurnState(Enum):
    QUIET = "quiet"
    SPEAKING = "speaking"
    TRAILING = "trailing"  # speech stopped, waiting to see whether it resumes


@dataclass
class Utterance:
    """A completed turn, ready for STT."""

    audio: bytes
    sample_rate: int
    speech_start_mono: float
    speech_end_mono: float
    endpoint_mono: float
    extended: bool = False
    frames: int = 0

    @property
    def duration_s(self) -> float:
        return len(self.audio) / (self.sample_rate * 2)

    @property
    def silence_wait_ms(self) -> float:
        """How long we waited after the last speech before calling the turn."""
        return (self.endpoint_mono - self.speech_end_mono) * 1000.0


@dataclass
class TurnEvent:
    """Timestamped state transition, for the M5 debugging surface (§7.6)."""

    at_mono: float
    kind: str
    detail: str = ""
    confidence: float = 0.0


class VoiceActivityDetector:
    """Thin wrapper over Silero that also reports frame energy."""

    def __init__(self, sample_rate: int = 16000, confidence: float = 0.5):
        self._analyzer = SileroVADAnalyzer(
            sample_rate=sample_rate,
            params=VADParams(confidence=confidence),
        )
        # The constructor argument only populates _init_sample_rate; the analyzer's
        # real _sample_rate stays 0 until set_sample_rate() runs, which normally
        # happens inside a pipecat transport. Skipping this makes the ONNX model
        # reject every frame with "Supported sampling rates: [8000, 16000]" while
        # voice_confidence() swallows the exception and returns 0.0, so the VAD
        # silently never fires. Do not remove.
        self._analyzer.set_sample_rate(sample_rate)
        self.sample_rate = sample_rate

        expected = 512 if sample_rate == 16000 else 256
        if self._analyzer.num_frames_required() != expected:
            raise RuntimeError(
                "Silero VAD did not accept the sample rate: "
                f"num_frames_required()={self._analyzer.num_frames_required()}, "
                f"expected {expected} at {sample_rate} Hz"
            )

    @property
    def frame_samples(self) -> int:
        return self._analyzer.num_frames_required()

    def confidence(self, frame: bytes) -> float:
        # voice_confidence() returns the raw ONNX output, which is a shape-(1,)
        # array rather than a scalar. numpy 2.x refuses float() on any array with
        # ndim > 0, so unwrap it explicitly instead.
        value = self._analyzer.voice_confidence(frame)
        return float(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def dbfs(frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return 20.0 * math.log10(rms) if rms > 1e-9 else -120.0


class Endpointer:
    """Decides when a turn is over (spec §7.2).

    Feed it one frame at a time. It returns an Utterance on the frame where the
    turn completes, and None otherwise.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        *,
        vad_threshold: float = 0.5,
        base_silence_ms: int = 700,
        max_wait_ms: int = 3000,
        min_speech_ms: int = 200,
        preroll_ms: int = 320,
        frame_ms: float = 32.0,
        probe_interval_ms: int = 300,
        on_event: Callable[[TurnEvent], None] | None = None,
    ):
        self.vad = vad
        self.vad_threshold = vad_threshold
        self.base_silence_ms = base_silence_ms
        self.max_wait_ms = max_wait_ms
        self.min_speech_ms = min_speech_ms
        self.frame_ms = frame_ms
        # The turn-completeness probe is expensive (SmartTurn measured at
        # 125-200 ms) and runs on the capture thread. Probing on every frame
        # means 125 ms of inference per 32 ms of audio, i.e. roughly 4x
        # real time: the microphone queue backs up, frames are dropped, and the
        # silence counter drifts badly behind the wall clock. Observed in a real
        # session as a 9.3 s wait on a turn whose ceiling was 3 s.
        self.probe_interval_ms = probe_interval_ms
        self._last_probe_at_ms = -1.0
        self._on_event = on_event

        # Pre-roll keeps the audio immediately before speech was confirmed, so
        # the first phoneme is not clipped off the front of the transcript.
        self._preroll = deque(maxlen=max(1, int(round(preroll_ms / frame_ms))))

        self.state = TurnState.QUIET
        self._buffer = bytearray()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_start_mono = 0.0
        self._last_voiced_mono = 0.0
        self._frames = 0
        self._extended = False

        # A hook for M5. Returning False means "the user is not finished", which
        # extends the wait up to max_wait_ms.
        self.turn_complete_probe: Callable[[bytes], bool] | None = None

    # -- events ------------------------------------------------------------

    def _emit(self, kind: str, detail: str = "", confidence: float = 0.0) -> None:
        if self._on_event:
            self._on_event(TurnEvent(time.monotonic(), kind, detail, confidence))

    # -- main entry point --------------------------------------------------

    def process(self, frame: bytes) -> Utterance | None:
        now = time.monotonic()
        conf = self.vad.confidence(frame)
        voiced = conf >= self.vad_threshold

        if self.state is TurnState.QUIET:
            self._preroll.append(frame)
            if voiced:
                self.state = TurnState.SPEAKING
                self._buffer = bytearray(b"".join(self._preroll))
                self._preroll.clear()
                self._speech_ms = self.frame_ms
                self._silence_ms = 0.0
                self._speech_start_mono = now
                self._last_voiced_mono = now
                self._frames = 1
                self._extended = False
                self._emit("speech_start", f"conf={conf:.2f}", conf)
            return None

        # SPEAKING or TRAILING: everything gets buffered, including the trailing
        # silence, because whisper transcribes better with a little tail.
        self._buffer.extend(frame)
        self._frames += 1

        if voiced:
            self._speech_ms += self.frame_ms
            self._last_voiced_mono = now
            if self.state is TurnState.TRAILING:
                self._emit("speech_resumed", f"after {self._silence_ms:.0f}ms", conf)
                self.state = TurnState.SPEAKING
            self._silence_ms = 0.0
            # Speech resumed, so the next silence gets a fresh probe as soon as
            # it crosses base_silence rather than inheriting the old schedule.
            self._last_probe_at_ms = -1.0
            return None

        # Unvoiced frame.
        self._silence_ms += self.frame_ms
        if self.state is TurnState.SPEAKING:
            self.state = TurnState.TRAILING
            self._emit("speech_pause", f"conf={conf:.2f}", conf)

        waited = self._silence_ms
        if waited < self.base_silence_ms:
            return None

        # Base silence elapsed. Ask whether the turn is actually complete, but
        # rate-limited: see probe_interval_ms above.
        if self.turn_complete_probe is not None and waited < self.max_wait_ms:
            due = (
                self._last_probe_at_ms < 0
                or (waited - self._last_probe_at_ms) >= self.probe_interval_ms
            )
            if not due:
                return None
            self._last_probe_at_ms = waited
            if not self.turn_complete_probe(bytes(self._buffer)):
                if not self._extended:
                    self._extended = True
                    self._emit("turn_extended", f"waited={waited:.0f}ms")
                return None

        # Turn is over. Discard anything too short to be a real utterance.
        if self._speech_ms < self.min_speech_ms:
            self._emit("discarded", f"only {self._speech_ms:.0f}ms of speech")
            self._reset()
            return None

        utt = Utterance(
            audio=bytes(self._buffer),
            sample_rate=self.vad.sample_rate,
            speech_start_mono=self._speech_start_mono,
            speech_end_mono=self._last_voiced_mono,
            endpoint_mono=now,
            extended=self._extended,
            frames=self._frames,
        )
        self._emit("turn_complete", f"{utt.duration_s:.2f}s audio, waited {waited:.0f}ms")
        self._reset()
        return utt

    def reset(self) -> None:
        """Discard any turn in progress.

        Used after a barge-in, where the endpointer must be re-primed with the
        audio captured while the assistant was still talking.
        """
        self._reset()

    def _reset(self) -> None:
        self.state = TurnState.QUIET
        self._buffer = bytearray()
        self._preroll.clear()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._frames = 0
        self._extended = False
        self._last_probe_at_ms = -1.0
