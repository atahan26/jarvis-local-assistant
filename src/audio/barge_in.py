"""Barge-in policy: deciding when the user has taken the floor.

Spec §1 calls this the primary success criterion of the whole project, and §7.2
splits it into two failure modes that pull in opposite directions:

  * cutting the user off      -> handled by Endpointer (§7.2)
  * ignoring the user         -> handled here (§7.3)

The hard part is telling a backchannel ("mhm", "yeah") from a real interruption
within the 60 ms budget. Spec §3 assumed SmartTurn would do this. It does not:
measured directly, SmartTurn labels "yeah", "okay", "right" and even "actually
stop" as COMPLETE, because it predicts grammatical completeness, not intent.

So the primary discriminator is DURATION, which is cheap and available
immediately. A backchannel is a short noise that stops by itself (measured
0.4-0.8 s); a real interruption keeps going. Speech that ends before confirm_ms
never stops the audio at all, so the assistant genuinely "keeps talking" as §10
requires. Speech still going at confirm_ms is confirmed, and only then does the
60 ms flush clock start.

A transcript veto runs afterwards as a second line of defence. It cannot delay
the flush because by then the audio has already stopped; it only decides whether
to resume.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum


class BargeInState(Enum):
    IDLE = "idle"           # assistant speaking, no user speech
    CANDIDATE = "candidate"  # user speech detected, not yet confirmed
    CONFIRMED = "confirmed"  # user has the floor, audio should be cut


@dataclass
class BargeInDecision:
    state: BargeInState
    speech_ms: float = 0.0
    reason: str = ""
    confidence: float = 0.0


def normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def is_backchannel(text: str, phrases: list[str], max_words: int = 3) -> bool:
    """True when the transcript is nothing but acknowledgement noise.

    Deliberately strict: a false positive here means ignoring a real command, so
    anything with content words falls through to being a real interruption.
    """
    normalized = normalize_phrase(text)
    if not normalized:
        return True  # nothing was said; certainly not a command

    words = normalized.split()
    if len(words) > max_words:
        return False

    allowed = {normalize_phrase(p) for p in phrases}
    # Whole-phrase match first ("uh huh", "got it").
    if normalized in allowed:
        return True
    # Punctuation and spacing vary by transcriber: whisper writes the same noise
    # as "Mm-hmm", "mm hmm" or "mhm". Compare with separators removed so one
    # config entry covers all of them.
    compact = normalized.replace(" ", "")
    if compact in {p.replace(" ", "") for p in allowed}:
        return True
    # Otherwise every token must itself be a backchannel ("yeah yeah", "mhm ok").
    single_words = {p for p in allowed if " " not in p}
    return all(word in single_words for word in words)


class BargeInPolicy:
    """Tracks user speech while the assistant is talking."""

    def __init__(
        self,
        *,
        confirm_ms: int = 350,
        vad_threshold: float = 0.5,
        frame_ms: float = 32.0,
        echo_mode: str = "headphones",
        speaker_vad_threshold: float = 0.6,
        speaker_sustained_ms: int = 250,
        speaker_energy_dbfs: float = -35.0,
    ):
        self.confirm_ms = confirm_ms
        self.frame_ms = frame_ms
        self.echo_mode = echo_mode

        # §7.1: on speakers the mic hears the assistant's own output, so every
        # gate is tightened. This is not AEC; it trades responsiveness for not
        # interrupting itself, and it is explicitly a degraded mode.
        if echo_mode == "speakers":
            self.vad_threshold = speaker_vad_threshold
            self.confirm_ms = max(confirm_ms, speaker_sustained_ms)
            self.energy_floor_dbfs = speaker_energy_dbfs
        else:
            self.vad_threshold = vad_threshold
            self.energy_floor_dbfs = -120.0

        self.state = BargeInState.IDLE
        self._speech_ms = 0.0
        self._gap_ms = 0.0
        self._onset_mono = 0.0
        # A brief dip below threshold mid-word should not reset the timer.
        self.max_gap_ms = 150.0

    def reset(self) -> None:
        self.state = BargeInState.IDLE
        self._speech_ms = 0.0
        self._gap_ms = 0.0
        self._onset_mono = 0.0

    @property
    def onset_mono(self) -> float:
        return self._onset_mono

    def process(self, confidence: float, dbfs: float) -> BargeInDecision:
        """Feed one frame captured while the assistant is speaking."""
        voiced = confidence >= self.vad_threshold and dbfs >= self.energy_floor_dbfs

        if self.state is BargeInState.CONFIRMED:
            return BargeInDecision(self.state, self._speech_ms, "already confirmed", confidence)

        if voiced:
            if self.state is BargeInState.IDLE:
                self.state = BargeInState.CANDIDATE
                self._onset_mono = time.monotonic()
                self._speech_ms = 0.0
            self._gap_ms = 0.0
            self._speech_ms += self.frame_ms

            if self._speech_ms >= self.confirm_ms:
                self.state = BargeInState.CONFIRMED
                return BargeInDecision(
                    self.state,
                    self._speech_ms,
                    f"sustained {self._speech_ms:.0f}ms >= {self.confirm_ms}ms",
                    confidence,
                )
            return BargeInDecision(self.state, self._speech_ms, "accumulating", confidence)

        # Unvoiced frame.
        if self.state is BargeInState.CANDIDATE:
            self._gap_ms += self.frame_ms
            if self._gap_ms >= self.max_gap_ms:
                # Stopped on its own before confirming: a backchannel. The
                # assistant never paused, which is exactly what §10 asks for.
                spoken = self._speech_ms
                self.reset()
                return BargeInDecision(
                    BargeInState.IDLE,
                    spoken,
                    f"backchannel: {spoken:.0f}ms of speech, stopped on its own",
                )
        return BargeInDecision(self.state, self._speech_ms, "quiet")


@dataclass
class BargeInMetrics:
    """Spec §7.6 counters."""

    confirmed: int = 0
    backchannels_ignored: int = 0
    vetoed_by_transcript: int = 0
    flush_latencies_ms: list[float] = field(default_factory=list)
    cancel_latencies_ms: list[float] = field(default_factory=list)
    onset_to_silence_ms: list[float] = field(default_factory=list)

    @property
    def real_barge_ins(self) -> int:
        return self.confirmed - self.vetoed_by_transcript

    @property
    def false_barge_in_rate(self) -> float:
        total = self.confirmed + self.backchannels_ignored
        if total == 0:
            return 0.0
        return self.vetoed_by_transcript / total
