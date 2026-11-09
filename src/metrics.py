"""Per-turn metrics and the timestamped event log (spec §7.6, §12).

§12: "Log aggressively during M5. Timestamped events for VAD state changes, turn
decisions, flush calls, and cancel calls are the only way to debug turn-taking."

Events go to logs/metrics.jsonl as one JSON object per line, with a monotonic
timestamp relative to session start so a session can be replayed and reasoned
about after the fact.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TurnRecord:
    """One complete user turn, from endpoint decision to end of reply."""

    index: int
    transcript: str = ""
    reply: str = ""
    stt_ms: float = 0.0
    ttft_ms: float = 0.0
    first_audio_ms: float = 0.0
    smart_turn_ms: float = 0.0
    turn_extended: bool = False
    silence_wait_ms: float = 0.0
    interrupted: bool = False
    spoken_text: str = ""
    dropped_text: str = ""
    chunks: int = 0
    tools_used: int = 0

    @property
    def round_trip_ms(self) -> float:
        return self.stt_ms + self.ttft_ms + self.first_audio_ms

    @property
    def gap_ms(self) -> float:
        """Turn-taking gap: user stops speaking to assistant starts (§7.6)."""
        return self.silence_wait_ms + self.smart_turn_ms + self.round_trip_ms


class MetricsLog:
    def __init__(self, path: Path | None = None, *, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._handle = None
        if enabled and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

        self.turns: list[TurnRecord] = []
        self.flush_latencies_ms: list[float] = []
        self.cancel_latencies_ms: list[float] = []
        self.onset_to_silence_ms: list[float] = []
        self.barge_ins = 0
        self.backchannels_ignored = 0
        self.vetoed = 0
        self.self_interruptions = 0

    # -- event stream ------------------------------------------------------

    def event(self, event_type: str, /, **fields: Any) -> None:
        """Write one event.

        `event_type` is positional-only so a caller passing a field literally
        named `kind` or `event_type` cannot collide with it. Fields are also
        prevented from overwriting the reserved keys, which would otherwise
        silently rewrite an event's identity in the log.
        """
        if not self.enabled or self._handle is None:
            return
        safe = {k: v for k, v in fields.items() if k not in ("t", "kind")}
        if len(safe) != len(fields):
            safe["_shadowed"] = sorted(set(fields) - set(safe))
        record = {
            "t": round(time.monotonic() - self._started, 4),
            "kind": event_type,
            **safe,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.event("turn", **asdict(turn), round_trip_ms=round(turn.round_trip_ms, 1))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # -- summary -----------------------------------------------------------

    @staticmethod
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * p))))
        return ordered[k]

    def summary_lines(self) -> list[str]:
        out: list[str] = []
        if self.turns:
            out.append(f"  {len(self.turns)} turns")
            out.append("")
            out.append(f"  {'':<26}{'median':>10}{'p95':>9}{'max':>9}")
            for label, values in [
                ("whisper STT", [t.stt_ms for t in self.turns]),
                ("SmartTurn decision", [t.smart_turn_ms for t in self.turns]),
                ("LLM first token", [t.ttft_ms for t in self.turns]),
                ("TTS first audio", [t.first_audio_ms for t in self.turns]),
                ("ROUND TRIP", [t.round_trip_ms for t in self.turns]),
                ("turn-taking gap", [t.gap_ms for t in self.turns]),
            ]:
                if not any(values):
                    continue
                out.append(
                    f"  {label:<26}{statistics.median(values):>8.0f}ms"
                    f"{self._pct(values, 0.95):>8.0f}ms{max(values):>8.0f}ms"
                )
            out.append("")

        # §7.6 barge-in metrics.
        attempts = self.barge_ins + self.backchannels_ignored
        out.append("  barge-in")
        out.append(f"    confirmed            {self.barge_ins}")
        out.append(f"    backchannels ignored {self.backchannels_ignored}")
        out.append(f"    vetoed by transcript {self.vetoed}")
        if attempts:
            false_rate = 100.0 * self.vetoed / attempts
            out.append(f"    false barge-in rate  {false_rate:.1f}%  (target < 2%)")
        if self.self_interruptions:
            out.append(f"    SELF-INTERRUPTIONS   {self.self_interruptions}  (echo leaking into the mic)")
        if self.flush_latencies_ms:
            out.append(
                f"    TTS flush            median {statistics.median(self.flush_latencies_ms):.1f}ms"
                f"  p95 {self._pct(self.flush_latencies_ms, 0.95):.1f}ms  (budget 60ms)"
            )
        if self.cancel_latencies_ms:
            out.append(
                f"    LLM cancel           median {statistics.median(self.cancel_latencies_ms):.1f}ms"
                f"  p95 {self._pct(self.cancel_latencies_ms, 0.95):.1f}ms  (budget 40ms)"
            )
        if self.onset_to_silence_ms:
            out.append(
                f"    you spoke -> silence median "
                f"{statistics.median(self.onset_to_silence_ms):.0f}ms"
                f"  p95 {self._pct(self.onset_to_silence_ms, 0.95):.0f}ms"
            )
        return out
