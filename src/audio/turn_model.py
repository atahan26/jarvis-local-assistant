"""SmartTurn: "has the user finished their turn?" (spec §7.2).

Wraps pipecat's LocalSmartTurnAnalyzerV3, a bundled 8.7 MB ONNX model that runs
on CPU. It is used only as a probe: when the endpointer's base silence elapses,
it decides whether to fire now or keep waiting up to the max_wait ceiling.

Two measured facts drive this wrapper:

* `set_sample_rate()` must be called explicitly. Like SileroVADAnalyzer, the
  constructor argument only sets `_init_sample_rate` and the effective rate
  stays 0 until a pipecat transport would call the setter. We are not a
  transport, so we call it, then assert.
* Inference costs 156-203 ms on this machine. That lands directly in the
  turn-taking gap, so it runs once per endpoint decision, never per frame.

What it does NOT do, despite spec §3: classify backchannels. Measured, it labels
"yeah", "okay", "right" and "actually stop" all COMPLETE, because it predicts
grammatical completeness rather than intent. Backchannel handling lives in
barge_in.py instead.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class TurnVerdict:
    complete: bool
    latency_ms: float
    error: str | None = None


class SmartTurnModel:
    def __init__(self, sample_rate: int = 16000, *, cpu_count: int = 1):
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        self._analyzer = LocalSmartTurnAnalyzerV3(cpu_count=cpu_count)
        # Without this the effective sample rate is 0 and the model sees garbage.
        self._analyzer.set_sample_rate(sample_rate)
        if self._analyzer.sample_rate != sample_rate:
            raise RuntimeError(
                f"SmartTurn sample rate is {self._analyzer.sample_rate}, expected {sample_rate}"
            )
        self.sample_rate = sample_rate
        self._loop = asyncio.new_event_loop()

    def append(self, frame: bytes, is_speech: bool) -> None:
        """Accumulate a frame. Cheap; safe to call from the capture loop."""
        self._analyzer.append_audio(frame, is_speech)

    def clear(self) -> None:
        self._analyzer.clear()

    def close(self) -> None:
        try:
            self._loop.close()
        except Exception:
            pass

    def is_complete(self) -> TurnVerdict:
        """Run the model over everything appended so far.

        pipecat exposes only an async entry point, which internally just hands
        the work to an executor. A private event loop is kept so this can be
        called from a plain worker thread without an asyncio runtime.
        """
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        started = time.monotonic()
        try:
            state, _metrics = self._loop.run_until_complete(
                self._analyzer.analyze_end_of_turn()
            )
        except Exception as exc:  # noqa: BLE001
            # Never let a model failure wedge the conversation: fall back to
            # "the turn is over", which is the behaviour without SmartTurn.
            return TurnVerdict(
                complete=True,
                latency_ms=(time.monotonic() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return TurnVerdict(
            complete=state == EndOfTurnState.COMPLETE,
            latency_ms=(time.monotonic() - started) * 1000.0,
        )

    def warm_up(self) -> float:
        """First inference pays ONNX setup; do it before a user is waiting."""
        silence = b"\x00\x00" * (self.sample_rate // 2)
        self.append(silence, True)
        verdict = self.is_complete()
        self.clear()
        return verdict.latency_ms
