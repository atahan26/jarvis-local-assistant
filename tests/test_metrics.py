"""Metrics log.

Regression coverage for a crash that reached the user: `event()` took `kind` as
its first parameter, and the VAD call site passed a field also called `kind`,
so every speech_start raised
`TypeError: event() got multiple values for argument 'kind'`.

Every other test drove the turn handler directly and never went through the
capture loop, so nothing exercised the on_event callback. These tests cover the
event plumbing itself, which is what was actually broken.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_metrics -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.audio.vad import TurnEvent
from src.metrics import MetricsLog, TurnRecord


class TestEventPlumbing(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "metrics.jsonl"
        self.log = MetricsLog(self.path)

    def tearDown(self):
        self.log.close()
        self.dir.cleanup()

    def records(self) -> list[dict]:
        self.log._handle.flush()
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_a_field_named_kind_does_not_collide(self):
        """The exact call that crashed at runtime."""
        self.log.event("vad", kind="speech_start", detail="conf=0.85")
        record = self.records()[0]
        self.assertEqual(record["kind"], "vad")

    def test_reserved_fields_cannot_overwrite_event_identity(self):
        self.log.event("vad", kind="impostor", t=999)
        record = self.records()[0]
        self.assertEqual(record["kind"], "vad")
        self.assertNotEqual(record["t"], 999)
        self.assertIn("_shadowed", record)

    def test_the_real_vad_call_site_shape(self):
        ev = TurnEvent(at_mono=1.0, kind="speech_start", detail="conf=0.9", confidence=0.9)
        self.log.event("vad", event=ev.kind, detail=ev.detail, confidence=round(ev.confidence, 3))
        record = self.records()[0]
        self.assertEqual(record["kind"], "vad")
        self.assertEqual(record["event"], "speech_start")
        self.assertEqual(record["confidence"], 0.9)

    def test_every_conversation_call_site_is_writable(self):
        """Guards against another signature collision slipping through."""
        calls = [
            ("state", {"from": "listening", "to": "thinking"}),
            ("vad", {"event": "speech_start", "detail": "", "confidence": 0.8}),
            ("smart_turn", {"complete": True, "latency_ms": 180.0, "error": None}),
            ("barge_in", {"reason": "sustained", "speech_ms": 352.0, "flush_ms": 22.0,
                          "spoken": "hello", "dropped": "world"}),
            ("backchannel", {"speech_ms": 250.0}),
            ("backchannel_turn", {"text": "mhm"}),
            ("turn_error", {"error": "boom"}),
            ("session_start", {"voice": "af_heart", "model": "qwen"}),
        ]
        for name, fields in calls:
            with self.subTest(event=name):
                self.log.event(name, **fields)
        self.assertEqual(len(self.records()), len(calls))

    def test_add_turn_writes_a_record(self):
        turn = TurnRecord(index=1, transcript="hello", reply="hi", stt_ms=600.0,
                          ttft_ms=140.0, first_audio_ms=800.0)
        self.log.add_turn(turn)
        record = self.records()[0]
        self.assertEqual(record["kind"], "turn")
        self.assertEqual(record["transcript"], "hello")
        self.assertAlmostEqual(record["round_trip_ms"], 1540.0, places=1)

    def test_disabled_log_writes_nothing_and_does_not_crash(self):
        log = MetricsLog(None, enabled=False)
        log.event("vad", kind="speech_start")
        log.add_turn(TurnRecord(index=1))
        self.assertEqual(len(log.turns), 1)
        log.close()


class TestSummary(unittest.TestCase):
    def test_summary_without_turns_still_reports_barge_in_counters(self):
        log = MetricsLog(None, enabled=False)
        log.backchannels_ignored = 3
        lines = "\n".join(log.summary_lines())
        self.assertIn("backchannels ignored 3", lines)
        log.close()

    def test_false_barge_in_rate_is_reported(self):
        log = MetricsLog(None, enabled=False)
        log.barge_ins = 8
        log.vetoed = 1
        log.backchannels_ignored = 1
        lines = "\n".join(log.summary_lines())
        self.assertIn("false barge-in rate", lines)
        log.close()

    def test_turn_gap_includes_the_silence_wait(self):
        turn = TurnRecord(index=1, stt_ms=600.0, ttft_ms=140.0, first_audio_ms=800.0,
                          smart_turn_ms=180.0, silence_wait_ms=700.0)
        self.assertAlmostEqual(turn.round_trip_ms, 1540.0, places=1)
        self.assertAlmostEqual(turn.gap_ms, 2420.0, places=1)


if __name__ == "__main__":
    unittest.main()
