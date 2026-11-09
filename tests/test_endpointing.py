"""Endpointing state machine (spec §7.2).

These use a scripted fake VAD rather than Silero so the turn logic is tested
deterministically and without loading an ONNX model. The real Silero path is
covered by tests/test_vad_live.py, which needs the servers up.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest

from src.audio.vad import Endpointer, TurnState

FRAME_MS = 32.0
SAMPLES = 512
FRAME = b"\x00\x00" * SAMPLES


class FakeVAD:
    """Returns confidences from a script, one per process() call."""

    sample_rate = 16000

    def __init__(self, script: list[float]):
        self.script = list(script)
        self.calls = 0

    def confidence(self, frame: bytes) -> float:
        value = self.script[self.calls] if self.calls < len(self.script) else 0.0
        self.calls += 1
        return value


def frames(*, speech: int = 0, silence: int = 0) -> list[float]:
    return [0.9] * speech + [0.05] * silence


def ms(n: int) -> int:
    """Frames needed to cover n milliseconds."""
    return int(round(n / FRAME_MS))


def drive(ep: Endpointer, count: int):
    """Feed `count` frames, returning the first utterance produced."""
    out = None
    for _ in range(count):
        result = ep.process(FRAME)
        if result is not None and out is None:
            out = result
    return out


def build(script: list[float], **kwargs) -> tuple[Endpointer, FakeVAD]:
    vad = FakeVAD(script)
    defaults = dict(
        vad_threshold=0.5,
        base_silence_ms=700,
        max_wait_ms=3000,
        min_speech_ms=200,
        frame_ms=FRAME_MS,
        preroll_ms=320,
    )
    defaults.update(kwargs)
    return Endpointer(vad, **defaults), vad


class TestBasicTurn(unittest.TestCase):
    def test_silence_alone_produces_nothing(self):
        ep, _ = build(frames(silence=100))
        self.assertIsNone(drive(ep, 100))
        self.assertIs(ep.state, TurnState.QUIET)

    def test_speech_then_silence_completes_a_turn(self):
        speech, silence = ms(1000), ms(900)
        ep, _ = build(frames(speech=speech, silence=silence))
        utt = drive(ep, speech + silence)
        self.assertIsNotNone(utt)
        self.assertIs(ep.state, TurnState.QUIET)
        self.assertGreater(utt.duration_s, 1.0)

    def test_turn_does_not_fire_before_base_silence(self):
        # 600ms of silence is under the 700ms threshold: still waiting.
        speech, silence = ms(1000), ms(600)
        ep, _ = build(frames(speech=speech, silence=silence))
        self.assertIsNone(drive(ep, speech + silence))
        self.assertIs(ep.state, TurnState.TRAILING)

    def test_preroll_is_prepended_so_the_first_word_survives(self):
        # 10 frames of silence, then speech. Pre-roll should capture some of it.
        speech, lead = ms(1000), 10
        ep, _ = build(frames(silence=lead) + frames(speech=speech, silence=ms(900)))
        utt = drive(ep, lead + speech + ms(900))
        self.assertIsNotNone(utt)
        captured_frames = len(utt.audio) // (SAMPLES * 2)
        # More frames than the speech+trailing alone, i.e. pre-roll made it in.
        self.assertGreater(captured_frames, speech + ms(900))


class TestMidSentencePause(unittest.TestCase):
    """§10: a 600ms mid-sentence pause must NOT end the turn."""

    def test_600ms_pause_does_not_split_the_turn(self):
        script = (
            frames(speech=ms(800))
            + frames(silence=ms(600))     # the pause under test
            + frames(speech=ms(800))
            + frames(silence=ms(900))     # real end of turn
        )
        ep, _ = build(script)
        utterances = [u for u in (ep.process(FRAME) for _ in script) if u is not None]
        self.assertEqual(len(utterances), 1, "the 600ms pause split one turn into two")

    def test_800ms_pause_does_end_the_turn(self):
        script = (
            frames(speech=ms(800))
            + frames(silence=ms(800))
            + frames(speech=ms(800))
            + frames(silence=ms(900))
        )
        ep, _ = build(script)
        utterances = [u for u in (ep.process(FRAME) for _ in script) if u is not None]
        self.assertEqual(len(utterances), 2)


class TestShortNoise(unittest.TestCase):
    def test_blip_shorter_than_min_speech_is_discarded(self):
        events = []
        script = frames(speech=ms(100)) + frames(silence=ms(900))
        ep, _ = build(script, on_event=events.append)
        self.assertIsNone(drive(ep, len(script)))
        self.assertIn("discarded", [e.kind for e in events])

    def test_speech_just_over_min_is_kept(self):
        script = frames(speech=ms(300)) + frames(silence=ms(900))
        ep, _ = build(script)
        self.assertIsNotNone(drive(ep, len(script)))


class TestTurnCompletionProbe(unittest.TestCase):
    """The M5 SmartTurn hook: extend the wait when the turn looks incomplete."""

    def test_incomplete_turn_extends_past_base_silence(self):
        script = frames(speech=ms(800)) + frames(silence=ms(900))
        ep, _ = build(script)
        ep.turn_complete_probe = lambda _audio: False   # never complete
        self.assertIsNone(drive(ep, len(script)))
        self.assertIs(ep.state, TurnState.TRAILING)

    def test_probe_is_overridden_at_the_max_wait_ceiling(self):
        script = frames(speech=ms(800)) + frames(silence=ms(3500))
        ep, _ = build(script)
        ep.turn_complete_probe = lambda _audio: False
        utt = drive(ep, len(script))
        self.assertIsNotNone(utt, "max_wait_ms ceiling did not force the turn to end")
        self.assertTrue(utt.extended)

    def test_complete_turn_fires_at_base_silence(self):
        script = frames(speech=ms(800)) + frames(silence=ms(900))
        ep, _ = build(script)
        ep.turn_complete_probe = lambda _audio: True
        utt = drive(ep, len(script))
        self.assertIsNotNone(utt)
        self.assertFalse(utt.extended)


class TestProbeRateLimit(unittest.TestCase):
    """The probe is expensive; calling it per frame overloads the capture loop.

    Regression: SmartTurn (125-200 ms) was invoked on every 32 ms frame, roughly
    4x real time. The mic queue backed up and a real session logged a 9.3 s wait
    against a 3 s ceiling.
    """

    def test_probe_is_not_called_on_every_frame(self):
        calls = []
        script = frames(speech=ms(500)) + frames(silence=ms(3000))
        ep, _ = build(script, probe_interval_ms=300)
        ep.turn_complete_probe = lambda audio: (calls.append(1), False)[1]
        drive(ep, len(script))

        silence_frames = ms(3000 - 700)          # frames past base silence
        self.assertLess(len(calls), silence_frames / 2, "probe ran far too often")
        # ~2300ms of probing at 300ms intervals is roughly 8 calls.
        self.assertLessEqual(len(calls), 12)
        self.assertGreaterEqual(len(calls), 4)

    def test_probe_still_fires_promptly_at_base_silence(self):
        calls = []
        script = frames(speech=ms(500)) + frames(silence=ms(800))
        ep, _ = build(script, probe_interval_ms=300)
        ep.turn_complete_probe = lambda audio: (calls.append(1), True)[1]
        utt = drive(ep, len(script))
        self.assertEqual(len(calls), 1, "should probe once, immediately at base silence")
        self.assertIsNotNone(utt)

    def test_max_wait_ceiling_is_still_honoured_with_rate_limiting(self):
        script = frames(speech=ms(500)) + frames(silence=ms(4000))
        ep, _ = build(script, probe_interval_ms=300)
        ep.turn_complete_probe = lambda audio: False
        utt = drive(ep, len(script))
        self.assertIsNotNone(utt, "max_wait_ms ceiling did not fire")
        self.assertTrue(utt.extended)

    def test_resumed_speech_resets_the_probe_schedule(self):
        calls = []
        script = (
            frames(speech=ms(400))
            + frames(silence=ms(800))     # probes once here
            + frames(speech=ms(400))      # resumes
            + frames(silence=ms(800))     # must probe again promptly
        )
        ep, _ = build(script, probe_interval_ms=300)
        ep.turn_complete_probe = lambda audio: (calls.append(1), False)[1]
        drive(ep, len(script))
        self.assertGreaterEqual(len(calls), 2, "second silence never probed")


class TestConsecutiveTurns(unittest.TestCase):
    def test_state_resets_cleanly_between_turns(self):
        one = frames(speech=ms(500)) + frames(silence=ms(900))
        ep, _ = build(one * 3)
        got = [u for u in (ep.process(FRAME) for _ in one * 3) if u is not None]
        self.assertEqual(len(got), 3)
        for utt in got:
            self.assertGreater(utt.duration_s, 0.4)
            self.assertLess(utt.duration_s, 2.5, "buffer leaked across turns")


if __name__ == "__main__":
    unittest.main()
