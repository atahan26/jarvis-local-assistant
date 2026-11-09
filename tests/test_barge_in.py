"""Barge-in policy and the §10 acceptance table.

Spec §10: "Barge-in is either right or it isn't, and it is not testable by vibes.
Write these as scripted tests first (same phrase, same timestamp, same expected
cancel behaviour), then validate with unscripted human testing."

These are the scripted half. Frames are fed synthetically so timing is exact and
a regression shows up as a failing assertion rather than a bad feeling.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_barge_in -v
"""

from __future__ import annotations

import unittest

from src.audio.barge_in import BargeInPolicy, BargeInState, is_backchannel, normalize_phrase

FRAME_MS = 32.0
PHRASES = ["mhm", "mm", "uh huh", "yeah", "yep", "okay", "ok", "right", "sure", "got it"]


def frames_for(ms: float) -> int:
    return int(round(ms / FRAME_MS))


def build(**kwargs) -> BargeInPolicy:
    defaults = dict(confirm_ms=350, vad_threshold=0.5, frame_ms=FRAME_MS)
    defaults.update(kwargs)
    return BargeInPolicy(**defaults)


def speak(policy: BargeInPolicy, ms: float, confidence: float = 0.9, dbfs: float = -20.0):
    """Feed `ms` of speech, returning the last decision."""
    decision = None
    for _ in range(frames_for(ms)):
        decision = policy.process(confidence, dbfs)
    return decision


def quiet(policy: BargeInPolicy, ms: float):
    """Feed `ms` of silence, returning the last decision."""
    decision = None
    for _ in range(frames_for(ms)):
        decision = policy.process(0.02, -70.0)
    return decision


def quiet_decisions(policy: BargeInPolicy, ms: float) -> list:
    """Every decision across `ms` of silence.

    A backchannel is announced on exactly one frame, the one where the gap
    crosses max_gap_ms; frames after that report plain "quiet". Tests that care
    about the announcement have to look at all of them, not just the last.
    """
    return [policy.process(0.02, -70.0) for _ in range(frames_for(ms))]


class TestBackchannelClassifier(unittest.TestCase):
    def test_pure_backchannels(self):
        for word in ["mhm", "Yeah", "okay.", "right!", "uh huh", "got it", "OK"]:
            with self.subTest(word=word):
                self.assertTrue(is_backchannel(word, PHRASES), word)

    def test_repeated_backchannels(self):
        self.assertTrue(is_backchannel("yeah yeah", PHRASES))
        self.assertTrue(is_backchannel("mhm ok", PHRASES))

    def test_real_commands_are_not_backchannels(self):
        for phrase in [
            "actually stop",
            "wait no",
            "stop talking",
            "what is the weather",
            "okay open spotify",
            "no that is wrong",
        ]:
            with self.subTest(phrase=phrase):
                self.assertFalse(is_backchannel(phrase, PHRASES), phrase)

    def test_long_utterances_are_never_backchannels(self):
        self.assertFalse(is_backchannel("yeah yeah yeah sure okay right", PHRASES))

    def test_empty_transcript_counts_as_nothing_said(self):
        self.assertTrue(is_backchannel("", PHRASES))
        self.assertTrue(is_backchannel("  ...  ", PHRASES))

    def test_whisper_spelling_variants_of_the_same_noise(self):
        """A real session turned "Mm-hmm" into a whole turn.

        It normalizes to the two tokens "mm" and "hmm"; "hmm" was not in the
        list, so the all-tokens check failed and it was treated as a command.
        """
        from src import config as config_mod

        configured = list(config_mod.load().get("turn.backchannel_phrases"))
        for spelling in ["Mm-hmm", "mm hmm", "Mmhmm", "mhm", "Hmm", "uh-huh", "Yup"]:
            with self.subTest(spelling=spelling):
                self.assertTrue(
                    is_backchannel(spelling, configured),
                    f"{spelling!r} should be a backchannel",
                )

    def test_variants_do_not_swallow_real_commands(self):
        from src import config as config_mod

        configured = list(config_mod.load().get("turn.backchannel_phrases"))
        for phrase in ["okay stop", "right now play music", "sure open spotify"]:
            with self.subTest(phrase=phrase):
                self.assertFalse(is_backchannel(phrase, configured), phrase)

    def test_normalize(self):
        self.assertEqual(normalize_phrase("  Uh-huh!! "), "uh huh")


class TestAcceptanceTable(unittest.TestCase):
    """The rows of §10 that this layer is responsible for."""

    def test_mhm_while_assistant_speaking_does_not_interrupt(self):
        """§10: assistant CONTINUES speaking through a backchannel."""
        policy = build(confirm_ms=350)
        decision = speak(policy, 250)          # a short "mhm"
        self.assertIsNot(decision.state, BargeInState.CONFIRMED)

        decisions = quiet_decisions(policy, 300)   # they stop on their own
        self.assertTrue(
            any(d.reason.startswith("backchannel") for d in decisions),
            "the short utterance was never classified as a backchannel",
        )
        # Crucially, it never confirmed, so playback was never touched.
        self.assertFalse(any(d.state is BargeInState.CONFIRMED for d in decisions))
        self.assertIs(policy.state, BargeInState.IDLE)

    def test_sustained_speech_confirms_a_barge_in(self):
        """§10: 'actually, stop' must take the floor."""
        policy = build(confirm_ms=350)
        decision = speak(policy, 500)
        self.assertIs(decision.state, BargeInState.CONFIRMED)

    def test_confirmation_happens_close_to_the_threshold(self):
        policy = build(confirm_ms=350)
        elapsed = 0.0
        for _ in range(100):
            decision = policy.process(0.9, -20.0)
            elapsed += FRAME_MS
            if decision.state is BargeInState.CONFIRMED:
                break
        self.assertGreaterEqual(elapsed, 350)
        self.assertLess(elapsed, 350 + 2 * FRAME_MS)

    def test_a_brief_dip_mid_word_does_not_reset_confirmation(self):
        """Natural speech dips below threshold between syllables."""
        policy = build(confirm_ms=350)
        speak(policy, 200)
        quiet(policy, 64)      # two frames, under max_gap_ms
        decision = speak(policy, 200)
        self.assertIs(decision.state, BargeInState.CONFIRMED)

    def test_a_long_gap_does_reset(self):
        policy = build(confirm_ms=350)
        speak(policy, 200)
        quiet(policy, 300)     # well past max_gap_ms
        decision = speak(policy, 200)
        self.assertIsNot(decision.state, BargeInState.CONFIRMED)

    def test_silence_alone_never_confirms(self):
        policy = build()
        decision = quiet(policy, 5000)
        self.assertIs(decision.state, BargeInState.IDLE)

    def test_low_confidence_noise_never_confirms(self):
        policy = build(confirm_ms=350)
        decision = speak(policy, 3000, confidence=0.2)
        self.assertIs(decision.state, BargeInState.IDLE)


class TestSpeakerMode(unittest.TestCase):
    """§7.1 speaker fallback: every gate tightened, explicitly degraded."""

    def make(self) -> BargeInPolicy:
        return BargeInPolicy(
            confirm_ms=350,
            vad_threshold=0.5,
            frame_ms=FRAME_MS,
            echo_mode="speakers",
            speaker_vad_threshold=0.6,
            speaker_sustained_ms=250,
            speaker_energy_dbfs=-35.0,
        )

    def test_quiet_audio_is_rejected_even_when_vad_is_confident(self):
        """Assistant echo bleeding through speakers is quiet; real speech is not."""
        policy = self.make()
        decision = speak(policy, 2000, confidence=0.95, dbfs=-50.0)
        self.assertIs(decision.state, BargeInState.IDLE)

    def test_loud_confident_speech_still_confirms(self):
        policy = self.make()
        decision = speak(policy, 2000, confidence=0.95, dbfs=-20.0)
        self.assertIs(decision.state, BargeInState.CONFIRMED)

    def test_vad_threshold_is_raised_relative_to_headphone_mode(self):
        speakers = self.make()
        headphones = build()
        self.assertGreater(speakers.vad_threshold, headphones.vad_threshold)

    def test_confirm_window_is_never_shorter_than_the_sustained_requirement(self):
        policy = BargeInPolicy(
            confirm_ms=100,           # would be too twitchy on speakers
            frame_ms=FRAME_MS,
            echo_mode="speakers",
            speaker_sustained_ms=250,
        )
        self.assertGreaterEqual(policy.confirm_ms, 250)


class TestReset(unittest.TestCase):
    def test_reset_clears_progress(self):
        policy = build(confirm_ms=350)
        speak(policy, 200)
        policy.reset()
        self.assertIs(policy.state, BargeInState.IDLE)
        decision = speak(policy, 200)
        self.assertIsNot(decision.state, BargeInState.CONFIRMED)

    def test_confirmed_state_is_sticky_until_reset(self):
        policy = build(confirm_ms=350)
        speak(policy, 500)
        decision = quiet(policy, 1000)
        self.assertIs(decision.state, BargeInState.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
