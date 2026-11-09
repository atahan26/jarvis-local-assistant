"""Wake word detection and the follow-up window (spec §7.5).

The detector tests use synthesized speech, so they check the thing that actually
matters: "hey jarvis" scores far above unrelated speech, and the refractory
period stops one utterance from waking it repeatedly.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_wake_word -v
"""

from __future__ import annotations

import unittest

import numpy as np

from src import config as config_mod
from src.audio.wake_word import (
    MODEL_FILES,
    is_wake_phrase_only,
    model_path_for,
    strip_wake_phrase,
)

CFG = config_mod.load()
SR = 16000
FRAME = 512
WAKE_DIR = CFG.models_dir / str(CFG.get("wake_word.models_dir", "openwakeword"))


def _wake_models_present() -> bool:
    name = str(CFG.get("wake_word.model", "hey_jarvis"))
    return (
        model_path_for(WAKE_DIR, name).exists()
        and (WAKE_DIR / "melspectrogram.onnx").exists()
        and (WAKE_DIR / "embedding_model.onnx").exists()
    )


class TestPaths(unittest.TestCase):
    def test_known_models_map_to_onnx_files(self):
        for name in MODEL_FILES:
            self.assertTrue(model_path_for(WAKE_DIR, name).name.endswith(".onnx"))

    def test_unknown_name_falls_back_to_onnx_suffix(self):
        self.assertEqual(model_path_for(WAKE_DIR, "custom").name, "custom.onnx")

    def test_models_are_not_inside_the_venv(self):
        """A venv rebuild must not delete the wake models."""
        self.assertNotIn(".venv", str(WAKE_DIR).lower())


@unittest.skipUnless(_wake_models_present(), "wake word models not downloaded; run .\\setup.ps1")
@unittest.skipUnless(CFG.tts_model.exists(), "Kokoro model files not present")
class TestDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kokoro_onnx import Kokoro

        from src.audio.wake_word import WakeWordDetector

        cls.kokoro = Kokoro(str(CFG.tts_model), str(CFG.tts_voices))
        cls.detector = WakeWordDetector(
            WAKE_DIR,
            name=str(CFG.get("wake_word.model", "hey_jarvis")),
            threshold=float(CFG.get("wake_word.threshold", 0.5)),
            refractory_ms=int(CFG.get("wake_word.refractory_ms", 2000)),
        )

    def speech(self, text: str, voice: str = "am_michael") -> np.ndarray:
        import soxr

        samples, sr = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        resampled = np.asarray(soxr.resample(samples, sr, SR), dtype=np.float32)
        return np.clip(resampled * 32767, -32768, 32767).astype(np.int16)

    def peak_score(self, audio: np.ndarray) -> float:
        self.detector.reset()
        best = 0.0
        for i in range(0, len(audio) - FRAME, FRAME):
            self.detector.process(audio[i : i + FRAME].tobytes())
            best = max(best, self.detector.last_score)
        return best

    def fires_on(self, audio: np.ndarray) -> int:
        self.detector.reset()
        count = 0
        for i in range(0, len(audio) - FRAME, FRAME):
            if self.detector.process(audio[i : i + FRAME].tobytes()) is not None:
                count += 1
        return count

    def test_wake_phrase_is_detected(self):
        self.assertEqual(self.fires_on(self.speech("Hey Jarvis")), 1)

    def test_wake_phrase_detected_in_another_voice(self):
        self.assertGreaterEqual(self.peak_score(self.speech("Hey Jarvis", "af_heart")), 0.5)

    def test_wake_phrase_at_the_start_of_a_command(self):
        self.assertGreaterEqual(
            self.peak_score(self.speech("Hey Jarvis, what's the weather?")), 0.5
        )

    def test_unrelated_speech_does_not_fire(self):
        for phrase in ["What is the capital of Japan?", "Turn the volume down", "Hey there"]:
            with self.subTest(phrase=phrase):
                self.assertEqual(self.fires_on(self.speech(phrase)), 0)

    def test_silence_does_not_fire(self):
        self.assertEqual(self.fires_on(np.zeros(SR * 3, dtype=np.int16)), 0)

    def test_refractory_period_prevents_repeat_firing(self):
        """The score stays high for several frames after the phrase."""
        audio = np.concatenate(
            [self.speech("Hey Jarvis"), np.zeros(SR // 2, dtype=np.int16)]
        )
        self.assertEqual(self.fires_on(audio), 1, "one utterance woke it more than once")

    def test_reset_clears_the_refractory_timer(self):
        """Regression: reset() cleared model state but not the fire timestamp.

        A short interaction (wake, quick exchange, window expires) left the
        detector deaf to the next "hey jarvis" for the rest of the refractory
        period. It also made these tests order-dependent.
        """
        wake = self.speech("Hey Jarvis")
        self.assertEqual(self.fires_on(wake), 1)
        # Immediately again, with only a reset in between: must still fire.
        self.assertEqual(self.fires_on(wake), 1, "reset() did not clear the refractory")


class TestTranscriptCleanup(unittest.TestCase):
    """The wake audio is replayed into the turn, so it lands in the transcript.

    Reported by the user: "hey jarvis, what do you think about the Roman Empire"
    arrived as "about the Roman Empire" because the pre-wake audio was discarded.
    Replaying it fixes the loss but puts the wake phrase at the front.
    """

    def test_leading_wake_phrase_is_removed(self):
        for said, expected in [
            ("Hey Jarvis, what do you think about the Roman Empire?",
             "what do you think about the Roman Empire?"),
            ("hey jarvis what is the capital of France?",
             "what is the capital of France?"),
            ("Hey Jarvis. Turn the volume down.", "Turn the volume down."),
            ("Hey, Jarvis, play some music", "play some music"),
        ]:
            with self.subTest(said=said):
                self.assertEqual(strip_wake_phrase(said), expected)

    def test_transcript_without_the_wake_phrase_is_untouched(self):
        text = "And what about Germany?"
        self.assertEqual(strip_wake_phrase(text), text)

    def test_wake_phrase_is_not_removed_from_the_middle(self):
        text = "Tell me why they called it hey jarvis"
        self.assertEqual(strip_wake_phrase(text), text)

    def test_bare_wake_phrase_is_recognised_as_content_free(self):
        for said in ["Hey Jarvis", "hey jarvis.", "Hey, Jarvis!", "  hey jarvis  "]:
            with self.subTest(said=said):
                self.assertTrue(is_wake_phrase_only(said))

    def test_wake_phrase_with_a_request_is_not_content_free(self):
        for said in ["Hey Jarvis, hello", "hey jarvis what time is it", "Hey Jarvis stop"]:
            with self.subTest(said=said):
                self.assertFalse(is_wake_phrase_only(said))

    def test_stripping_never_returns_an_empty_string(self):
        """Callers print this; an empty transcript would be a silent no-op."""
        self.assertTrue(strip_wake_phrase("Hey Jarvis"))


class TestFollowUpWindowConfig(unittest.TestCase):
    def test_window_is_configured_and_positive(self):
        self.assertGreater(int(CFG.get("turn.follow_up_window_ms")), 0)

    def test_spec_default_is_eight_seconds(self):
        self.assertEqual(int(CFG.get("turn.follow_up_window_ms")), 8000)


if __name__ == "__main__":
    unittest.main()
