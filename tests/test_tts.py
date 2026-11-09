"""Sentence chunking and the flushable playback queue (spec §7.3).

The queue tests drive SpeakerOutput's callback directly with a numpy buffer
instead of opening a real audio device, so they run headless and deterministically.
That is the right level: §7.3 is about queue semantics, and the audible check is
`--speak --flush-test`.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_tts -v
"""

from __future__ import annotations

import unittest

import numpy as np

from src import config as config_mod
from src.audio.output import SpeakerOutput, _prefix_by_fraction
from src.services.kokoro_tts import split_for_speech, strip_asides

CFG = config_mod.load()
SR = 24000


def tone(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def pump(speaker: SpeakerOutput, frames: int) -> np.ndarray:
    """Run one callback's worth of audio, as PortAudio would."""
    out = np.zeros((frames, 1), dtype=np.float32)
    speaker._callback(out, frames, None, None)
    return out


class TestChunking(unittest.TestCase):
    def test_empty_text_yields_nothing(self):
        self.assertEqual(split_for_speech(""), [])
        self.assertEqual(split_for_speech("   "), [])

    def test_single_short_sentence_is_one_chunk(self):
        self.assertEqual(split_for_speech("Volume is now fifty percent."),
                         ["Volume is now fifty percent."])

    def test_first_chunk_is_kept_short_for_fast_first_audio(self):
        text = (
            "Sure thing. I have turned the volume down to about fifty percent, "
            "and I also paused the music that was playing in the background."
        )
        chunks = split_for_speech(text, max_chars=180, first_chunk_chars=90)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), 90)

    def test_no_chunk_exceeds_the_limit(self):
        text = " ".join(f"This is sentence number {i}." for i in range(40))
        for chunk in split_for_speech(text, max_chars=120):
            self.assertLessEqual(len(chunk), 120)

    def test_a_very_long_clauseless_sentence_is_still_split(self):
        text = "word " * 300
        chunks = split_for_speech(text, max_chars=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_first_chunk_limit_is_enforced_even_on_a_long_opening_sentence(self):
        """A long first sentence must still be cut, or first-audio latency blows up."""
        text = (
            "This is a deliberately long opening sentence that keeps going for a "
            "while, because the point is to interrupt it somewhere in the middle."
        )
        chunks = split_for_speech(text, max_chars=180, first_chunk_chars=60)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), 60)

    def test_head_split_prefers_a_clause_boundary(self):
        text = "Sure thing boss, I have gone ahead and turned the volume all the way down for you now."
        chunks = split_for_speech(text, max_chars=180, first_chunk_chars=40)
        self.assertTrue(chunks[0].endswith(","), f"expected a clause boundary, got {chunks[0]!r}")

    def test_no_words_are_lost(self):
        text = (
            "First sentence here. Second one, with a comma, keeps going; and a "
            "semicolon too. Third and final sentence!"
        )
        chunks = split_for_speech(text, max_chars=60)
        self.assertEqual(" ".join(chunks).split(), text.split())


class TestStripAsides(unittest.TestCase):
    """Nothing bracketed may reach the speaker.

    Reported from a real session: the assistant read "[interrupted by user]"
    aloud. Root cause was an in-band history marker the model learned to copy,
    but a local model will produce stage directions on its own too, so the TTS
    path refuses them regardless.
    """

    def test_square_bracket_marker_is_removed(self):
        self.assertEqual(
            strip_asides("The Ottoman Empire lasted centuries. [interrupted by user]"),
            "The Ottoman Empire lasted centuries.",
        )

    def test_asterisk_stage_direction_is_removed(self):
        self.assertEqual(strip_asides("Sure thing *laughs* here you go"),
                         "Sure thing here you go")

    def test_short_parenthetical_is_removed(self):
        self.assertEqual(strip_asides("Okay (pause) moving on"), "Okay moving on")

    def test_ordinary_text_is_untouched(self):
        text = "The capital of France is Paris."
        self.assertEqual(strip_asides(text), text)

    def test_text_that_is_only_an_aside_becomes_empty(self):
        self.assertEqual(strip_asides("[interrupted by user]"), "")

    def test_long_parenthetical_prose_is_kept(self):
        """Only short asides are stripped; real parenthetical content survives."""
        text = ("Rome fell in 476 (at least in the west, since the eastern half "
                "carried on for another thousand years) and that was that.")
        self.assertIn("eastern half", strip_asides(text))


class TestSpokenPrefix(unittest.TestCase):
    def test_fraction_maps_to_words(self):
        self.assertEqual(_prefix_by_fraction("one two three four", 0.0), "")
        self.assertEqual(_prefix_by_fraction("one two three four", 1.0), "one two three four")
        self.assertEqual(_prefix_by_fraction("one two three four", 0.5), "one two")

    def test_full_fraction_returns_text_untouched(self):
        text = "exactly this, punctuation and all."
        self.assertEqual(_prefix_by_fraction(text, 1.0), text)


class TestPlaybackQueue(unittest.TestCase):
    def make(self) -> SpeakerOutput:
        # device_sample_rate is pinned so these stay deterministic. Left to
        # autodetect they inherit whatever the machine's default output runs at
        # (48 kHz here), which silently resamples every fixture and changes how
        # many frames a pump() call consumes.
        return SpeakerOutput(sample_rate=SR, blocksize=256, device_sample_rate=SR)

    def test_queue_reports_duration_and_chunk_count(self):
        s = self.make()
        s.enqueue("one", tone(1.0))
        s.enqueue("two", tone(0.5))
        self.assertEqual(s.queued_chunks, 2)
        self.assertAlmostEqual(s.queued_seconds, 1.5, places=2)
        self.assertTrue(s.is_speaking)

    def test_flush_empties_the_queue(self):
        s = self.make()
        s.enqueue("one", tone(1.0))
        s.enqueue("two", tone(1.0))
        result = s.flush()
        self.assertEqual(s.queued_chunks, 0)
        self.assertFalse(s.is_speaking)
        self.assertEqual(result.dropped_chunks, 2)

    def test_flush_is_far_inside_the_60ms_budget(self):
        s = self.make()
        for i in range(50):
            s.enqueue(f"chunk {i}", tone(0.5))
        result = s.flush()
        self.assertLess(result.latency_ms, 60.0)

    def test_completed_chunks_count_as_spoken(self):
        s = self.make()
        s.enqueue("hello there", tone(0.01))   # 240 samples
        pump(s, 256)                            # drains it completely
        self.assertEqual(s.spoken_text(), "hello there")

    def test_unplayed_chunks_are_not_spoken(self):
        """§7.3: text generated but never played must NOT enter history."""
        s = self.make()
        s.enqueue("heard this", tone(0.01))
        s.enqueue("never heard this", tone(5.0))
        pump(s, 256)
        result = s.flush()
        self.assertIn("heard this", result.spoken_text)
        self.assertNotIn("never heard", result.spoken_text)
        self.assertIn("never heard this", result.dropped_text)

    def test_partially_played_chunk_is_split_proportionally(self):
        s = self.make()
        # 8 words over 1.0s; play roughly half, expect roughly half the words.
        s.enqueue("one two three four five six seven eight", tone(1.0))
        pump(s, SR // 2)
        result = s.flush()
        spoken_words = result.spoken_text.split()
        self.assertGreater(len(spoken_words), 2)
        self.assertLess(len(spoken_words), 7)
        self.assertTrue(result.spoken_text.startswith("one"))

    def test_silence_is_emitted_when_nothing_is_queued(self):
        s = self.make()
        out = pump(s, 256)
        self.assertTrue(np.all(out == 0.0))

    def test_playback_continues_across_chunk_boundaries(self):
        s = self.make()
        s.enqueue("a", np.ones(300, dtype=np.float32))
        s.enqueue("b", np.ones(300, dtype=np.float32))
        out = pump(s, 512)
        # All 512 frames should be filled from the two chunks, not zero-padded.
        self.assertTrue(np.all(out[:512, 0] == 1.0))

    def test_reset_spoken_clears_history(self):
        s = self.make()
        s.enqueue("hello", tone(0.01))
        pump(s, 256)
        self.assertTrue(s.spoken_text())
        s.reset_spoken()
        self.assertEqual(s.spoken_text(), "")

    def test_flush_reports_device_buffer_in_the_audible_figure(self):
        """§7.3 is judged on when sound stops, not when the queue empties."""
        s = self.make()
        s.enqueue("something", tone(5.0))
        result = s.flush()
        self.assertAlmostEqual(
            result.audible_stop_ms,
            result.latency_ms + result.device_latency_ms,
            places=6,
        )
        self.assertGreaterEqual(result.audible_stop_ms, result.latency_ms)


class TestResampling(unittest.TestCase):
    def test_audio_is_resampled_to_the_device_rate(self):
        s = SpeakerOutput(sample_rate=24000, blocksize=256, device_sample_rate=48000)
        s.enqueue("x", np.zeros(24000, dtype=np.float32))   # 1.0 s at 24 kHz
        self.assertAlmostEqual(s.queued_seconds, 1.0, places=2)

    def test_no_resampling_when_rates_match(self):
        s = SpeakerOutput(sample_rate=24000, blocksize=256, device_sample_rate=24000)
        s.enqueue("x", np.zeros(24000, dtype=np.float32))
        self.assertAlmostEqual(s.queued_seconds, 1.0, places=2)


@unittest.skipUnless(CFG.tts_model.exists(), "Kokoro model files not present")
class TestKokoroSynthesis(unittest.TestCase):
    def test_synthesizes_audio_at_24k(self):
        from src.services.kokoro_tts import KokoroTTS

        tts = KokoroTTS(CFG.tts_model, CFG.tts_voices, voice=str(CFG.get("tts.voice")))
        chunk = tts.synthesize("Volume is now fifty percent.")
        self.assertEqual(chunk.sample_rate, 24000)
        self.assertGreater(chunk.duration_s, 0.5)
        self.assertEqual(chunk.samples.dtype, np.float32)
        self.assertGreater(float(np.abs(chunk.samples).max()), 0.01, "audio is silent")

    def test_configured_voice_exists(self):
        from src.services.kokoro_tts import KokoroTTS

        tts = KokoroTTS(CFG.tts_model, CFG.tts_voices)
        self.assertIn(str(CFG.get("tts.voice")), tts.voices())


if __name__ == "__main__":
    unittest.main()
