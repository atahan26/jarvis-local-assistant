"""End-to-end VAD + endpointing + whisper, without a human at the microphone.

Kokoro synthesizes an utterance, it is padded with silence and pushed through the
real Silero VAD frame by frame, and the resulting turn is transcribed. This
verifies the whole M1 chain minus the sounddevice capture itself.

Skipped automatically when whisper-server is not running.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_vad_live -v
"""

from __future__ import annotations

import unittest

import numpy as np

from src import config as config_mod
from src.audio.io import SILERO_FRAME_SAMPLES
from src.audio.vad import Endpointer, VoiceActivityDetector
from src.services.whisper_stt import WhisperClient

CFG = config_mod.load()
SR = 16000


def _whisper_up() -> bool:
    with WhisperClient(CFG.whisper_url) as c:
        return c.is_up()


def _kokoro_available() -> bool:
    return CFG.tts_model.exists() and CFG.tts_voices.exists()


def synth(text: str, trim: bool = True) -> bytes:
    """Return 16 kHz mono int16 PCM for `text`.

    Kokoro pads its output with roughly 75 ms of leading and 150 ms of trailing
    near-silence. Left in place, concatenating two clips around a "600 ms" gap
    actually produces an ~830 ms gap, which legitimately ends the turn and makes
    the mid-sentence-pause test measure the wrong thing. Trim by default so the
    gap in a fixture is the gap under test.
    """
    import soxr
    from kokoro_onnx import Kokoro

    global _KOKORO
    try:
        kokoro = _KOKORO
    except NameError:
        kokoro = _KOKORO = Kokoro(str(CFG.tts_model), str(CFG.tts_voices))
    samples, sr = kokoro.create(text, voice=str(CFG.get("tts.voice")), speed=1.0, lang="en-us")
    resampled = np.asarray(soxr.resample(samples, sr, SR), dtype=np.float32)

    if trim:
        amplitude = np.abs(resampled)
        loud = np.flatnonzero(amplitude > amplitude.max() * 0.02)
        if loud.size:
            resampled = resampled[loud[0] : loud[-1] + 1]

    return np.clip(resampled * 32767, -32768, 32767).astype("<i2").tobytes()


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(SR * ms / 1000)


def to_frames(pcm: bytes) -> list[bytes]:
    n = SILERO_FRAME_SAMPLES * 2
    usable = (len(pcm) // n) * n
    return [pcm[i : i + n] for i in range(0, usable, n)]


@unittest.skipUnless(_kokoro_available(), "Kokoro model files not present")
class TestSileroOnRealSpeech(unittest.TestCase):
    def make_endpointer(self) -> Endpointer:
        vad = VoiceActivityDetector(
            sample_rate=SR, confidence=float(CFG.get("turn.vad_threshold"))
        )
        return Endpointer(
            vad,
            vad_threshold=float(CFG.get("turn.vad_threshold")),
            base_silence_ms=int(CFG.get("turn.base_silence_ms")),
            max_wait_ms=int(CFG.get("turn.max_wait_ms")),
            min_speech_ms=int(CFG.get("turn.min_speech_ms")),
            frame_ms=1000.0 * SILERO_FRAME_SAMPLES / SR,
        )

    def test_silence_produces_no_turn(self):
        ep = self.make_endpointer()
        got = [u for u in (ep.process(f) for f in to_frames(silence(3000))) if u is not None]
        self.assertEqual(got, [], "Silero fired on pure silence")

    def test_speech_produces_exactly_one_turn(self):
        ep = self.make_endpointer()
        pcm = silence(400) + synth("Turn the volume down please.") + silence(1200)
        got = [u for u in (ep.process(f) for f in to_frames(pcm)) if u is not None]
        self.assertEqual(len(got), 1, f"expected 1 turn, got {len(got)}")
        self.assertGreater(got[0].duration_s, 0.5)

    def test_mid_sentence_pause_does_not_split(self):
        """§10: a 600 ms pause mid-thought must not end the turn."""
        ep = self.make_endpointer()
        pcm = (
            silence(400)
            + synth("I was thinking that maybe")
            + silence(600)
            + synth("we could try it tomorrow.")
            + silence(1200)
        )
        got = [u for u in (ep.process(f) for f in to_frames(pcm)) if u is not None]
        self.assertEqual(len(got), 1, "the 600ms pause split one turn into two")

    def test_pause_longer_than_the_threshold_does_split(self):
        """The complement: 900 ms of real silence should end the turn."""
        ep = self.make_endpointer()
        pcm = (
            silence(400)
            + synth("I was thinking that maybe")
            + silence(900)
            + synth("we could try it tomorrow.")
            + silence(1200)
        )
        got = [u for u in (ep.process(f) for f in to_frames(pcm)) if u is not None]
        self.assertEqual(len(got), 2, "a 900ms gap should have ended the first turn")


@unittest.skipUnless(_kokoro_available(), "Kokoro model files not present")
@unittest.skipUnless(_whisper_up(), "whisper-server not running; start it with .\\run.ps1")
class TestFullChain(unittest.TestCase):
    def test_endpointed_turn_transcribes_correctly(self):
        vad = VoiceActivityDetector(sample_rate=SR, confidence=float(CFG.get("turn.vad_threshold")))
        ep = Endpointer(
            vad,
            vad_threshold=float(CFG.get("turn.vad_threshold")),
            base_silence_ms=int(CFG.get("turn.base_silence_ms")),
            min_speech_ms=int(CFG.get("turn.min_speech_ms")),
            frame_ms=1000.0 * SILERO_FRAME_SAMPLES / SR,
        )
        phrase = "Set a timer for ten minutes."
        pcm = silence(400) + synth(phrase) + silence(1200)
        turns = [u for u in (ep.process(f) for f in to_frames(pcm)) if u is not None]
        self.assertEqual(len(turns), 1)

        with WhisperClient(CFG.whisper_url) as stt:
            result = stt.transcribe_pcm(turns[0].audio, SR)

        self.assertFalse(result.is_empty, "whisper returned nothing")
        words = set(result.text.lower().replace(".", "").split())
        self.assertIn("timer", words, f"got: {result.text!r}")
        self.assertIn("minutes", words, f"got: {result.text!r}")

    def test_vocabulary_biases_transcription_of_proper_nouns(self):
        """Whisper mangles names it has no reason to expect.

        Measured: "My name is Atahan" -> "My name is Adehan" without a hint,
        correct with one. This is why config carries a vocabulary list.
        """
        pcm = synth("My name is Atahan.")
        with WhisperClient(CFG.whisper_url) as plain:
            without = plain.transcribe_pcm(pcm, SR).text
        with WhisperClient(CFG.whisper_url, vocabulary=["Atahan", "Jarvis"]) as biased:
            with_hint = biased.transcribe_pcm(pcm, SR).text
        self.assertIn("atahan", with_hint.lower(), f"hint ignored, got {with_hint!r}")
        # The point of the feature is that it changes something.
        if "atahan" in without.lower():
            self.skipTest("whisper got the name right unaided; nothing to prove")

    def test_empty_vocabulary_sends_no_prompt(self):
        client = WhisperClient(CFG.whisper_url, vocabulary=[])
        self.assertEqual(client.prompt, "")
        client.close()

    def test_vocabulary_is_deduplicated_and_bounded(self):
        client = WhisperClient(CFG.whisper_url, vocabulary=["Atahan", "atahan", "Jarvis"])
        self.assertEqual(client.prompt.lower().count("atahan"), 1)
        client.set_vocabulary([f"Name{i}" for i in range(500)])
        self.assertLessEqual(len(client.prompt), WhisperClient.MAX_PROMPT_CHARS + 1)
        client.close()

    def test_non_speech_annotations_are_stripped(self):
        with WhisperClient(CFG.whisper_url) as stt:
            result = stt.transcribe_pcm(silence(1500), SR)
        self.assertNotIn("[", result.text)
        self.assertTrue(result.is_empty, f"silence produced text: {result.text!r}")


if __name__ == "__main__":
    unittest.main()
