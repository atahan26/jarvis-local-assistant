"""The capture loop, driven by a fake microphone.

Why this exists: a crash reached the user because every other test called
Assistant._handle() directly and nothing ever executed _capture ->
endpointer.process -> _on_turn_event -> metrics.event. The turn handler was well
covered; the loop that feeds it was not covered at all.

A FakeMic satisfies the same contract as MicrophoneStream, so the real loop runs
unmodified with no audio hardware involved on the input side.

Needs both servers and the Kokoro models; skipped otherwise.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_capture_loop -v
"""

from __future__ import annotations

import threading
import time
import traceback
import unittest

import numpy as np

from src import config as config_mod
from src.audio.io import SILERO_FRAME_SAMPLES
from src.services.llm import LLMClient
from src.services.whisper_stt import WhisperClient

CFG = config_mod.load()
SR = 16000


def _servers_up() -> bool:
    with WhisperClient(CFG.whisper_url) as w, LLMClient(CFG.llm_url) as l:
        return w.is_up() and l.is_up()


def _models_present() -> bool:
    return CFG.tts_model.exists() and CFG.tts_voices.exists()


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(SR * ms / 1000)


class FakeMic:
    """Same contract as MicrophoneStream: yields int16 frames, never ends."""

    def __init__(self, pcm: bytes, realtime: bool = False):
        self.pcm = pcm
        self.realtime = realtime
        self.dropped_frames = 0
        self.overflow_count = 0

    def frames(self, timeout=None):
        n = SILERO_FRAME_SAMPLES * 2
        pace = (n / 2 / SR) if self.realtime else 0.0
        for i in range(0, (len(self.pcm) // n) * n, n):
            if pace:
                time.sleep(pace)
            yield self.pcm[i : i + n]
        while True:
            time.sleep(pace or 0.001)
            yield b"\x00" * n


@unittest.skipUnless(_models_present(), "Kokoro model files not present")
@unittest.skipUnless(_servers_up(), "servers not running; start them with .\\run.ps1")
class TestCaptureLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kokoro_onnx import Kokoro

        cls.kokoro = Kokoro(str(CFG.tts_model), str(CFG.tts_voices))

    def speech(self, text: str) -> bytes:
        import soxr

        samples, sr = self.kokoro.create(text, voice="am_michael", speed=1.0, lang="en-us")
        resampled = np.asarray(soxr.resample(samples, sr, SR), dtype=np.float32)
        amplitude = np.abs(resampled)
        loud = np.flatnonzero(amplitude > amplitude.max() * 0.02)
        resampled = resampled[loud[0] : loud[-1] + 1]
        return np.clip(resampled * 32767, -32768, 32767).astype("<i2").tobytes()

    def run_capture(self, pcm: bytes, wait_s: float = 60.0, awake: bool = True):
        """Run the real loop until a turn completes; return (assistant, error).

        Since M6 the assistant starts in IDLE behind the wake word. Most of these
        tests are about the capture loop rather than §7.5, so by default they
        start already awake. Pass awake=False to exercise the wake gate.
        """
        from src.conversation import Assistant, State

        assistant = Assistant(CFG)
        assistant.tts.warm_up()
        if awake:
            assistant._set_state(State.LISTENING)
        errors: list[str] = []

        def capture():
            try:
                assistant._capture(FakeMic(pcm, realtime=True))
            except Exception:
                errors.append(traceback.format_exc())

        assistant.speaker.start()
        worker = threading.Thread(target=assistant._worker_loop, daemon=True)
        worker.start()
        cap = threading.Thread(target=capture, daemon=True)
        cap.start()

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if errors or assistant.metrics.turns:
                break
            time.sleep(0.05)
        time.sleep(1.5)   # let the turn settle
        assistant._stop.set()
        assistant._cancel.set()
        time.sleep(0.2)
        assistant.speaker.close()
        assistant.close()
        return assistant, errors

    def test_capture_loop_survives_a_full_turn(self):
        """The exact path that crashed: speech_start -> on_event -> metrics."""
        pcm = silence(400) + self.speech("What is the capital of Japan?") + silence(2500)
        assistant, errors = self.run_capture(pcm)
        self.assertEqual(errors, [], f"capture loop raised:\n{errors[0] if errors else ''}")
        self.assertTrue(assistant.metrics.turns, "no turn completed")

    def test_transcript_and_reply_are_recorded(self):
        pcm = silence(400) + self.speech("What is the capital of Japan?") + silence(2500)
        assistant, errors = self.run_capture(pcm)
        self.assertEqual(errors, [])
        turn = assistant.metrics.turns[0]
        self.assertIn("japan", turn.transcript.lower())
        self.assertTrue(turn.reply.strip())
        self.assertGreater(turn.stt_ms, 0)
        self.assertGreater(turn.ttft_ms, 0)

    def test_silence_alone_produces_no_turn(self):
        assistant, errors = self.run_capture(silence(4000), wait_s=6.0)
        self.assertEqual(errors, [])
        self.assertEqual(assistant.metrics.turns, [])

    def test_speech_while_idle_is_ignored_without_the_wake_word(self):
        """§7.5: the wake word gates IDLE."""
        if not CFG.get("wake_word.enabled", True):
            self.skipTest("wake word disabled in config")
        pcm = silence(400) + self.speech("What is the capital of Japan?") + silence(2500)
        assistant, errors = self.run_capture(pcm, wait_s=12.0, awake=False)
        self.assertEqual(errors, [])
        self.assertEqual(
            assistant.metrics.turns, [], "answered a command while still asleep"
        )


if __name__ == "__main__":
    unittest.main()
