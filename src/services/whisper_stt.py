"""whisper.cpp HTTP client.

Spec §5 wraps whisper over HTTP deliberately, to avoid compiling Python bindings
on Windows. The server exposes POST /inference taking a multipart WAV.

Note this is the CPU/OpenBLAS build: upstream ships no Windows Vulkan binary.
Whisper's encoder always runs over a full 30 s mel window, so transcription cost
is roughly FLAT regardless of how short the utterance is. `whisper.audio_ctx` in
config.yaml shrinks that window and is the main latency lever available here.
"""

from __future__ import annotations

import io
import re
import time
import wave
from dataclasses import dataclass

import httpx

# whisper.cpp emits bracketed annotations for non-speech. They are not words and
# must never reach the LLM or a transcript the user sees.
_NON_SPEECH = re.compile(r"\[(BLANK_AUDIO|SILENCE|MUSIC|SOUND|NOISE|INAUDIBLE)[^\]]*\]", re.I)
_BRACKETED = re.compile(r"[\(\[\*][^\)\]\*]{0,40}[\)\]\*]")


@dataclass
class Transcript:
    text: str
    latency_ms: float
    audio_s: float
    raw: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def realtime_factor(self) -> float:
        return (self.latency_ms / 1000.0) / self.audio_s if self.audio_s > 0 else 0.0


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def clean_transcript(text: str) -> str:
    text = _NON_SPEECH.sub(" ", text)
    text = _BRACKETED.sub(" ", text)
    # whisper splits on sentence boundaries with newlines; a turn is one line.
    text = " ".join(text.split())
    return text.strip()


class WhisperClient:
    # whisper caps the initial prompt at 224 tokens and starts hallucinating if
    # it is stuffed, so the vocabulary hint is kept to a short list.
    MAX_PROMPT_CHARS = 600

    def __init__(
        self,
        base_url: str,
        *,
        language: str = "en",
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        vocabulary: list[str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout_s)
        self.prompt = ""
        self.set_vocabulary(vocabulary or [])

    def set_vocabulary(self, words: list[str]) -> None:
        """Bias decoding toward names whisper would otherwise mangle.

        Measured: "My name is Atahan" came back as "My name is Adehan", and
        "Hey Jarvis, Atahan here" as "Hey Jarvis, add a hand here". With the
        name in the initial prompt, both transcribe correctly. Proper nouns are
        exactly what a general model gets wrong, so app names and the user's own
        name are worth spending prompt budget on.
        """
        seen, cleaned = set(), []
        for word in words:
            word = " ".join(str(word).split())
            key = word.lower()
            if word and key not in seen:
                seen.add(key)
                cleaned.append(word)
        prompt = ", ".join(cleaned)
        self.prompt = (prompt[: self.MAX_PROMPT_CHARS].rsplit(",", 1)[0]
                       if len(prompt) > self.MAX_PROMPT_CHARS else prompt)
        if self.prompt:
            self.prompt += "."

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WhisperClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def is_up(self) -> bool:
        try:
            self._client.get(f"{self.base_url}/", timeout=2.0)
            return True
        except httpx.HTTPError:
            return False

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        audio_s = len(pcm) / (sample_rate * 2)
        wav = pcm_to_wav(pcm, sample_rate)

        payload = {
            "response_format": "json",
            "temperature": str(self.temperature),
            "language": self.language,
        }
        if self.prompt:
            payload["prompt"] = self.prompt

        started = time.monotonic()
        response = self._client.post(
            f"{self.base_url}/inference",
            files={"file": ("turn.wav", wav, "audio/wav")},
            data=payload,
        )
        response.raise_for_status()
        latency_ms = (time.monotonic() - started) * 1000.0

        raw = str(response.json().get("text", ""))
        return Transcript(
            text=clean_transcript(raw),
            latency_ms=latency_ms,
            audio_s=audio_s,
            raw=raw.strip(),
        )
