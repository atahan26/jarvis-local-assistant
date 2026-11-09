"""Kokoro-82M text to speech, CPU, sentence-chunked.

Spec §7.3 forbids synthesizing a whole response into one blob: the audio queue
has to stay flushable inside 60 ms. So text is split into speakable chunks and
each is synthesized separately.

kokoro-onnx does expose `create_stream()`, but it chunks internally and does not
report which text produced which audio. §7.3 also requires knowing exactly what
the user HEARD before an interruption, so chunking is done here instead, where
the text/audio mapping is preserved.

Spec §11: synthesis must never run on the event loop. Callers submit to a thread
pool; this module is deliberately synchronous and thread-safe to construct.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Sentence-final punctuation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Fallback split points inside an over-long sentence, in order of preference.
_CLAUSE_END = re.compile(r"(?<=[;:,])\s+")

# Stage directions and bracketed asides must never reach the speaker. The system
# prompt forbids them, but a local model will still produce "[interrupted by
# user]", "*laughs*" or "(pause)" now and then, and Kokoro will happily read them
# out character by character.
_ASIDE = re.compile(r"\[[^\]]{0,80}\]|\*[^*\n]{0,80}\*|\([^)\n]{0,40}\)")


def strip_asides(text: str) -> str:
    """Remove bracketed asides that should not be spoken."""
    return " ".join(_ASIDE.sub(" ", text).split())


@dataclass
class SpeechChunk:
    """One synthesized piece, with the text that produced it."""

    text: str
    samples: np.ndarray
    sample_rate: int
    synth_ms: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def _split_head(piece: str, limit: int) -> tuple[str, str]:
    """Cut `piece` at the latest clause or word boundary at or before `limit`."""
    if len(piece) <= limit:
        return piece, ""

    window = piece[: limit + 1]
    # Prefer a clause boundary, so the opening chunk ends somewhere natural.
    for punctuation in (",", ";", ":"):
        cut = window.rfind(punctuation)
        if cut > limit // 3:
            return piece[: cut + 1].strip(), piece[cut + 1 :].strip()

    cut = window.rfind(" ")
    if cut <= 0:
        return piece, ""
    return piece[:cut].strip(), piece[cut:].strip()


def split_for_speech(text: str, max_chars: int = 180, first_chunk_chars: int = 60) -> list[str]:
    """Split text into speakable chunks.

    The first chunk is kept deliberately short: time-to-first-audio is what the
    user perceives as responsiveness (§3 budgets 150 ms), and a short opening
    clause reaches the speaker much sooner than a full paragraph would.
    """
    text = " ".join(text.split())
    if not text:
        return []

    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]

    # Break any sentence that is too long on its own.
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        clauses = [c for c in _CLAUSE_END.split(sentence) if c.strip()]
        buffer = ""
        for clause in clauses:
            if len(clause) > max_chars:
                # Still too long: fall back to a hard word-boundary wrap.
                if buffer:
                    pieces.append(buffer.strip())
                    buffer = ""
                words, line = clause.split(), ""
                for word in words:
                    if len(line) + len(word) + 1 > max_chars:
                        pieces.append(line.strip())
                        line = word
                    else:
                        line = f"{line} {word}".strip()
                if line:
                    buffer = line
            elif len(buffer) + len(clause) + 1 > max_chars:
                pieces.append(buffer.strip())
                buffer = clause
            else:
                buffer = f"{buffer} {clause}".strip()
        if buffer:
            pieces.append(buffer.strip())

    # The opening piece can still be longer than the first-chunk limit, because
    # the loop above only enforces max_chars. Break it once more so the very
    # first synthesis really is short; this is the whole point of the limit.
    if pieces and len(pieces[0]) > first_chunk_chars:
        head, tail = _split_head(pieces[0], first_chunk_chars)
        if tail:
            pieces = [head, tail, *pieces[1:]]

    # Merge adjacent short pieces, but keep the first chunk small.
    chunks: list[str] = []
    for piece in pieces:
        # The limit applies to the chunk being merged INTO, so while we are still
        # extending chunks[0] the tighter first-chunk limit governs. Using
        # max_chars here would glue a short opening sentence onto a long one and
        # throw away the fast-first-audio win entirely.
        limit = first_chunk_chars if len(chunks) == 1 else max_chars
        if chunks and len(chunks[-1]) + len(piece) + 1 <= limit:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return [c for c in chunks if c.strip()]


class StreamingChunker:
    """Turns a stream of LLM tokens into speakable chunks as early as possible.

    Waiting for the full response before synthesizing would add the LLM's entire
    generation time to the first spoken word. Instead a chunk is emitted the
    moment a sentence completes, so Kokoro starts working while the model is
    still writing.

    The first chunk is held to a tighter limit than the rest, for the same
    reason split_for_speech does it: first audio is what the user perceives.
    """

    def __init__(self, max_chars: int = 180, first_chunk_chars: int = 60):
        self.max_chars = max_chars
        self.first_chunk_chars = first_chunk_chars
        self._buffer = ""
        self._emitted = 0

    @property
    def _limit(self) -> int:
        return self.first_chunk_chars if self._emitted == 0 else self.max_chars

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any chunks that are now ready to speak."""
        self._buffer += text
        ready: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                break
            ready.append(chunk)
            self._emitted += 1
        return ready

    def flush(self) -> list[str]:
        """Emit whatever is left once the stream has ended."""
        remainder = " ".join(self._buffer.split())
        self._buffer = ""
        if not remainder:
            return []
        chunks = split_for_speech(
            remainder,
            max_chars=self.max_chars,
            first_chunk_chars=self._limit,
        )
        self._emitted += len(chunks)
        return chunks

    def _take(self) -> str | None:
        buffer = self._buffer.lstrip()
        if not buffer:
            self._buffer = buffer
            return None

        limit = self._limit
        match = _SENTENCE_END.search(buffer)
        if match:
            candidate = buffer[: match.start()].strip()
            rest = buffer[match.end() :]
            if len(candidate) <= limit:
                self._buffer = rest
                return candidate
            head, tail = _split_head(candidate, limit)
            if tail:
                self._buffer = f"{tail} {rest}"
                return head
            self._buffer = rest
            return candidate

        # No complete sentence yet. Only cut early if the buffer already exceeds
        # the limit, otherwise wait for more tokens rather than chopping a
        # sentence in half mid-thought.
        if len(buffer) > limit:
            head, tail = _split_head(buffer, limit)
            if tail:
                self._buffer = tail
                return head

        self._buffer = buffer
        return None


class KokoroTTS:
    def __init__(
        self,
        model_path: Path | str,
        voices_path: Path | str,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
        max_chunk_chars: int = 180,
        first_chunk_chars: int = 60,
    ):
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(model_path), str(voices_path))
        # onnxruntime sessions are not guaranteed re-entrant; serialize calls so
        # a thread pool cannot corrupt state.
        self._lock = threading.Lock()
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.max_chunk_chars = max_chunk_chars
        self.first_chunk_chars = first_chunk_chars
        self.sample_rate = 24000

    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())

    def split(self, text: str) -> list[str]:
        return split_for_speech(
            text, max_chars=self.max_chunk_chars, first_chunk_chars=self.first_chunk_chars
        )

    def synthesize(self, text: str) -> SpeechChunk:
        text = strip_asides(text)
        if not text:
            return SpeechChunk(
                text="", samples=np.zeros(0, dtype=np.float32), sample_rate=self.sample_rate
            )
        started = time.monotonic()
        with self._lock:
            samples, sample_rate = self._kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.lang
            )
        self.sample_rate = sample_rate
        return SpeechChunk(
            text=text,
            samples=np.asarray(samples, dtype=np.float32),
            sample_rate=sample_rate,
            synth_ms=(time.monotonic() - started) * 1000.0,
        )

    def warm_up(self) -> float:
        """First synthesis pays ONNX graph setup. Do it before the user is waiting."""
        started = time.monotonic()
        self.synthesize("Ready.")
        return (time.monotonic() - started) * 1000.0
