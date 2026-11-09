"""StreamingChunker: turning LLM tokens into speakable chunks early.

This is what lets audio start before the model has finished writing, so the
behaviour that matters is "emits as soon as a sentence completes" and "never
loses or reorders a word".

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_streaming_chunker -v
"""

from __future__ import annotations

import unittest

from src.services.kokoro_tts import StreamingChunker


def feed_tokens(chunker: StreamingChunker, text: str, size: int = 4) -> list[str]:
    """Push text through in small pieces, the way tokens actually arrive."""
    out: list[str] = []
    for i in range(0, len(text), size):
        out.extend(chunker.feed(text[i : i + size]))
    out.extend(chunker.flush())
    return out


class TestStreamingChunker(unittest.TestCase):
    def test_nothing_is_emitted_before_a_boundary(self):
        ch = StreamingChunker(max_chars=180, first_chunk_chars=60)
        self.assertEqual(ch.feed("The capital"), [])
        self.assertEqual(ch.feed(" of Japan"), [])

    def test_first_sentence_is_emitted_as_soon_as_it_completes(self):
        ch = StreamingChunker(max_chars=180, first_chunk_chars=60)
        self.assertEqual(ch.feed("Tokyo is the capital."), [])
        ready = ch.feed(" It has ")
        self.assertEqual(ready, ["Tokyo is the capital."])

    def test_flush_emits_the_tail(self):
        ch = StreamingChunker(max_chars=180, first_chunk_chars=60)
        ch.feed("No trailing punctuation here")
        self.assertEqual(ch.flush(), ["No trailing punctuation here"])

    def test_flush_on_empty_buffer_is_empty(self):
        ch = StreamingChunker()
        self.assertEqual(ch.flush(), [])

    def test_no_words_are_lost_or_reordered(self):
        text = (
            "Tokyo is the capital of Japan. It has about thirty seven million "
            "people in the wider area, which makes it the largest city on earth. "
            "Seoul is smaller, but still very large."
        )
        chunks = feed_tokens(StreamingChunker(max_chars=180, first_chunk_chars=60), text)
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_first_chunk_respects_the_tighter_limit(self):
        text = (
            "I have gone ahead and turned the volume down for you, and I also "
            "paused the music that was playing. Anything else?"
        )
        chunks = feed_tokens(StreamingChunker(max_chars=180, first_chunk_chars=60), text)
        self.assertLessEqual(len(chunks[0]), 60)

    def test_later_chunks_may_use_the_full_limit(self):
        text = "Short. " + ("word " * 60).strip() + "."
        chunks = feed_tokens(StreamingChunker(max_chars=180, first_chunk_chars=40), text)
        self.assertLessEqual(len(chunks[0]), 40)
        self.assertTrue(any(len(c) > 40 for c in chunks[1:]))

    def test_a_runaway_sentence_is_still_cut(self):
        """No sentence boundary must never mean no audio."""
        ch = StreamingChunker(max_chars=80, first_chunk_chars=40)
        emitted = []
        for _ in range(30):
            emitted.extend(ch.feed("word "))
        self.assertTrue(emitted, "chunker never emitted despite exceeding the limit")
        for chunk in emitted:
            self.assertLessEqual(len(chunk), 80)

    def test_no_chunk_exceeds_max_chars(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(30))
        chunks = feed_tokens(StreamingChunker(max_chars=120, first_chunk_chars=60), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 120)

    def test_no_empty_chunks(self):
        text = "One.  Two.   Three.    Four."
        chunks = feed_tokens(StreamingChunker(max_chars=180, first_chunk_chars=60), text)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_single_token_stream_still_works(self):
        text = "Yes."
        chunks = feed_tokens(StreamingChunker(), text, size=1)
        self.assertEqual(chunks, ["Yes."])

    def test_emits_earlier_than_waiting_for_the_whole_response(self):
        """The point of the class: first audio should not wait for the last token."""
        text = (
            "Sure thing. " + "This is a much longer follow up sentence that keeps going. " * 5
        )
        ch = StreamingChunker(max_chars=180, first_chunk_chars=60)
        first_emit_at = None
        consumed = 0
        for i in range(0, len(text), 4):
            consumed += len(text[i : i + 4])
            if ch.feed(text[i : i + 4]) and first_emit_at is None:
                first_emit_at = consumed
        self.assertIsNotNone(first_emit_at)
        self.assertLess(
            first_emit_at, len(text) / 2, "first chunk waited for most of the response"
        )


if __name__ == "__main__":
    unittest.main()
