"""LLM streaming, cancellation, and history semantics.

The cancellation tests matter well beyond M2: spec §7.3 gives barge-in a 40 ms
budget to stop an in-flight generation, and that only works if closing the
stream really stops the server rather than just muting output we still pay for.

Skipped automatically when llama-server is not running.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_llm -v
"""

from __future__ import annotations

import threading
import time
import unittest

from src import config as config_mod
from src.services.llm import Conversation, Generation, LLMClient

CFG = config_mod.load()


def _llm_up() -> bool:
    with LLMClient(CFG.llm_url) as c:
        return c.is_up()


class TestConversation(unittest.TestCase):
    def test_system_prompt_is_first_and_not_in_history(self):
        convo = Conversation()
        convo.add_user("hello")
        request = convo.for_request()
        self.assertEqual(request[0]["role"], "system")
        self.assertEqual(request[1]["role"], "user")
        self.assertNotIn("system", [m["role"] for m in convo.messages])

    def test_empty_assistant_text_is_not_appended(self):
        convo = Conversation()
        convo.add_user("hi")
        convo.add_assistant("   ")
        self.assertEqual(len(convo.messages), 1)

    def test_interrupted_reply_keeps_only_what_was_spoken(self):
        """§7.3: history holds what the user heard, and nothing else.

        Regression: an earlier version appended "[interrupted by user]" to the
        content. The model learned the pattern from its own history, started
        producing that string in later replies, and Kokoro read it out loud.
        No in-band markers.
        """
        convo = Conversation()
        convo.add_user("tell me a long story")
        convo.add_assistant("Once upon a", interrupted=True)
        content = convo.messages[-1]["content"]
        self.assertEqual(content, "Once upon a")
        self.assertNotIn("[", content)
        self.assertNotIn("interrupted", content.lower())

    def test_interruption_is_explained_in_the_system_prompt_instead(self):
        convo = Conversation()
        self.assertIn("interrupted", convo.system_prompt.lower())

    def test_system_prompt_forbids_spoken_asides(self):
        convo = Conversation()
        self.assertIn("bracketed", convo.system_prompt.lower())

    def test_history_is_trimmed(self):
        convo = Conversation(max_turns=2)
        for i in range(10):
            convo.add_user(f"u{i}")
            convo.add_assistant(f"a{i}")
        self.assertLessEqual(len(convo.messages), 4)
        self.assertIn("a9", convo.messages[-1]["content"])


@unittest.skipUnless(_llm_up(), "llama-server not running; start it with .\\run.ps1")
class TestStreaming(unittest.TestCase):
    def client(self) -> LLMClient:
        return LLMClient(
            CFG.llm_url,
            temperature=float(CFG.get("llm.temperature")),
            top_p=float(CFG.get("llm.top_p")),
            max_tokens=int(CFG.get("llm.max_tokens")),
            enable_thinking=bool(CFG.get("llm.enable_thinking")),
        )

    def test_streams_multiple_deltas_and_records_ttft(self):
        with self.client() as llm:
            gen = Generation()
            deltas = list(
                llm.stream(
                    [{"role": "user", "content": "Count from one to eight."}],
                    into=gen,
                    max_tokens=80,
                )
            )
        self.assertGreater(len(deltas), 1, "response did not stream incrementally")
        self.assertIsNotNone(gen.ttft_ms)
        self.assertTrue(gen.text.strip())
        self.assertIsNone(gen.error)
        # Exactly one delta may be flagged as the first.
        self.assertEqual(sum(1 for d in deltas if d.is_first), 1)

    def test_ttft_is_much_lower_than_total(self):
        with self.client() as llm:
            gen = Generation()
            for _ in llm.stream(
                [{"role": "user", "content": "Write three sentences about rain."}],
                into=gen,
                max_tokens=160,
            ):
                pass
        self.assertIsNotNone(gen.ttft_ms)
        self.assertLess(gen.ttft_ms, gen.total_ms, "TTFT should precede completion")

    def test_thinking_is_disabled_so_content_actually_arrives(self):
        """Qwen3 reasoning burns the whole budget before emitting any answer."""
        with self.client() as llm:
            gen = Generation()
            for _ in llm.stream(
                [{"role": "user", "content": "Say the word ready and nothing else."}],
                into=gen,
                max_tokens=40,
            ):
                pass
        self.assertTrue(gen.text.strip(), "no content: reasoning is probably still on")
        self.assertEqual(gen.reasoning, "")

    def test_cancel_event_stops_generation_early(self):
        cancel = threading.Event()
        gen = Generation()
        with self.client() as llm:
            stream = llm.stream(
                [{"role": "user", "content": "Slowly count from one to two hundred."}],
                cancel=cancel,
                into=gen,
                max_tokens=400,
            )
            for i, _delta in enumerate(stream):
                if i == 5:
                    cancel.set()
        self.assertTrue(gen.cancelled)
        self.assertIsNotNone(gen.cancel_ms)
        self.assertLess(len(gen.text), 400, "generation did not stop early")

    def test_closing_the_stream_meets_the_40ms_cancel_budget(self):
        """§7.3: the barge-in cancel path must return within 40 ms."""
        gen = Generation()
        with self.client() as llm:
            stream = llm.stream(
                [{"role": "user", "content": "Slowly count from one to two hundred."}],
                into=gen,
                max_tokens=400,
            )
            for i, _delta in enumerate(stream):
                if i == 5:
                    break
            started = time.monotonic()
            stream.close()
            elapsed_ms = (time.monotonic() - started) * 1000.0

        self.assertLess(
            elapsed_ms, 40.0, f"cancel took {elapsed_ms:.1f}ms, over the 40ms budget"
        )

    def test_partial_text_is_available_after_an_abandoned_stream(self):
        """What the user heard must survive the interruption, for §7.3 history."""
        gen = Generation()
        with self.client() as llm:
            stream = llm.stream(
                [{"role": "user", "content": "Count slowly from one to fifty."}],
                into=gen,
                max_tokens=300,
            )
            for i, _ in enumerate(stream):
                if i == 4:
                    break
            stream.close()
        self.assertTrue(gen.text.strip(), "no partial text captured")


if __name__ == "__main__":
    unittest.main()
