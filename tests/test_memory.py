"""Long-term memory across restarts.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_memory -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tools import ToolError, build_registry
from src.tools.memory import MAX_FACTS, MemoryStore, make_memory_tool


class MemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "memory.json"
        self.store = MemoryStore(self.path)
        self.memory = make_memory_tool(self.store)

    def tearDown(self):
        self.dir.cleanup()


class TestPersistence(MemoryTestCase):
    def test_a_fact_survives_a_restart(self):
        """The whole point: close the process, reopen, still know you."""
        self.memory("remember", text="the user's name is Atahan")
        reopened = MemoryStore(self.path)
        self.assertEqual(len(reopened), 1)
        self.assertIn("Atahan", reopened.all()[0].text)

    def test_file_is_valid_json(self):
        self.memory("remember", text="likes the volume at thirty")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)
        self.assertEqual(len(data["facts"]), 1)

    def test_a_corrupt_file_does_not_stop_startup(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        store = MemoryStore(self.path)
        self.assertEqual(len(store), 0)

    def test_a_missing_file_is_fine(self):
        self.assertEqual(len(MemoryStore(Path(self.dir.name) / "nope.json")), 0)

    def test_no_temp_file_is_left_behind(self):
        self.memory("remember", text="something")
        self.assertEqual([p.name for p in Path(self.dir.name).glob("*.tmp")], [])


class TestRemember(MemoryTestCase):
    def test_remember_then_recall(self):
        self.memory("remember", text="works as a software developer")
        result = self.memory("recall", text="job")
        self.assertTrue(result.ok)

    def test_duplicates_are_not_stored_twice(self):
        self.memory("remember", text="likes the volume at thirty")
        result = self.memory("remember", text="Likes the volume at thirty.")
        self.assertIn("already knew", result.summary)
        self.assertEqual(len(self.store), 1)

    def test_empty_text_is_refused(self):
        with self.assertRaises(ToolError):
            self.memory("remember", text="   ")

    def test_very_long_facts_are_truncated(self):
        self.memory("remember", text="x" * 5000)
        self.assertLess(len(self.store.all()[0].text), 250)

    def test_the_store_is_capped(self):
        """Unbounded memory is a leak: every fact costs tokens on every turn."""
        for i in range(MAX_FACTS + 20):
            self.memory("remember", text=f"fact number {i}")
        self.assertLessEqual(len(self.store), MAX_FACTS)

    def test_the_cap_drops_the_oldest(self):
        for i in range(MAX_FACTS + 5):
            self.memory("remember", text=f"fact number {i}")
        texts = [f.text for f in self.store.all()]
        self.assertNotIn("fact number 0", texts)
        self.assertIn(f"fact number {MAX_FACTS + 4}", texts)


class TestForget(MemoryTestCase):
    def test_forget_by_substring(self):
        self.memory("remember", text="likes the volume at thirty")
        self.memory("remember", text="uses Discord every day")
        result = self.memory("forget", text="volume")
        self.assertTrue(result.ok)
        self.assertEqual(len(self.store), 1)

    def test_forgetting_something_unknown_is_a_clean_failure(self):
        self.assertFalse(self.memory("forget", text="unicorns").ok)

    def test_forget_all_clears_everything(self):
        self.memory("remember", text="a")
        self.memory("remember", text="b")
        self.memory("forget_all")
        self.assertEqual(len(self.store), 0)

    def test_forget_all_is_gated_behind_confirmation(self):
        tool = build_registry(memory_store=self.store).get("memory")
        self.assertTrue(tool.needs_confirmation({"action": "forget_all"}))
        self.assertFalse(tool.needs_confirmation({"action": "remember"}))

    def test_forgetting_persists(self):
        self.memory("remember", text="likes the volume at thirty")
        self.memory("forget", text="volume")
        self.assertEqual(len(MemoryStore(self.path)), 0)


class TestPromptInjection(MemoryTestCase):
    def test_empty_memory_adds_nothing_to_the_prompt(self):
        self.assertEqual(self.store.as_prompt(), "")

    def test_facts_appear_in_the_prompt(self):
        self.memory("remember", text="the user's name is Atahan")
        prompt = self.store.as_prompt()
        self.assertIn("Atahan", prompt)
        self.assertIn("previous conversations", prompt)

    def test_prompt_warns_against_inventing_memories(self):
        self.memory("remember", text="anything")
        self.assertIn("not claim to remember", self.store.as_prompt())

    def test_prompt_grows_with_facts_but_stays_bounded(self):
        for i in range(MAX_FACTS + 10):
            self.memory("remember", text=f"fact number {i} about something")
        # Bounded by MAX_FACTS * MAX_FACT_CHARS, well inside a 16k context.
        self.assertLess(len(self.store.as_prompt()), 16000)


class TestRegistryIntegration(unittest.TestCase):
    def test_memory_is_absent_when_no_store_is_given(self):
        self.assertNotIn("memory", build_registry().names)

    def test_memory_is_registered_when_a_store_is_given(self):
        with tempfile.TemporaryDirectory() as d:
            store = MemoryStore(Path(d) / "m.json")
            registry = build_registry(memory_store=store)
            self.assertIn("memory", registry.names)
            self.assertEqual(len(registry.schemas()), len(registry))

    def test_still_under_the_tool_ceiling_with_memory(self):
        with tempfile.TemporaryDirectory() as d:
            store = MemoryStore(Path(d) / "m.json")
            self.assertLessEqual(len(build_registry(memory_store=store)), 15)


if __name__ == "__main__":
    unittest.main()
