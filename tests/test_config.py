"""Config loading and path resolution.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src import config as config_mod


class TestDeepMerge(unittest.TestCase):
    def test_nested_override_keeps_siblings(self):
        base = {"turn": {"base_silence_ms": 700, "max_wait_ms": 3000}, "a": 1}
        over = {"turn": {"base_silence_ms": 900}}
        merged = config_mod._deep_merge(base, over)
        self.assertEqual(merged["turn"]["base_silence_ms"], 900)
        self.assertEqual(merged["turn"]["max_wait_ms"], 3000)
        self.assertEqual(merged["a"], 1)

    def test_base_is_not_mutated(self):
        base = {"turn": {"x": 1}}
        config_mod._deep_merge(base, {"turn": {"x": 2}})
        self.assertEqual(base["turn"]["x"], 1)

    def test_list_is_replaced_not_merged(self):
        merged = config_mod._deep_merge({"p": [1, 2, 3]}, {"p": [9]})
        self.assertEqual(merged["p"], [9])


class TestPathResolution(unittest.TestCase):
    def _cfg(self, data_dir: str) -> config_mod.Config:
        return config_mod.Config(
            {
                "paths": {
                    "data_dir": data_dir,
                    "bin_dir": "bin",
                    "models_dir": "models",
                    "downloads_dir": "downloads",
                    "log_dir": "logs",
                },
                "whisper": {"exe": "whisper/whisper-server.exe", "model": "m.bin", "host": "127.0.0.1", "port": 8081},
                "llm": {"exe": "llama/llama-server.exe", "model": "q.gguf", "host": "127.0.0.1", "port": 8080},
                "tts": {"model": "kokoro/k.onnx", "voices": "kokoro/v.bin"},
            }
        )

    def test_absolute_data_dir_wins(self):
        cfg = self._cfg("C:/jarvis-data")
        self.assertEqual(cfg.data_dir, Path("C:/jarvis-data"))
        self.assertEqual(cfg.bin_dir, Path("C:/jarvis-data/bin"))
        self.assertEqual(cfg.llama_exe, Path("C:/jarvis-data/bin/llama/llama-server.exe"))
        self.assertEqual(cfg.llm_model, Path("C:/jarvis-data/models/q.gguf"))
        self.assertEqual(cfg.tts_voices, Path("C:/jarvis-data/models/kokoro/v.bin"))

    def test_relative_data_dir_resolves_against_project_root(self):
        cfg = self._cfg("data")
        self.assertEqual(cfg.data_dir, config_mod.PROJECT_ROOT / "data")

    def test_log_dir_is_relative_to_project_not_data(self):
        cfg = self._cfg("C:/jarvis-data")
        self.assertEqual(cfg.log_dir, config_mod.PROJECT_ROOT / "logs")

    def test_endpoints(self):
        cfg = self._cfg("C:/jarvis-data")
        self.assertEqual(cfg.llm_url, "http://127.0.0.1:8080")
        self.assertEqual(cfg.whisper_url, "http://127.0.0.1:8081")


class TestGet(unittest.TestCase):
    def test_dotted_lookup_and_default(self):
        cfg = config_mod.Config({"turn": {"vad_threshold": 0.5}})
        self.assertEqual(cfg.get("turn.vad_threshold"), 0.5)
        self.assertEqual(cfg.get("turn.nope", 42), 42)
        with self.assertRaises(KeyError):
            cfg.get("turn.nope")

    def test_falsy_values_survive_the_default_sentinel(self):
        cfg = config_mod.Config({"a": {"b": 0, "c": False, "d": None}})
        self.assertEqual(cfg.get("a.b", 99), 0)
        self.assertIs(cfg.get("a.c", True), False)
        self.assertIsNone(cfg.get("a.d", "x"))


class TestRealConfigFile(unittest.TestCase):
    """The shipped config.yaml must contain every key the code reads."""

    def test_required_keys_present(self):
        cfg = config_mod.load()
        for key in [
            "whisper.host", "whisper.port", "whisper.threads", "whisper.language",
            "whisper.startup_timeout_s",
            "llm.host", "llm.port", "llm.n_gpu_layers", "llm.context_size",
            "llm.flash_attn", "llm.cache_type_k", "llm.cache_type_v",
            "llm.require_gpu", "llm.startup_timeout_s",
            "tts.voice", "tts.sample_rate",
            "audio.input_sample_rate", "audio.frame_ms",
            "turn.base_silence_ms", "turn.max_wait_ms", "turn.vad_threshold",
            "barge_in.echo_mode", "barge_in.tts_flush_budget_ms",
            "barge_in.llm_cancel_budget_ms",
            "wake_word.model",
        ]:
            with self.subTest(key=key):
                cfg.get(key)

    def test_spec_mandated_defaults(self):
        cfg = config_mod.load()
        # §7.2: 700ms base, not the 500ms most tutorials use.
        self.assertEqual(cfg.get("turn.base_silence_ms"), 700)
        self.assertEqual(cfg.get("turn.max_wait_ms"), 3000)
        # §3 latency budget.
        self.assertEqual(cfg.get("barge_in.tts_flush_budget_ms"), 60)
        self.assertEqual(cfg.get("barge_in.llm_cancel_budget_ms"), 40)
        # §7.1: headphones are the supported default.
        self.assertEqual(cfg.get("barge_in.echo_mode"), "headphones")


if __name__ == "__main__":
    unittest.main()
