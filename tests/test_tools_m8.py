"""The rest of the §8.3 tool set.

Note what is deliberately NOT tested by actually doing it: sending hotkeys and
closing apps have real side effects on the machine running the suite. Those are
covered at the allowlist and validation layer, which is where §5 is enforced
anyway.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_tools_m8 -v
"""

from __future__ import annotations

import time
import unittest

from src.tools import ToolError, build_registry
from src.tools.conversation_control import FORGET, SLEEP, conversation_control
from src.tools.media import (
    ALLOWED_KEYS,
    DISCORD_SHORTCUTS,
    TimerStore,
    _normalise_combo,
    _say_duration,
    clipboard,
    discord_control,
    make_timer_tool,
)


class TestRegistrySize(unittest.TestCase):
    def test_the_full_v1_set_is_registered(self):
        registry = build_registry()
        self.assertEqual(len(registry), 11)

    def test_still_under_the_hallucination_ceiling(self):
        """§8.1: an 8B model degrades past roughly 10-15 tools."""
        registry = build_registry()
        self.assertLessEqual(len(registry), 15)
        self.assertEqual(len(registry.schemas()), 11)

    def test_every_spec_tool_is_present(self):
        expected = {
            "system_volume", "app_control", "send_hotkey", "discord_control",
            "web_search", "clipboard", "timer", "window_info",
            "screenshot_describe", "conversation_control",
        }
        self.assertTrue(expected.issubset(set(build_registry().names)))


class TestConversationControl(unittest.TestCase):
    """§8.3 tool 10, extended with sleep-on-command."""

    def test_sleep_returns_a_control_directive(self):
        result = conversation_control("sleep")
        self.assertTrue(result.ok)
        self.assertEqual(result.data["control"], SLEEP)

    def test_sleep_says_how_to_wake_it(self):
        self.assertIn("hey Jarvis", conversation_control("sleep").summary)

    def test_forget_returns_a_directive(self):
        self.assertEqual(conversation_control("forget").data["control"], FORGET)

    def test_unknown_action_is_refused(self):
        with self.assertRaises(ToolError):
            conversation_control("self_destruct")

    def test_directives_are_data_not_side_effects(self):
        """Tools must not reach into the Assistant; the loop reads the directive."""
        for action in ("sleep", "stop_talking", "repeat", "forget"):
            with self.subTest(action=action):
                self.assertIn("control", conversation_control(action).data)


class TestHotkeyAllowlist(unittest.TestCase):
    """§5: an unrestricted key injector is arbitrary execution with extra steps."""

    def test_ordinary_combos_are_allowed(self):
        self.assertEqual(_normalise_combo(["ctrl", "c"]), ("ctrl", "c"))
        self.assertEqual(_normalise_combo(["Ctrl", "Shift", "M"]), ("ctrl", "shift", "m"))

    def test_aliases_are_normalised(self):
        self.assertEqual(_normalise_combo(["control", "return"]), ("ctrl", "enter"))
        self.assertEqual(_normalise_combo(["windows", "d"]), ("win", "d"))

    def test_unknown_keys_are_refused(self):
        for keys in (["ctrl", "eject"], ["media_play"], ["\n"], ["cmd /c del"]):
            with self.subTest(keys=keys):
                with self.assertRaises(ToolError):
                    _normalise_combo(keys)

    def test_empty_combo_is_refused(self):
        with self.assertRaises(ToolError):
            _normalise_combo([])

    def test_absurdly_long_combos_are_refused(self):
        with self.assertRaises(ToolError):
            _normalise_combo(["ctrl", "shift", "alt", "win", "a"])

    def test_allowlist_contains_only_single_keys_never_text(self):
        """The model must not be able to type prose into a terminal.

        Every entry is either one character (a letter or digit) or one of a
        fixed set of named non-printing keys. Nothing composes into a sentence.
        """
        named = {
            "ctrl", "shift", "alt", "win", "tab", "enter", "esc", "escape",
            "space", "backspace", "delete", "home", "end", "pageup", "pagedown",
            "up", "down", "left", "right",
            *(f"f{n}" for n in range(1, 13)),
        }
        for key in ALLOWED_KEYS:
            with self.subTest(key=key):
                self.assertTrue(
                    (len(key) == 1 and key.isalnum()) or key in named,
                    f"{key!r} is neither a single character nor a known named key",
                )

    def test_allowlist_cannot_express_a_word(self):
        """A combo is capped at 4 keys, so no allowlisted sequence spells anything."""
        with self.assertRaises(ToolError):
            _normalise_combo(list("format"))

    def test_sending_keys_requires_confirmation(self):
        tool = build_registry().get("send_hotkey")
        self.assertTrue(tool.needs_confirmation({"keys": ["ctrl", "c"]}))


class TestDiscord(unittest.TestCase):
    """§8.4: keyboard shortcuts only, never a self-bot."""

    def test_documented_shortcuts_only(self):
        self.assertEqual(DISCORD_SHORTCUTS["toggle_mute"], ("ctrl", "shift", "m"))
        self.assertEqual(DISCORD_SHORTCUTS["toggle_deafen"], ("ctrl", "shift", "d"))

    def test_disconnect_is_refused_rather_than_guessed(self):
        """Discord has no default keybind for this; guessing would send the
        deafen shortcut and silently do the wrong thing."""
        with self.assertRaises(ToolError) as ctx:
            discord_control("disconnect")
        self.assertIn("no default shortcut", str(ctx.exception))

    def test_no_discord_library_is_imported(self):
        import src.tools.media as media

        with open(media.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for banned in ("import discord", "discord.py", "discum", "selfbot"):
            self.assertNotIn(banned, source)


class TestTimers(unittest.TestCase):
    def setUp(self):
        self.store = TimerStore()
        self.timer = make_timer_tool(self.store)

    def tearDown(self):
        self.store.cancel_all()

    def test_set_and_list(self):
        self.timer("set", name="pasta", seconds=60)
        result = self.timer("list")
        self.assertIn("pasta", result.summary)

    def test_duration_is_spoken_naturally(self):
        self.assertEqual(_say_duration(30), "30 seconds")
        self.assertEqual(_say_duration(60), "1 minute")
        self.assertEqual(_say_duration(600), "10 minutes")
        self.assertIn("hour", _say_duration(3700))

    def test_cancel_removes_it(self):
        self.timer("set", name="tea", seconds=60)
        self.assertTrue(self.timer("cancel", name="tea").ok)
        self.assertIn("No timers", self.timer("list").summary)

    def test_cancelling_a_missing_timer_is_a_clean_failure(self):
        self.assertFalse(self.timer("cancel", name="ghost").ok)

    def test_setting_without_a_duration_is_refused(self):
        with self.assertRaises(ToolError):
            self.timer("set", name="x")

    def test_absurd_durations_are_refused(self):
        with self.assertRaises(ToolError):
            self.timer("set", name="x", seconds=99999999)

    def test_ambiguous_cancel_asks_which(self):
        self.timer("set", name="pasta", seconds=60)
        self.timer("set", name="tea", seconds=60)
        with self.assertRaises(ToolError) as ctx:
            self.timer("cancel")
        self.assertIn("more than one", str(ctx.exception))

    def test_elapsed_timer_fires_the_callback(self):
        fired = []
        self.store.on_elapsed = fired.append
        self.store.set("quick", 0.15)
        time.sleep(0.5)
        self.assertEqual(fired, ["quick"])

    def test_resetting_a_name_replaces_the_old_timer(self):
        self.store.set("x", 60)
        self.store.set("x", 30)
        self.assertEqual(len(self.store.list()), 1)


class TestClipboard(unittest.TestCase):
    def test_write_then_read_round_trip(self):
        marker = "jarvis clipboard test 12345"
        clipboard("write", text=marker)
        self.assertIn(marker, clipboard("read").summary)

    def test_write_without_text_is_refused(self):
        with self.assertRaises(ToolError):
            clipboard("write")

    def test_long_clipboard_content_is_truncated_for_speech(self):
        clipboard("write", text="x" * 5000)
        result = clipboard("read")
        self.assertLess(len(result.summary), 600)
        self.assertIn("goes on", result.summary)

    def test_empty_clipboard_is_reported_plainly(self):
        clipboard("clear")
        self.assertIn("empty", clipboard("read").summary)


class TestScreenshotStub(unittest.TestCase):
    def test_it_admits_it_cannot_see(self):
        from src.tools.system import screenshot_describe

        result = screenshot_describe("what's on screen")
        self.assertFalse(result.ok)
        self.assertIn("can't see", result.summary)

    def test_description_warns_the_model_off(self):
        tool = build_registry().get("screenshot_describe")
        self.assertIn("NOT AVAILABLE", tool.description)


if __name__ == "__main__":
    unittest.main()
