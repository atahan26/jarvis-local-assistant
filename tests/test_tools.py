"""Tool registry, schemas, validation and permissions (spec §8, §5).

Non-negotiable §5 is the point of most of these: the model names a registered
tool and nothing else reaches the OS. Hallucinated tools, hallucinated
arguments and wrong types must all be refused rather than forwarded.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_tools -v
"""

from __future__ import annotations

import unittest

from src.conversation import classify_confirmation
from src.tools import Registry, Tool, ToolError, ToolResult, build_registry
from src.tools.apps import (
    DEFAULT_APPS,
    DEFAULT_SITES,
    KnownApp,
    _as_url,
    _guess_site,
    _resolve,
    _resolve_site,
)


def echo(action: str, level: int | None = None) -> ToolResult:
    return ToolResult(True, f"action={action} level={level}", {"action": action})


def make_tool(**kwargs) -> Tool:
    defaults = dict(
        name="demo",
        description="A demo tool.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "close"]},
                "level": {"type": "integer"},
            },
            "required": ["action"],
        },
        handler=echo,
    )
    defaults.update(kwargs)
    return Tool(**defaults)


class TestRegistration(unittest.TestCase):
    def test_duplicate_registration_is_refused(self):
        reg = Registry()
        reg.register(make_tool())
        with self.assertRaises(ValueError):
            reg.register(make_tool())

    def test_schema_shape_matches_the_openai_contract(self):
        reg = Registry()
        reg.register(make_tool())
        schema = reg.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "demo")
        self.assertIn("description", schema["function"])
        self.assertIn("properties", schema["function"]["parameters"])

    def test_exceeding_max_exposed_is_loud(self):
        """§8.1: past ~15 tools a local model hallucinates arguments."""
        reg = Registry(max_exposed=2)
        for i in range(3):
            reg.register(make_tool(name=f"demo{i}"))
        with self.assertRaises(RuntimeError) as ctx:
            reg.schemas()
        self.assertIn("intent router", str(ctx.exception))


class TestDispatchSafety(unittest.TestCase):
    """§5: nothing unregistered reaches the OS."""

    def setUp(self):
        self.reg = Registry()
        self.reg.register(make_tool())

    def test_unknown_tool_is_refused_not_guessed(self):
        result = self.reg.call("dem0", '{"action": "get"}')
        self.assertFalse(result.ok)
        self.assertIn("no tool called", result.summary)

    def test_invalid_json_arguments_are_refused(self):
        result = self.reg.call("demo", "{not json")
        self.assertFalse(result.ok)
        self.assertIn("JSON", result.summary)

    def test_missing_required_argument_is_refused(self):
        result = self.reg.call("demo", '{"level": 5}')
        self.assertFalse(result.ok)
        self.assertIn("missing required", result.summary)

    def test_value_outside_the_enum_is_refused(self):
        result = self.reg.call("demo", '{"action": "delete_everything"}')
        self.assertFalse(result.ok)
        self.assertIn("must be one of", result.summary)

    def test_hallucinated_extra_arguments_are_dropped(self):
        result = self.reg.call("demo", '{"action": "get", "rm_rf": "/"}')
        self.assertTrue(result.ok)
        self.assertNotIn("rm_rf", result.summary)

    def test_a_handler_raising_does_not_kill_the_loop(self):
        def boom(action: str, level: int | None = None) -> ToolResult:
            raise RuntimeError("kaboom")

        reg = Registry()
        reg.register(make_tool(name="boom", handler=boom))
        result = reg.call("boom", '{"action": "get"}')
        self.assertFalse(result.ok)
        self.assertIn("kaboom", result.summary)

    def test_tool_error_becomes_a_clean_failure(self):
        def refuse(action: str, level: int | None = None) -> ToolResult:
            raise ToolError("I can't do that")

        reg = Registry()
        reg.register(make_tool(name="refuse", handler=refuse))
        result = reg.call("refuse", '{"action": "get"}')
        self.assertFalse(result.ok)
        self.assertEqual(result.summary, "I can't do that")


class TestCoercion(unittest.TestCase):
    def test_numeric_string_is_coerced(self):
        reg = Registry()
        reg.register(make_tool())
        result = reg.call("demo", '{"action": "set", "level": "30"}')
        self.assertTrue(result.ok)
        self.assertIn("level=30", result.summary)

    def test_non_numeric_level_is_refused(self):
        reg = Registry()
        reg.register(make_tool())
        result = reg.call("demo", '{"action": "set", "level": "loud"}')
        self.assertFalse(result.ok)
        self.assertIn("whole number", result.summary)

    def test_enum_case_is_normalised(self):
        reg = Registry()
        reg.register(make_tool())
        result = reg.call("demo", '{"action": "GET"}')
        self.assertTrue(result.ok)


class TestConfirmationGating(unittest.TestCase):
    """§8.1: destructive actions need a spoken yes."""

    def test_action_listed_as_destructive_needs_confirmation(self):
        tool = make_tool(confirm_actions=("close",))
        self.assertTrue(tool.needs_confirmation({"action": "close"}))
        self.assertFalse(tool.needs_confirmation({"action": "get"}))

    def test_whole_tool_can_require_confirmation(self):
        tool = make_tool(requires_confirmation=True)
        self.assertTrue(tool.needs_confirmation({"action": "get"}))

    def test_closing_an_app_is_gated(self):
        registry = build_registry()
        app_control = registry.get("app_control")
        self.assertTrue(app_control.needs_confirmation({"action": "close", "name": "Spotify"}))
        self.assertFalse(app_control.needs_confirmation({"action": "launch", "name": "Spotify"}))

    def test_volume_is_not_gated(self):
        registry = build_registry()
        self.assertFalse(registry.get("system_volume").needs_confirmation({"action": "mute"}))


class TestConfirmationParsing(unittest.TestCase):
    def test_yes_variants(self):
        for phrase in ["yes", "Yeah", "yep, do it", "sure", "okay", "go ahead", "confirm"]:
            with self.subTest(phrase=phrase):
                self.assertIs(classify_confirmation(phrase), True)

    def test_no_variants(self):
        for phrase in ["no", "nope", "cancel", "never mind", "don't", "stop"]:
            with self.subTest(phrase=phrase):
                self.assertIs(classify_confirmation(phrase), False)

    def test_ambiguous_is_neither(self):
        """Silence is not consent; anything unclear must not run the action."""
        for phrase in ["what?", "the weather in Istanbul", "maybe later", ""]:
            with self.subTest(phrase=phrase):
                self.assertIsNone(classify_confirmation(phrase))


class TestAppAllowlist(unittest.TestCase):
    """§5: launching resolves against an allowlist, never arbitrary input."""

    def test_known_app_resolves(self):
        self.assertEqual(_resolve("spotify", DEFAULT_APPS).name, "Spotify")
        self.assertEqual(_resolve("Discord", DEFAULT_APPS).name, "Discord")

    def test_alias_resolves(self):
        self.assertEqual(_resolve("music", DEFAULT_APPS).name, "Spotify")

    def test_unknown_app_is_refused_with_the_known_list(self):
        with self.assertRaises(ToolError) as ctx:
            _resolve("cmd.exe && format c:", DEFAULT_APPS)
        self.assertIn("don't know an app", str(ctx.exception))
        self.assertIn("Spotify", str(ctx.exception))

    def test_launch_targets_are_fixed_strings_not_user_text(self):
        for app in DEFAULT_APPS:
            with self.subTest(app=app.name):
                self.assertIsInstance(app.launch, str)
                self.assertNotIn(" ", app.launch)
                self.assertNotIn("&", app.launch)

    def test_matching_is_case_insensitive(self):
        app = KnownApp("Thing", "thing", ("thing.exe",), ("widget",))
        self.assertTrue(app.matches("THING"))
        self.assertTrue(app.matches("Widget"))

    def test_matching_respects_word_boundaries(self):
        """Regression: a substring test matched "thing" inside "something",
        so an unrelated request could have closed the wrong application."""
        app = KnownApp("Thing", "thing", ("thing.exe",), ("widget",))
        self.assertFalse(app.matches("something else entirely"))
        self.assertFalse(app.matches("nothing"))
        self.assertTrue(app.matches("open thing please"))

    def test_steam_does_not_match_a_longer_word(self):
        steam = next(a for a in DEFAULT_APPS if a.name == "Steam")
        self.assertFalse(steam.matches("steamroller"))
        self.assertTrue(steam.matches("open steam"))

    def test_trailing_words_still_resolve(self):
        self.assertEqual(_resolve("spotify please", DEFAULT_APPS).name, "Spotify")
        self.assertEqual(_resolve("open google chrome now", DEFAULT_APPS).name, "Chrome")


class TestWebsites(unittest.TestCase):
    """Reported: "open YouTube" opened Chrome with a blank page."""

    def test_known_sites_resolve(self):
        for spoken, expected in [
            ("YouTube", "YouTube"), ("youtube", "YouTube"), ("yt", "YouTube"),
            ("open youtube please", "YouTube"), ("gmail", "Gmail"),
            ("wikipedia", "Wikipedia"), ("twitch", "Twitch"),
        ]:
            with self.subTest(spoken=spoken):
                site = _resolve_site(spoken, DEFAULT_SITES)
                self.assertIsNotNone(site, spoken)
                self.assertEqual(site.name, expected)

    def test_every_known_site_is_https(self):
        for site in DEFAULT_SITES:
            with self.subTest(site=site.name):
                self.assertTrue(site.url.startswith("https://"))

    def test_bare_domains_become_https_urls(self):
        self.assertEqual(_as_url("example.com"), "https://example.com")
        self.assertEqual(_as_url("www.example.com"), "https://example.com")
        self.assertEqual(_as_url("example dot com"), "https://example.com")
        self.assertEqual(_as_url("https://example.com"), "https://example.com")

    def test_dangerous_urls_are_refused(self):
        """§5: a URL is an easy way to smuggle something past an allowlist."""
        for bad in [
            "file:///C:/Windows/System32",
            "javascript:alert(1)",
            "user:pass@example.com",
            "example.com:8080",
            "127.0.0.1",
            "localhost",
            "not a domain at all",
            "\\\\server\\share",
        ]:
            with self.subTest(bad=bad):
                self.assertIsNone(_as_url(bad), bad)

    def test_an_app_name_still_wins_over_a_site(self):
        """"Spotify" should launch the app, not open the web player."""
        self.assertTrue(any(a.matches("spotify") for a in DEFAULT_APPS))

    def test_unknown_brands_fall_back_to_a_website(self):
        """Reported: "open Instagram" failed outright instead of opening the site."""
        for spoken in ("Instagram", "Pinterest", "Duolingo"):
            with self.subTest(spoken=spoken):
                site = _resolve_site(spoken, DEFAULT_SITES)
                self.assertIsNotNone(site, spoken)
                self.assertTrue(site.url.startswith("https://"))

    def test_filler_words_are_not_guessed_as_brands(self):
        """"open please" must not become please.com."""
        for spoken in ("please", "the thing", "my app", "something", "it", "a window"):
            with self.subTest(spoken=spoken):
                self.assertIsNone(_guess_site(spoken), spoken)

    def test_long_phrases_are_not_guessed_as_brands(self):
        self.assertIsNone(_guess_site("the weather in Istanbul tomorrow"))

    def test_guessing_can_be_turned_off(self):
        self.assertIsNone(_resolve_site("Pinterest", DEFAULT_SITES, guess=False))

    def test_guessed_urls_still_pass_validation(self):
        site = _guess_site("Instagram")
        self.assertEqual(site.url, "https://instagram.com")


class TestNoContentGatekeeping(unittest.TestCase):
    """The tool layer opens what it is asked to open.

    Reported: the assistant said "I can't open that website" for an ordinary
    adult site. That refusal came from the model, not from here, and measured
    5/10 once there was conversation history. The tool layer never had an
    opinion and must not grow one: it validates URL SHAPE, not subject matter.
    """

    def test_url_validation_is_about_shape_not_subject(self):
        for domain in ("xnxx.com", "pornhub.com", "reddit.com", "example.com"):
            with self.subTest(domain=domain):
                self.assertEqual(_as_url(domain), f"https://{domain}")

    def test_no_keyword_blocklist_exists_in_the_code(self):
        import src.tools.apps as apps_module

        with open(apps_module.__file__, encoding="utf-8") as handle:
            source = handle.read().lower()
        for word in ("blocklist", "blacklist", "nsfw", "adult", "porn", "explicit"):
            self.assertNotIn(word, source, f"{word!r} suggests content filtering crept in")

    def test_the_system_prompt_tells_the_model_not_to_gatekeep(self):
        from src.conversation import TOOLS_NOTE

        lowered = TOOLS_NOTE.lower()
        self.assertIn("without", lowered)
        self.assertIn("moralising", lowered)

    def test_destructive_actions_are_still_gated(self):
        """Removing content gatekeeping must not remove the real safety gate."""
        registry = build_registry()
        self.assertTrue(
            registry.get("app_control").needs_confirmation({"action": "close", "name": "Steam"})
        )
        self.assertTrue(
            registry.get("send_hotkey").needs_confirmation({"keys": ["ctrl", "a"]})
        )


class TestAppVersusWebsite(unittest.TestCase):
    """"Spotify" is the app; "Spotify web" is the web player."""

    def setUp(self):
        from src.tools.apps import _WEB_HINT

        self.web_hint = _WEB_HINT

    def test_web_hint_is_detected(self):
        for spoken in ("spotify web", "spotify in the browser", "discord website",
                       "open steam online"):
            with self.subTest(spoken=spoken):
                self.assertTrue(self.web_hint.search(spoken))

    def test_plain_app_name_has_no_web_hint(self):
        for spoken in ("spotify", "discord", "steam", "visual studio code"):
            with self.subTest(spoken=spoken):
                self.assertIsNone(self.web_hint.search(spoken))

    def test_web_hint_is_stripped_before_guessing(self):
        site = _guess_site("pinterest web")
        self.assertIsNotNone(site)
        self.assertEqual(site.url, "https://pinterest.com")


class TestBuiltRegistry(unittest.TestCase):
    def test_the_first_three_tools_are_registered(self):
        # Membership, not an exact list: the set grows across milestones and
        # tests that pin the whole thing just have to be edited every time.
        registry = build_registry()
        for name in ("app_control", "system_volume", "web_search"):
            self.assertIn(name, registry.names)

    def test_all_schemas_build(self):
        registry = build_registry()
        self.assertEqual(len(registry.schemas()), len(registry))

    def test_every_tool_has_a_one_line_description(self):
        for name in build_registry().names:
            tool = build_registry().get(name)
            with self.subTest(tool=name):
                self.assertTrue(tool.description)
                self.assertNotIn("\n", tool.description)


class TestResultRendering(unittest.TestCase):
    def test_success_is_spoken_as_its_summary(self):
        self.assertEqual(ToolResult(True, "Volume set to 30 percent.").for_model(),
                         "Volume set to 30 percent.")

    def test_failure_is_marked(self):
        self.assertIn("Failed", ToolResult(False, "nope").for_model())


if __name__ == "__main__":
    unittest.main()
