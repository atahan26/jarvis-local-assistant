"""Launch, focus and close applications (spec §8.2, §8.3 tool 2).

§5 forbids arbitrary code execution from the LLM, so "launch" is NOT a shell
passthrough. The model supplies a name, that name is matched against a
configured allowlist of known applications, and anything unmatched is refused.
The model can ask to open Spotify; it cannot ask to run an arbitrary command.

Closing is destructive, so it is declared in `confirm_actions` and gated behind
spoken confirmation (§8.1).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from .registry import Tool, ToolError, ToolResult


@dataclass(frozen=True)
class KnownApp:
    """One allowlisted application."""

    name: str
    # Where to launch it from. A bare name is resolved via the shell's app paths
    # (start menu / App Paths registry), not by searching the filesystem.
    launch: str
    # Process names that count as "this app is running".
    processes: tuple[str, ...]
    # Substrings that a user might say.
    aliases: tuple[str, ...] = ()

    def matches(self, spoken: str) -> bool:
        """Match on whole words only.

        A plain substring test is not safe here: "thing" is inside "something",
        and "steam" is inside "steamroller", so loose matching would close the
        wrong application. Word boundaries keep "open spotify please" working
        while refusing accidental hits.
        """
        needle = " ".join(spoken.strip().lower().split())
        if not needle:
            return False
        candidates = (self.name.lower(), *(a.lower() for a in self.aliases))
        for candidate in candidates:
            if needle == candidate:
                return True
            if re.search(rf"\b{re.escape(candidate)}\b", needle):
                return True
        return False


@dataclass(frozen=True)
class KnownSite:
    """A website that can be opened by name.

    "Open YouTube" resolved to Chrome and opened an empty browser, because
    websites were not a thing this tool understood. They are now first-class:
    the user does not care whether something is an app or a page.
    """

    name: str
    url: str
    aliases: tuple[str, ...] = ()

    def matches(self, spoken: str) -> bool:
        needle = " ".join(spoken.strip().lower().split())
        if not needle:
            return False
        for candidate in (self.name.lower(), *(a.lower() for a in self.aliases)):
            if needle == candidate or re.search(rf"\b{re.escape(candidate)}\b", needle):
                return True
        return False


DEFAULT_SITES: tuple[KnownSite, ...] = (
    KnownSite("YouTube", "https://www.youtube.com", ("youtube", "yt")),
    KnownSite("Gmail", "https://mail.google.com", ("gmail", "my email", "email")),
    KnownSite("Google", "https://www.google.com", ("google",)),
    KnownSite("Google Maps", "https://maps.google.com", ("maps", "google maps")),
    KnownSite("GitHub", "https://github.com", ("github",)),
    KnownSite("Reddit", "https://www.reddit.com", ("reddit",)),
    KnownSite("Twitch", "https://www.twitch.tv", ("twitch",)),
    KnownSite("Netflix", "https://www.netflix.com", ("netflix",)),
    KnownSite("Wikipedia", "https://www.wikipedia.org", ("wikipedia", "wiki")),
    KnownSite("X", "https://x.com", ("twitter", "x dot com")),
    KnownSite("WhatsApp Web", "https://web.whatsapp.com", ("whatsapp", "whatsapp web")),
    KnownSite("Spotify Web", "https://open.spotify.com", ("spotify web",)),
    KnownSite("Instagram", "https://www.instagram.com", ("instagram", "insta", "ig")),
    KnownSite("Facebook", "https://www.facebook.com", ("facebook", "fb")),
    KnownSite("TikTok", "https://www.tiktok.com", ("tiktok", "tik tok")),
    KnownSite("LinkedIn", "https://www.linkedin.com", ("linkedin",)),
    KnownSite("Amazon", "https://www.amazon.com", ("amazon",)),
    KnownSite("ChatGPT", "https://chatgpt.com", ("chatgpt", "chat gpt")),
    KnownSite("Google Drive", "https://drive.google.com", ("drive", "google drive")),
    KnownSite("Google Translate", "https://translate.google.com", ("translate",)),
)

# Spoken hints that the user wants the website even though an app exists.
_WEB_HINT = re.compile(r"\b(web|website|site|browser|online|in the browser)\b")


# Deliberately a fixed list rather than "search the filesystem for an exe".
DEFAULT_APPS: tuple[KnownApp, ...] = (
    KnownApp("Spotify", "spotify", ("Spotify.exe",), ("music", "spotify")),
    KnownApp("Discord", "discord", ("Discord.exe",), ("discord",)),
    KnownApp("Steam", "steam", ("steam.exe",), ("steam",)),
    KnownApp("Chrome", "chrome", ("chrome.exe",), ("google chrome", "browser")),
    KnownApp("Firefox", "firefox", ("firefox.exe",), ("mozilla",)),
    KnownApp("Notepad", "notepad", ("notepad.exe",), ("notes", "text editor")),
    KnownApp("Calculator", "calc", ("CalculatorApp.exe", "Calculator.exe"), ("calc",)),
    KnownApp("Explorer", "explorer", ("explorer.exe",), ("file explorer", "files")),
    KnownApp("Task Manager", "taskmgr", ("Taskmgr.exe",), ("task manager",)),
    KnownApp("Visual Studio Code", "code", ("Code.exe",), ("vscode", "vs code")),
)


def _resolve(name: str, apps: tuple[KnownApp, ...]) -> KnownApp:
    for app in apps:
        if app.matches(name):
            return app
    known = ", ".join(a.name for a in apps)
    raise ToolError(f"I don't know an app called {name}. I know: {known}.")


# A bare domain the user named out loud, e.g. "open example dot com". Kept
# strict: §5 forbids arbitrary execution, and a URL is one of the easier ways
# to smuggle something past an allowlist. No scheme other than https, no
# credentials, no local files, no IP literals, no ports.
_DOMAIN = re.compile(
    r"^(?!-)(?:[a-z0-9-]{1,63}\.)+(?:com|org|net|io|dev|co|edu|gov|tv|me|app|ai"
    r"|info|biz|tr|de|uk|fr|es|it|nl)$"
)


def _as_url(spoken: str) -> str | None:
    """Turn a spoken site name into a safe https URL, or None if it is not one."""
    text = " ".join(spoken.strip().lower().split())
    text = text.replace(" dot ", ".").replace(" ", "")
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.split("/")[0]
    if text.startswith("www."):
        text = text[4:]
    if "@" in text or ":" in text or not _DOMAIN.match(text):
        return None
    return f"https://{text}"


_BRAND = re.compile(r"^[a-z0-9][a-z0-9 -]{1,28}$")

# Words that are requests or descriptions, not brand names. Guessing
# "please.com" from "open please" would be worse than admitting ignorance.
_NOT_A_BRAND = {
    "the", "a", "an", "my", "some", "please", "app", "application", "program",
    "window", "browser", "web", "website", "site", "online", "thing", "it",
    "that", "this", "something", "anything", "everything", "here", "there",
}


def _guess_site(name: str) -> KnownSite | None:
    """Treat an unknown brand name as a .com.

    Requested after "open Instagram" failed outright: if something is not a known
    app, the website is nearly always what was meant. Deliberately conservative
    about what counts as a brand, and the URL still goes through _as_url, so the
    same validation applies as for a domain the user spelled out.
    """
    cleaned = " ".join(str(name).strip().lower().split())
    cleaned = _WEB_HINT.sub("", cleaned).strip()
    if not cleaned or not _BRAND.match(cleaned):
        return None
    words = cleaned.split()
    if len(words) > 2 or any(w in _NOT_A_BRAND for w in words):
        return None
    slug = "".join(words).replace("-", "")
    url = _as_url(f"{slug}.com")
    if not url:
        return None
    return KnownSite(name.strip().title(), url)


def _resolve_site(
    name: str, sites: tuple[KnownSite, ...], *, guess: bool = True
) -> KnownSite | None:
    for site in sites:
        if site.matches(name):
            return site
    url = _as_url(name)
    if url:
        return KnownSite(name.strip(), url)
    return _guess_site(name) if guess else None


def _open_url(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


def _running(app: KnownApp) -> list:
    import psutil

    wanted = {p.lower() for p in app.processes}
    found = []
    for proc in psutil.process_iter(["name", "pid"]):
        name = (proc.info.get("name") or "").lower()
        if name in wanted:
            found.append(proc)
    return found


def _focus(app: KnownApp) -> bool:
    """Bring an existing window to the front. Returns False if none was found."""
    try:
        import uiautomation as auto
    except ImportError:
        return False

    try:
        for window in auto.GetRootControl().GetChildren():
            title = (window.Name or "").lower()
            if not title:
                continue
            if app.name.lower() in title or any(a.lower() in title for a in app.aliases):
                window.SetActive()
                window.SetTopmost(True)
                window.SetTopmost(False)
                return True
    except Exception:
        return False
    return False


def make_app_control(
    apps: tuple[KnownApp, ...] = DEFAULT_APPS,
    sites: tuple[KnownSite, ...] = DEFAULT_SITES,
):
    def app_control(action: str, name: str) -> ToolResult:
        # Deciding between an app and a website:
        #   "Spotify"      -> the desktop app, if it is installed
        #   "Spotify web"  -> the web player, because they said so
        #   "Instagram"    -> the website, since there is no app for it
        wants_web = bool(_WEB_HINT.search(name.lower()))
        has_app = any(a.matches(name) for a in apps)
        site = _resolve_site(name, sites)

        if site is not None and (wants_web or not has_app):
            if action in ("launch", "focus"):
                _open_url(site.url)
                return ToolResult(True, f"Opening {site.name}.",
                                  {"site": site.name, "url": site.url})
            if action == "status":
                return ToolResult(True, f"{site.name} is a website; I can open it.",
                                  {"site": site.name})
            raise ToolError(f"{site.name} is a website, so I can only open it")

        app = _resolve(name, apps)

        if action == "launch":
            if _running(app):
                if _focus(app):
                    return ToolResult(True, f"{app.name} was already open, brought it to the front.")
                return ToolResult(True, f"{app.name} is already running.")
            try:
                # start via the shell so App Paths / start-menu entries resolve.
                # No user text ever reaches a shell string: `app.launch` comes
                # from the allowlist above, not from the model.
                subprocess.Popen(
                    ["cmd", "/c", "start", "", app.launch],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                raise ToolError(f"could not start {app.name}: {exc}") from exc
            return ToolResult(True, f"Opening {app.name}.", {"app": app.name})

        if action == "focus":
            if not _running(app):
                return ToolResult(False, f"{app.name} isn't running.")
            if _focus(app):
                return ToolResult(True, f"Switched to {app.name}.")
            return ToolResult(False, f"Couldn't find a {app.name} window to switch to.")

        if action == "close":
            procs = _running(app)
            if not procs:
                return ToolResult(False, f"{app.name} isn't running.")
            closed = 0
            for proc in procs:
                try:
                    proc.terminate()
                    closed += 1
                except Exception:
                    continue
            if not closed:
                raise ToolError(f"could not close {app.name}")
            return ToolResult(True, f"Closed {app.name}.", {"app": app.name, "processes": closed})

        if action == "status":
            procs = _running(app)
            if procs:
                return ToolResult(True, f"{app.name} is running.", {"running": True})
            return ToolResult(True, f"{app.name} is not running.", {"running": False})

        raise ToolError(f"unknown action {action!r}")

    return app_control


def build_tool(
    apps: tuple[KnownApp, ...] = DEFAULT_APPS,
    sites: tuple[KnownSite, ...] = DEFAULT_SITES,
) -> Tool:
    known = ", ".join(a.name for a in apps)
    known_sites = ", ".join(s.name for s in sites)
    return Tool(
        name="app_control",
        description=(
            f"Open, switch to, close, or check an application, or open a website. "
            f"Installed apps: {known}. Known sites: {known_sites}. Any other name "
            f"is opened as a website, so pass the name the user said rather than "
            f"refusing. Use this for 'open YouTube' instead of opening a browser. "
            f"If the user says 'web' or 'in the browser', include that word in the "
            f"name so the website is used instead of the app."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["launch", "focus", "close", "status"],
                    "description": "What to do. Websites can only be launched.",
                },
                "name": {
                    "type": "string",
                    "description": (
                        f"Which application or website. Apps: {known}. "
                        f"Sites: {known_sites}, or a domain like example.com."
                    ),
                },
            },
            "required": ["action", "name"],
        },
        handler=make_app_control(apps, sites),
        # §8.1: closing an app can lose the user's work, so it is confirmed.
        confirm_actions=("close",),
    )


TOOL = build_tool()


# ---------------------------------------------------------------------------
# window_info (§8.3 tool 8)
# ---------------------------------------------------------------------------


def _visible_windows() -> list[str]:
    try:
        import uiautomation as auto
    except ImportError:
        return []
    titles = []
    try:
        for window in auto.GetRootControl().GetChildren():
            title = (window.Name or "").strip()
            if title and title.lower() not in {"program manager", "windows input experience"}:
                titles.append(title)
    except Exception:
        return []
    return titles


def window_info(action: str = "focused") -> ToolResult:
    if action == "focused":
        try:
            import uiautomation as auto

            focused = auto.GetFocusedControl()
            window = focused.GetTopLevelControl() if focused else None
            title = (window.Name or "").strip() if window else ""
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"couldn't read the focused window ({type(exc).__name__})") from exc
        if not title:
            return ToolResult(True, "Nothing seems to be focused right now.", {"title": ""})
        return ToolResult(True, f"You're looking at {title}.", {"title": title})

    if action == "list":
        titles = _visible_windows()
        if not titles:
            return ToolResult(True, "I couldn't see any open windows.", {"windows": []})
        shown = titles[:8]
        return ToolResult(
            True,
            "Open windows: " + ", ".join(shown) + ("." if len(titles) <= 8 else ", and more."),
            {"windows": titles},
        )

    raise ToolError(f"unknown action {action!r}")


WINDOW_TOOL = Tool(
    name="window_info",
    description="Find out which window is focused, or what windows are open.",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["focused", "list"]},
        },
        "required": ["action"],
    },
    handler=window_info,
)
