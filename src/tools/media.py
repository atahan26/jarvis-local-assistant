"""Clipboard, timers, and Discord (spec §8.3 tools 4, 6, 7).

§8.4 is a hard constraint on the Discord tool: there is no supported API for
controlling your own Discord client, and self-botting risks a permanent ban. So
this drives Discord the way a person would, by focusing its window and sending
the documented keyboard shortcuts. No Discord library is used and none should be.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .registry import Tool, ToolError, ToolResult

# ---------------------------------------------------------------------------
# clipboard
# ---------------------------------------------------------------------------


def clipboard(action: str, text: str | None = None) -> ToolResult:
    import pyperclip

    if action == "read":
        try:
            content = pyperclip.paste() or ""
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"couldn't read the clipboard: {type(exc).__name__}") from exc
        if not content.strip():
            return ToolResult(True, "The clipboard is empty.", {"text": ""})
        # It is about to be spoken, so keep it short.
        spoken = content.strip()
        if len(spoken) > 400:
            spoken = spoken[:400] + ", and it goes on from there"
        return ToolResult(True, f"The clipboard says: {spoken}", {"text": content})

    if action == "write":
        if not text:
            raise ToolError("write needs some text")
        try:
            pyperclip.copy(text)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"couldn't write to the clipboard: {type(exc).__name__}") from exc
        return ToolResult(True, "Copied that to the clipboard.", {"text": text})

    if action == "clear":
        pyperclip.copy("")
        return ToolResult(True, "Cleared the clipboard.")

    raise ToolError(f"unknown action {action!r}")


CLIPBOARD_TOOL = Tool(
    name="clipboard",
    description="Read what is on the clipboard, copy new text to it, or clear it.",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write", "clear"]},
            "text": {"type": "string", "description": "The text to copy, for 'write'."},
        },
        "required": ["action"],
    },
    handler=clipboard,
)


# ---------------------------------------------------------------------------
# timers
# ---------------------------------------------------------------------------


@dataclass
class Timer:
    name: str
    seconds: float
    started_at: float
    timer: threading.Timer

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.started_at))


class TimerStore:
    """Named timers that fire a callback when they elapse.

    The callback is set by the conversation loop so an expiring timer can be
    spoken aloud; without one, timers still run and can still be queried.
    """

    def __init__(self):
        self._timers: dict[str, Timer] = {}
        self._lock = threading.Lock()
        self.on_elapsed = None

    def _fire(self, name: str) -> None:
        with self._lock:
            self._timers.pop(name, None)
        if self.on_elapsed:
            try:
                self.on_elapsed(name)
            except Exception:  # noqa: BLE001 - never let a callback kill the thread
                pass

    def set(self, name: str, seconds: float) -> None:
        with self._lock:
            existing = self._timers.pop(name, None)
        if existing:
            existing.timer.cancel()
        handle = threading.Timer(seconds, self._fire, args=(name,))
        handle.daemon = True
        with self._lock:
            self._timers[name] = Timer(name, seconds, time.monotonic(), handle)
        handle.start()

    def cancel(self, name: str) -> bool:
        with self._lock:
            timer = self._timers.pop(name, None)
        if timer is None:
            return False
        timer.timer.cancel()
        return True

    def cancel_all(self) -> int:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.timer.cancel()
        return len(timers)

    def list(self) -> list[Timer]:
        with self._lock:
            return sorted(self._timers.values(), key=lambda t: t.remaining)


TIMERS = TimerStore()


def _say_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second" + ("s" if seconds != 1 else "")
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        if rest:
            return f"{minutes} minute" + ("s" if minutes != 1 else "") + f" and {rest} seconds"
        return f"{minutes} minute" + ("s" if minutes != 1 else "")
    hours, minutes = divmod(minutes, 60)
    return f"{hours} hour" + ("s" if hours != 1 else "") + (f" {minutes} minutes" if minutes else "")


def make_timer_tool(store: TimerStore = TIMERS):
    def timer(action: str, name: str | None = None, seconds: int | None = None) -> ToolResult:
        label = (name or "timer").strip().lower()

        if action == "set":
            if not seconds or seconds <= 0:
                raise ToolError("a timer needs a duration in seconds")
            if seconds > 24 * 3600:
                raise ToolError("that's longer than a day; I can't set that")
            store.set(label, float(seconds))
            return ToolResult(
                True, f"{_say_duration(seconds)} timer set.",
                {"name": label, "seconds": seconds},
            )

        if action == "cancel":
            if name is None and len(store.list()) > 1:
                raise ToolError("there's more than one timer, which should I cancel?")
            target = label if name else (store.list()[0].name if store.list() else label)
            if store.cancel(target):
                return ToolResult(True, f"Cancelled the {target}.", {"name": target})
            return ToolResult(False, f"There's no {target} running.")

        if action == "cancel_all":
            count = store.cancel_all()
            return ToolResult(True, f"Cancelled {count} timer" + ("s." if count != 1 else "."))

        if action == "list":
            running = store.list()
            if not running:
                return ToolResult(True, "No timers running.", {"timers": []})
            parts = [f"{t.name} with {_say_duration(t.remaining)} left" for t in running]
            return ToolResult(
                True, "You have " + ", and ".join(parts) + ".",
                {"timers": [{"name": t.name, "remaining": t.remaining} for t in running]},
            )

        raise ToolError(f"unknown action {action!r}")

    return timer


TIMER_TOOL = Tool(
    name="timer",
    description=(
        "Set, cancel, or check countdown timers. Convert whatever the user says "
        "into seconds: 'ten minutes' is 600."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["set", "cancel", "cancel_all", "list"]},
            "name": {
                "type": "string",
                "description": "What to call it, e.g. 'pasta'. Defaults to 'timer'.",
            },
            "seconds": {"type": "integer", "description": "Duration in seconds, for 'set'."},
        },
        "required": ["action"],
    },
    handler=make_timer_tool(),
)


# ---------------------------------------------------------------------------
# Discord (§8.4: hotkeys against the focused window, never a self-bot)
# ---------------------------------------------------------------------------

# Discord's documented defaults. Disconnecting has no default binding at all.
DISCORD_SHORTCUTS = {
    "mute": ("ctrl", "shift", "m"),
    "unmute": ("ctrl", "shift", "m"),
    "toggle_mute": ("ctrl", "shift", "m"),
    "deafen": ("ctrl", "shift", "d"),
    "undeafen": ("ctrl", "shift", "d"),
    "toggle_deafen": ("ctrl", "shift", "d"),
}


def discord_control(action: str) -> ToolResult:
    from .apps import DEFAULT_APPS, _focus, _resolve, _running

    if action == "disconnect":
        # §8.4: there is no default keybind for this, and guessing one would
        # send a shortcut that means something else entirely.
        raise ToolError(
            "Discord has no default shortcut for leaving a voice channel. "
            "Set one in Discord's keybind settings and I can use it."
        )

    combo = DISCORD_SHORTCUTS.get(action)
    if combo is None:
        raise ToolError(f"unknown action {action!r}")

    app = _resolve("discord", DEFAULT_APPS)
    if not _running(app):
        return ToolResult(False, "Discord isn't running.")
    if not _focus(app):
        return ToolResult(
            False,
            "I couldn't bring Discord to the front, and the shortcut only works "
            "on the focused window.",
        )
    time.sleep(0.15)  # let the window actually take focus before typing at it
    _send_combo(combo)

    spoken = {
        "mute": "Toggled your mic.",
        "unmute": "Toggled your mic.",
        "toggle_mute": "Toggled your mic.",
        "deafen": "Toggled deafen.",
        "undeafen": "Toggled deafen.",
        "toggle_deafen": "Toggled deafen.",
    }[action]
    # These shortcuts toggle; Discord does not report state back, so claiming a
    # specific end state would be a guess.
    return ToolResult(True, spoken, {"combo": "+".join(combo)})


DISCORD_TOOL = Tool(
    name="discord_control",
    description=(
        "Mute or deafen yourself in Discord. Works by focusing Discord and "
        "sending its keyboard shortcut, so it toggles rather than setting a "
        "specific state."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["toggle_mute", "toggle_deafen", "disconnect"],
                "description": "What to do in Discord.",
            }
        },
        "required": ["action"],
    },
    handler=discord_control,
)


# ---------------------------------------------------------------------------
# raw hotkeys (§8.3 tool 3)
# ---------------------------------------------------------------------------

# Only these may be sent. §5 forbids arbitrary execution, and an unrestricted
# key injector is exactly that with extra steps: it could type anything into
# anything, including a terminal.
ALLOWED_KEYS = {
    "ctrl", "shift", "alt", "win", "tab", "enter", "esc", "escape", "space",
    "backspace", "delete", "home", "end", "pageup", "pagedown",
    "up", "down", "left", "right",
    *(chr(c) for c in range(ord("a"), ord("z") + 1)),
    *(str(d) for d in range(10)),
    *(f"f{n}" for n in range(1, 13)),
}

_KEY_ALIASES = {
    "control": "ctrl", "escape": "esc", "windows": "win", "super": "win",
    "return": "enter", "pgup": "pageup", "pgdn": "pagedown",
}


def _normalise_combo(keys: list[str]) -> tuple[str, ...]:
    cleaned = []
    for key in keys:
        name = str(key).strip().lower()
        name = _KEY_ALIASES.get(name, name)
        if name not in ALLOWED_KEYS:
            raise ToolError(f"I'm not allowed to send the key {key!r}")
        cleaned.append(name)
    if not cleaned:
        raise ToolError("no keys given")
    if len(cleaned) > 4:
        raise ToolError("that's too many keys at once")
    return tuple(cleaned)


def _send_combo(keys: tuple[str, ...]) -> None:
    from pynput.keyboard import Controller, Key

    keyboard = Controller()
    resolved = []
    for name in keys:
        special = getattr(Key, name, None)
        resolved.append(special if special is not None else name)

    for key in resolved:
        keyboard.press(key)
    time.sleep(0.03)
    for key in reversed(resolved):
        keyboard.release(key)


def send_hotkey(keys: list[str]) -> ToolResult:
    combo = _normalise_combo(keys)
    _send_combo(combo)
    return ToolResult(True, f"Sent {'+'.join(combo)}.", {"combo": "+".join(combo)})


HOTKEY_TOOL = Tool(
    name="send_hotkey",
    description=(
        "Send a keyboard shortcut to whatever window is focused, for example "
        "ctrl+c or alt+tab. Only modifier keys, letters, digits and function "
        "keys are allowed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The keys to press together, e.g. ['ctrl','shift','m'].",
            }
        },
        "required": ["keys"],
    },
    handler=send_hotkey,
    # Injecting keys into whatever happens to be focused can do real damage.
    requires_confirmation=True,
)
