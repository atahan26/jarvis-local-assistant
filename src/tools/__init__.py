"""Tool layer (spec §8).

`build_registry()` is the single place tools are wired up, so the set the model
can see is auditable in one glance. Nothing reaches the OS that is not listed
here (non-negotiable §5).
"""

from __future__ import annotations

from .registry import Registry, Tool, ToolError, ToolResult

__all__ = ["Registry", "Tool", "ToolError", "ToolResult", "build_registry"]


def build_registry(max_exposed: int = 15, memory_store=None) -> Registry:
    """The complete tool set (spec §8.3, plus memory).

    Twelve tools, each grouping related actions behind an `action` enum, which is
    what keeps the count under the ~10-15 an 8B model handles reliably. Adding a
    thirteenth is fine; adding twelve more is not, and Registry.schemas() will
    say so rather than letting accuracy rot quietly.
    """
    from . import apps, conversation_control, media, system, web

    registry = Registry(max_exposed=max_exposed)
    registry.register(system.TOOL)              # 1  system_volume
    registry.register(apps.TOOL)                # 2  app_control
    registry.register(media.HOTKEY_TOOL)        # 3  send_hotkey
    registry.register(media.DISCORD_TOOL)       # 4  discord_control
    registry.register(web.TOOL)                 # 5  web_search
    registry.register(media.CLIPBOARD_TOOL)     # 6  clipboard
    registry.register(media.TIMER_TOOL)         # 7  timer
    registry.register(apps.WINDOW_TOOL)         # 8  window_info
    registry.register(system.SCREENSHOT_TOOL)   # 9  screenshot_describe (stub)
    registry.register(conversation_control.TOOL)  # 10 conversation_control
    registry.register(system.INFO_TOOL)         # 11 system_info (time, date, location)
    if memory_store is not None:
        from . import memory as memory_module

        registry.register(memory_module.build_tool(memory_store))  # 12 memory
    return registry
