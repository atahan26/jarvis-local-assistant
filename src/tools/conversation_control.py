"""Controlling the assistant itself (spec §8.3 tool 10).

"Stop talking", "never mind", "go to sleep". These do not touch the OS; they
change what the assistant is doing, which means they need to reach the running
Assistant rather than just return text.

Rather than handing tools a reference to the Assistant (which would let any tool
reach into the loop), a control tool returns a directive in `ToolResult.data`
under the key `control`. The conversation loop is the only thing that reads it,
so the coupling stays one-way and auditable.
"""

from __future__ import annotations

from .registry import Tool, ToolError, ToolResult

# Directives the conversation loop understands.
SLEEP = "sleep"
STOP_SPEAKING = "stop_speaking"
FORGET = "forget"
REPEAT = "repeat"


def conversation_control(action: str) -> ToolResult:
    if action == "sleep":
        return ToolResult(
            True,
            "Going to sleep. Say hey Jarvis when you need me.",
            {"control": SLEEP},
        )
    if action == "stop_talking":
        return ToolResult(True, "", {"control": STOP_SPEAKING})
    if action == "repeat":
        return ToolResult(True, "", {"control": REPEAT})
    if action == "forget":
        return ToolResult(True, "Forgotten.", {"control": FORGET})
    raise ToolError(f"unknown action {action!r}")


TOOL = Tool(
    name="conversation_control",
    description=(
        "Control the assistant itself: go to sleep until the wake word is said "
        "again, stop talking, repeat the last answer, or forget the conversation "
        "so far. Use 'sleep' when the user says to go to sleep, stand by, stop "
        "listening, or that they are done."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sleep", "stop_talking", "repeat", "forget"],
                "description": "What the assistant should do with itself.",
            }
        },
        "required": ["action"],
    },
    handler=conversation_control,
)
