"""Tool registry: schema generation, dispatch, and permissions (spec §8.1).

Non-negotiable §5: "Tool calls that touch the OS must be explicitly registered
and permissioned. No arbitrary code execution from the LLM." So the model never
names a Python callable. It names a registered tool, the arguments are validated
against a declared schema, and anything not registered simply does not exist.

§8.1 also shapes the design:

* Few tools, grouped by an `action` enum, because a local 8B model degrades badly
  past roughly 10-15 of them.
* Every tool declares a JSON schema, a one-line natural description, and a
  `requires_confirmation` flag.
* Destructive actions are gated behind spoken confirmation rather than executed
  optimistically.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    """What a tool hands back to the model."""

    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def for_model(self) -> str:
        """The string the model sees as the tool's output.

        Kept short and plain: it is about to be turned into speech, and a local
        model will happily read a JSON blob out loud if handed one.
        """
        return self.summary if self.ok else f"Failed: {self.summary}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    requires_confirmation: bool = False
    # Actions within the tool that are destructive even if the tool as a whole
    # is not. "close" needs confirming; "launch" does not.
    confirm_actions: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def needs_confirmation(self, arguments: dict[str, Any]) -> bool:
        if self.requires_confirmation:
            return True
        action = str(arguments.get("action", "")).lower()
        return action in self.confirm_actions


class ToolError(Exception):
    """Raised by a handler when the request is understood but cannot be done."""


class Registry:
    def __init__(self, *, max_exposed: int = 15):
        self._tools: dict[str, Tool] = {}
        self.max_exposed = max_exposed

    # -- registration ------------------------------------------------------

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    # -- schemas -----------------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions for the OpenAI-compatible `tools` field.

        §8.1: past roughly 15 tools an 8B model starts inventing arguments, so
        exceeding the cap is surfaced rather than silently tolerated. The
        intent-router pre-filter the spec describes belongs here when that day
        comes.
        """
        if len(self._tools) > self.max_exposed:
            raise RuntimeError(
                f"{len(self._tools)} tools registered but max_exposed is "
                f"{self.max_exposed}. Add an intent router before going further "
                "(spec §8.1); local models hallucinate arguments past this point."
            )
        return [t.schema() for t in self._tools.values()]

    # -- dispatch ----------------------------------------------------------

    def parse_arguments(self, tool: Tool, raw: str | dict | None) -> dict[str, Any]:
        if isinstance(raw, dict):
            arguments = raw
        elif not raw:
            arguments = {}
        else:
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolError(f"arguments were not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a JSON object")
        return arguments

    def validate(self, tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check arguments against the declared schema.

        Deliberately strict about unknown keys: a hallucinated argument is a
        signal the model misunderstood, and passing it through to a handler that
        touches the OS is exactly what §5 forbids.
        """
        schema = tool.parameters
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolError(f"missing required argument(s): {', '.join(missing)}")

        cleaned: dict[str, Any] = {}
        for key, value in arguments.items():
            if key not in properties:
                continue  # drop hallucinated extras rather than forwarding them
            spec = properties[key]
            expected = spec.get("type")
            if expected == "integer":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ToolError(f"{key} must be a whole number, got {value!r}") from None
            elif expected == "number":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ToolError(f"{key} must be a number, got {value!r}") from None
            elif expected == "boolean":
                if isinstance(value, str):
                    value = value.strip().lower() in ("true", "yes", "1")
                value = bool(value)
            elif expected == "string":
                value = str(value)
            elif expected == "array":
                # Models hand back a bare string when they mean a one-item list,
                # or a JSON-encoded list as a string. Accept both.
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("["):
                        try:
                            value = json.loads(stripped)
                        except json.JSONDecodeError:
                            value = [stripped]
                    else:
                        value = [stripped]
                if not isinstance(value, list):
                    raise ToolError(f"{key} must be a list, got {value!r}")
                item_type = (spec.get("items") or {}).get("type")
                if item_type == "string":
                    value = [str(v) for v in value]

            choices = spec.get("enum")
            if choices and value not in choices:
                # Models often get the case wrong before anything else.
                lowered = {str(c).lower(): c for c in choices}
                if str(value).lower() in lowered:
                    value = lowered[str(value).lower()]
                else:
                    raise ToolError(
                        f"{key} must be one of {', '.join(map(str, choices))}, got {value!r}"
                    )
            cleaned[key] = value
        return cleaned

    def call(self, name: str, raw_arguments: str | dict | None) -> ToolResult:
        started = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            # The model invented a tool. Say so plainly; never guess a match.
            return ToolResult(
                ok=False,
                summary=f"There is no tool called {name}.",
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
        try:
            arguments = self.parse_arguments(tool, raw_arguments)
            arguments = self.validate(tool, arguments)
            result = tool.handler(**arguments)
        except ToolError as exc:
            result = ToolResult(ok=False, summary=str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the loop
            result = ToolResult(ok=False, summary=f"{type(exc).__name__}: {exc}")
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result
