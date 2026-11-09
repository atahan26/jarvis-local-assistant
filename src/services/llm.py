"""llama-server client (OpenAI-compatible, streaming).

Spec §3 budgets 250 ms to first token and §7.3 requires cancelling an in-flight
generation within 40 ms of a barge-in. Both of those shape this module:

* Generation is streamed, never awaited as a whole. Nothing downstream may wait
  for the full response.
* Every stream is cancellable. Closing the HTTP response makes llama-server stop
  generating and frees the slot, so cancellation is real, not just ignoring
  output we already paid for.
* The caller is told exactly how much text was actually emitted before a cancel,
  because §7.3 says only what the user really heard may enter history.

Qwen3 reasoning is disabled server-side by run.ps1 (`--reasoning off`). If that
is ever turned back on, `reasoning_content` deltas are surfaced separately so
they can never be mistaken for spoken text.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import httpx


@dataclass
class Delta:
    """One streamed piece of a response."""

    text: str
    elapsed_ms: float
    is_first: bool = False
    is_reasoning: bool = False


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    `arguments` arrives as JSON fragments spread across many streamed deltas and
    is accumulated here as a raw string; validation happens in the registry, not
    on the wire.
    """

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""

    def as_message_part(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


@dataclass
class Generation:
    """The outcome of one streamed completion."""

    text: str = ""
    reasoning: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0
    cancelled: bool = False
    cancel_ms: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def token_estimate(self) -> int:
        # llama-server does not report usage on every chunk; a rough count is
        # enough for the throughput line in the metrics log.
        return max(1, len(self.text) // 4)


class LLMClient:
    def __init__(
        self,
        base_url: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        timeout_s: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._timeout = timeout_s
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- health ------------------------------------------------------------

    def is_up(self) -> bool:
        try:
            return self._client.get(f"{self.base_url}/health", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def model_name(self) -> str:
        try:
            props = self._client.get(f"{self.base_url}/props", timeout=5.0).json()
            return str(props.get("model_path", "?")).replace("\\", "/").split("/")[-1]
        except (httpx.HTTPError, ValueError):
            return "?"

    # -- generation --------------------------------------------------------

    def _payload(
        self,
        messages: list[dict],
        max_tokens: int | None,
        tools: list[dict] | None = None,
    ) -> dict:
        body: dict = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if not self.enable_thinking:
            # Belt and braces: run.ps1 already passes --reasoning off, but a
            # server someone else started should still behave.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def stream(
        self,
        messages: list[dict],
        *,
        cancel: threading.Event | None = None,
        max_tokens: int | None = None,
        into: Generation | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[Delta]:
        """Yield deltas as they arrive.

        Pass `into` to receive the accumulated Generation even if the caller
        abandons the iterator early, which is what a barge-in looks like.
        """
        result = into if into is not None else Generation()
        started = time.monotonic()
        first_seen = False
        pending: dict[int, ToolCall] = {}

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=self._payload(messages, max_tokens, tools),
            ) as response:
                if response.status_code != 200:
                    response.read()
                    result.error = f"HTTP {response.status_code}: {response.text[:200]}"
                    return

                for line in response.iter_lines():
                    if cancel is not None and cancel.is_set():
                        cancel_started = time.monotonic()
                        result.cancelled = True
                        response.close()
                        result.cancel_ms = (time.monotonic() - cancel_started) * 1000.0
                        break

                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # llama-server reports mid-stream failures INSIDE a 200
                    # response, e.g. a lost Vulkan device:
                    #   data: {"error":{"code":500,"message":"decode() failed:
                    #          vk::Queue::submit: ErrorDeviceLost"}}
                    # Ignoring these makes a GPU crash look like an empty reply
                    # and the assistant goes silently mute.
                    if isinstance(chunk.get("error"), dict):
                        detail = chunk["error"].get("message", "unknown server error")
                        result.error = f"server error mid-stream: {detail}"
                        return
                    if chunk.get("error"):
                        result.error = f"server error mid-stream: {chunk['error']}"
                        return

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]

                    # Tool calls arrive as fragments: the name comes once, then
                    # the JSON arguments a few characters at a time, keyed by
                    # index. Accumulate rather than overwrite.
                    for part in delta.get("tool_calls") or []:
                        idx = int(part.get("index", 0))
                        call = pending.setdefault(idx, ToolCall(index=idx))
                        if part.get("id"):
                            call.id = part["id"]
                        function = part.get("function") or {}
                        if function.get("name"):
                            call.name = function["name"]
                        if function.get("arguments"):
                            call.arguments += function["arguments"]

                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        result.reasoning += reasoning
                        yield Delta(
                            text=reasoning,
                            elapsed_ms=(time.monotonic() - started) * 1000.0,
                            is_reasoning=True,
                        )

                    text = delta.get("content")
                    if not text:
                        continue

                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    is_first = not first_seen
                    if is_first:
                        first_seen = True
                        result.ttft_ms = elapsed_ms
                    result.text += text
                    yield Delta(text=text, elapsed_ms=elapsed_ms, is_first=is_first)

        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.total_ms = (time.monotonic() - started) * 1000.0
            # Populate even on cancellation, so a partially streamed tool call
            # is visible to the caller rather than silently lost.
            result.tool_calls = [pending[i] for i in sorted(pending) if pending[i].name]

    def complete(
        self,
        messages: list[dict],
        *,
        cancel: threading.Event | None = None,
        max_tokens: int | None = None,
    ) -> Generation:
        """Convenience wrapper that drains the stream."""
        result = Generation()
        for _ in self.stream(messages, cancel=cancel, max_tokens=max_tokens, into=result):
            pass
        return result


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a voice assistant running locally on the user's Windows PC. "
    "Your replies are spoken aloud, so keep them short and conversational: "
    "one or two sentences unless asked for detail. "
    "Never use markdown, bullet points, emoji, or code blocks. "
    "Never write stage directions or bracketed asides; every character you write "
    "is read out loud. "
    "Write numbers and units the way they should be read out loud. "
    "If one of your earlier replies stops mid-sentence, it is because the user "
    "interrupted you. Do not comment on it or repeat that reply unless asked. "
    "If you do not know something, say so briefly."
)


@dataclass
class Conversation:
    """Message history with interruption-aware appends (spec §7.3)."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    messages: list[dict] = field(default_factory=list)
    max_turns: int = 20

    def for_request(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str, *, interrupted: bool = False) -> None:
        """Record what the assistant said.

        §7.3: on a barge-in, only the text the user actually HEARD may be
        appended. Text that was generated but never played must be dropped, or
        the model will act as though the user knows things they never heard.

        `interrupted` is deliberately NOT written into the message text. An
        earlier version appended "[interrupted by user]" to the content, and the
        model promptly learned the pattern and started emitting that string in
        its own replies, which then got spoken aloud. The truncation itself is
        the signal, and the system prompt explains it; anything in-band here is
        training data for the next turn.
        """
        content = text.strip()
        if not content:
            return
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_calls(self, calls: list[ToolCall]) -> None:
        """Record the assistant's request to run tools."""
        if not calls:
            return
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.as_message_part() for c in calls],
            }
        )

    def add_tool_result(self, call: ToolCall, output: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output}
        )

    def _trim(self) -> None:
        limit = self.max_turns * 2
        if len(self.messages) <= limit:
            return
        trimmed = self.messages[-limit:]
        # A tool result whose originating assistant message was trimmed away is
        # rejected by the server ("tool_call_id not found"), so drop orphans.
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed.pop(0)
        self.messages = trimmed

    def reset(self) -> None:
        self.messages.clear()
