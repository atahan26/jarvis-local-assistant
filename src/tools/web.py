"""Web search via DuckDuckGo (spec §8.2, §8.3 tool 5).

Two things shape this:

* **It is slow.** Measured at ~2.3 s for three results, which is longer than an
  entire normal turn. The description therefore tells the model to use it only
  for things it cannot answer itself; without that, Qwen3 searched for "capital
  of France" in testing rather than just answering.
* **The result is about to be spoken.** Titles and URLs read terribly aloud, so
  the summary is prose built from the result snippets, not a list of links.
"""

from __future__ import annotations

import re

from .registry import Tool, ToolError, ToolResult

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", _TAGS.sub(" ", text or "")).strip()


def web_search(query: str, max_results: int = 3) -> ToolResult:
    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover
        raise ToolError("the ddgs package is not installed") from exc

    query = query.strip()
    if not query:
        raise ToolError("empty search query")

    max_results = max(1, min(5, int(max_results)))
    try:
        with DDGS() as client:
            raw = list(client.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - network, rate limits, parser changes
        raise ToolError(f"search failed: {type(exc).__name__}") from exc

    if not raw:
        return ToolResult(True, f"I couldn't find anything for {query}.", {"results": []})

    results = []
    for item in raw[:max_results]:
        results.append(
            {
                "title": _clean(str(item.get("title", ""))),
                "body": _clean(str(item.get("body", ""))),
                "url": str(item.get("href", "")),
            }
        )

    # Spoken summary: the snippets carry the answer; URLs are kept in data for
    # anything that wants them later, but never read out.
    spoken = " ".join(r["body"] for r in results if r["body"])[:600]
    if not spoken:
        spoken = " ".join(r["title"] for r in results)[:300]

    return ToolResult(
        True,
        f"Here's what I found about {query}: {spoken}",
        {"results": results, "query": query},
    )


TOOL = Tool(
    name="web_search",
    description=(
        "Search the web. Use this ONLY for current events, live data such as "
        "weather or prices, or facts you genuinely do not know. Do not use it "
        "for general knowledge you can already answer, and never for questions "
        "about this computer. The query must name a real place or subject: if "
        "the user says 'here' or 'where I live', call system_info with action "
        "location first and search for that city by name."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "max_results": {
                "type": "integer",
                "description": "How many results to read, 1 to 5. Default 3.",
            },
        },
        "required": ["query"],
    },
    handler=web_search,
)
