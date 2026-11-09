"""Console logging.

pipecat logs through loguru at DEBUG by default, which prints things like
"End of Turn result: EndOfTurnState.COMPLETE" into the middle of a conversation.
Spec §12 wants aggressive logging during M5, but that belongs in
logs/metrics.jsonl, not interleaved with what the assistant is saying.

So the console is quietened to warnings, and the detailed record goes to the
metrics file. `--verbose` puts the library chatter back.
"""

from __future__ import annotations

import sys


def configure(verbose: bool = False) -> None:
    try:
        from loguru import logger
    except ImportError:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "WARNING",
        format="<level>{level: <8}</level> | {name}:{function}:{line} - {message}",
    )
