"""M2: typed text -> llama-server -> streaming text out.

No microphone and no speech here. This isolates the LLM leg so TTFT is measured
without VAD or whisper in the way, which is exactly what spec §9 asks of M2.

Type a message and watch tokens arrive. Ctrl+C during generation exercises the
cancellation path that M5's barge-in depends on.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time

from .config import Config
from .services.llm import Conversation, Generation, LLMClient

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _tty() -> bool:
    return sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return text if not _tty() else f"{color}{text}{RESET}"


TTFT_BUDGET_MS = 250.0   # spec §3
CANCEL_BUDGET_MS = 40.0  # spec §7.3


def _report(gen: Generation, ttfts: list[float]) -> None:
    if gen.error:
        print(c(f"\n  ! {gen.error}", RED))
        return

    bits = []
    if gen.ttft_ms is not None:
        ttfts.append(gen.ttft_ms)
        mark = "" if gen.ttft_ms <= TTFT_BUDGET_MS else f" (over {TTFT_BUDGET_MS:.0f}ms budget)"
        bits.append(f"TTFT {gen.ttft_ms:.0f}ms{mark}")
    bits.append(f"total {gen.total_ms:.0f}ms")
    if gen.total_ms > 0 and gen.text:
        bits.append(f"~{gen.token_estimate / (gen.total_ms / 1000.0):.0f} tok/s")
    if gen.cancelled:
        over = "" if (gen.cancel_ms or 0) <= CANCEL_BUDGET_MS else " OVER BUDGET"
        bits.append(c(f"cancelled in {gen.cancel_ms:.1f}ms{over}", YELLOW))
    if gen.finish_reason and gen.finish_reason != "stop":
        bits.append(f"finish={gen.finish_reason}")
    if gen.reasoning:
        bits.append(c(f"{len(gen.reasoning)} chars of reasoning (thinking is ON)", YELLOW))

    print(c("    " + " | ".join(bits), DIM))


def run(cfg: Config) -> int:
    llm = LLMClient(
        cfg.llm_url,
        temperature=float(cfg.get("llm.temperature", 0.7)),
        top_p=float(cfg.get("llm.top_p", 0.8)),
        max_tokens=int(cfg.get("llm.max_tokens", 512)),
        enable_thinking=bool(cfg.get("llm.enable_thinking", False)),
    )

    if not llm.is_up():
        print(c(f"llama-server is not answering at {cfg.llm_url}", RED))
        print("Start it with:  .\\run.ps1")
        return 1

    convo = Conversation()
    ttfts: list[float] = []

    print()
    print(c("JARVIS M2: streaming LLM", BOLD))
    print(f"  server : {cfg.llm_url}")
    print(f"  model  : {llm.model_name()}")
    print(
        f"  sampling: temp {llm.temperature}, top_p {llm.top_p}, "
        f"max_tokens {llm.max_tokens}, thinking {'on' if llm.enable_thinking else 'off'}"
    )
    print()
    print(c("  Type a message and press Enter. Ctrl+C mid-answer tests cancellation.", DIM))
    print(c("  /reset clears history, /quit exits.", DIM))
    print()

    try:
        while True:
            try:
                prompt = input(c("you > ", CYAN)).strip()
            except EOFError:
                break

            if not prompt:
                continue
            if prompt in ("/quit", "/exit"):
                break
            if prompt == "/reset":
                convo.reset()
                print(c("  history cleared", DIM))
                continue

            convo.add_user(prompt)
            gen = Generation()
            cancel = threading.Event()
            interrupted = False

            sys.stdout.write(c("jarvis > ", GREEN))
            sys.stdout.flush()
            stream = llm.stream(convo.for_request(), cancel=cancel, into=gen)
            try:
                for delta in stream:
                    if delta.is_reasoning:
                        continue
                    sys.stdout.write(delta.text)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                # Same shape as a barge-in: stop generating and keep only what
                # was actually shown. Closing the generator raises GeneratorExit
                # at the yield, which unwinds the httpx stream context and drops
                # the connection, so llama-server really stops and frees the slot.
                interrupted = True
                cancel.set()
                cancel_started = time.monotonic()
                stream.close()
                gen.cancelled = True
                gen.cancel_ms = (time.monotonic() - cancel_started) * 1000.0
                print(c("  [interrupted]", YELLOW))

            print()
            convo.add_assistant(gen.text, interrupted=interrupted)
            _report(gen, ttfts)
            print()
    except KeyboardInterrupt:
        print()
    finally:
        llm.close()

    if ttfts:
        ordered = sorted(ttfts)
        p95 = ordered[max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))]
        print()
        print(c("=" * 60, DIM))
        print(f"  {len(ttfts)} generations")
        print(f"  TTFT median {statistics.median(ordered):.0f}ms | p95 {p95:.0f}ms | max {ordered[-1]:.0f}ms")
        median = statistics.median(ordered)
        if median <= TTFT_BUDGET_MS:
            print(c(f"  M2 target met: TTFT median <= {TTFT_BUDGET_MS:.0f}ms", GREEN))
        else:
            print(c(f"  M2 target missed: TTFT median {median:.0f}ms > {TTFT_BUDGET_MS:.0f}ms", YELLOW))
        print(c("=" * 60, DIM))
    return 0
