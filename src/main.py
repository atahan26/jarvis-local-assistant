"""JARVIS entry point.

M0 scope: prove the environment is sound and both external servers are up,
Vulkan-backed, and answering. The conversation loop arrives at M4.

Usage:
    python -m src.main --check        # run all health checks, exit non-zero on failure
    python -m src.main --check-static # files + GPU only, do not require servers
    python -m src.main --listen       # M1: speak, see the transcript
    python -m src.main --devices      # list microphones
"""

from __future__ import annotations

# Non-negotiable §1: refuse to run outside .venv, before importing anything that
# might have been installed into the wrong interpreter.
from .venv_guard import assert_venv

assert_venv()

import argparse  # noqa: E402
import sys  # noqa: E402

from . import config as config_mod  # noqa: E402
from . import health  # noqa: E402

# Only emit ANSI when attached to a real console; otherwise the escapes end up
# as literal noise in redirected logs.
_TTY = sys.stdout.isatty()
RESET = "\033[0m" if _TTY else ""
COLOR = (
    {"PASS": "\033[32m", "FAIL": "\033[31m", "WARN": "\033[33m"} if _TTY else {}
)


def _print(checks: list[health.Check]) -> bool:
    width = max((len(c.name) for c in checks), default=0)
    all_ok = True
    for c in checks:
        color = COLOR.get(c.symbol, "")
        print(f"  {color}[{c.symbol}]{RESET} {c.name.ljust(width)}  {c.detail}")
        if not c.ok and c.fatal:
            all_ok = False
    return all_ok


def run_checks(static_only: bool = False) -> int:
    cfg = config_mod.load()

    print(f"\nJARVIS health check")
    print(f"  project : {cfg.project_root}")
    print(f"  data    : {cfg.data_dir}")
    print(f"  python  : {sys.version.split()[0]}  ({sys.prefix})")

    print("\nEnvironment")
    env_ok = _print(
        [
            health.Check("venv isolated", True, f"{sys.prefix}"),
            health.check_gpu(cfg),
        ]
    )

    print("\nArtifacts")
    artifacts_ok = _print(health.check_artifacts(cfg))

    if static_only:
        ok = env_ok and artifacts_ok
        print(f"\n{'OK' if ok else 'FAILED'} (static checks only)\n")
        return 0 if ok else 1

    print("\nServices")
    service_checks = [health.wait_for_whisper(cfg)]
    if service_checks[-1].ok:
        service_checks.append(health.check_whisper_inference(cfg))
    service_checks.append(health.wait_for_llm(cfg))
    if service_checks[-1].ok:
        service_checks.append(health.check_llm_props(cfg))
        service_checks.append(health.check_llm_offload(cfg, cfg.log_dir))
    services_ok = _print(service_checks)

    ok = env_ok and artifacts_ok and services_ok
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}\n")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--check", action="store_true", help="run health checks and exit")
    parser.add_argument(
        "--check-static",
        action="store_true",
        help="check files and GPU only; do not require the servers to be running",
    )
    parser.add_argument("--listen", action="store_true", help="M1: transcribe the microphone")
    parser.add_argument("--chat", action="store_true", help="M2: typed text, streaming LLM reply")
    parser.add_argument("--speak", action="store_true", help="M3: typed text, spoken aloud")
    parser.add_argument("--talk", action="store_true", help="M4: full spoken conversation")
    parser.add_argument(
        "--flush-test",
        action="store_true",
        help="with --speak, run the scripted 60ms flush-latency test",
    )
    parser.add_argument("--out-device", default=None, help="output device index or name substring")
    parser.add_argument("--verbose", action="store_true", help="with --talk, print state transitions")
    parser.add_argument("--devices", action="store_true", help="list input devices and exit")
    parser.add_argument(
        "--device", default=None, help="input device index or name substring (overrides config)"
    )
    parser.add_argument(
        "--meter", action="store_true", help="with --listen, show a live VAD confidence meter"
    )
    args = parser.parse_args(argv)

    if args.devices:
        from .audio.io import list_input_devices

        print("\nInput devices:")
        list_input_devices()
        print()
        return 0

    if args.check_static:
        return run_checks(static_only=True)
    if args.check:
        return run_checks()

    if args.listen:
        from . import listen

        return listen.run(
            config_mod.load(), show_meter=args.meter, device_override=args.device
        )

    if args.chat:
        from . import chat

        return chat.run(config_mod.load())

    if args.speak:
        from . import speak

        cfg = config_mod.load()
        if args.flush_test:
            return speak.run_flush_test(cfg, device_override=args.out_device)
        return speak.run(cfg, device_override=args.out_device)

    if args.talk:
        from . import conversation, logging_setup

        # Library DEBUG chatter belongs in metrics.jsonl, not interleaved with
        # the conversation.
        logging_setup.configure(verbose=args.verbose)
        return conversation.run(
            config_mod.load(),
            device_override=args.device,
            out_device_override=args.out_device,
            verbose=args.verbose,
        )

    print("Nothing to do. Pick one:")
    print("  --talk               M4: full spoken conversation")
    print("  --check              verify the environment and both servers")
    print("  --listen             M1: speak into the mic, see the transcript")
    print("  --chat               M2: type text, watch the reply stream back")
    print("  --speak              M3: type text, hear it spoken")
    print("  --speak --flush-test M3: measure interrupt latency against the 60ms budget")
    print("  --devices            list microphones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
