# AGENTS.md: guide for AI coding agents

For any coding agent (Claude Code, Codex, Cursor, Copilot, ...) working in this repository.
Read `README.md` first, then this file.

## What this is
A local, Windows-only voice assistant. Pipeline: mic -> Silero VAD -> turn detection -> whisper.cpp (HTTP server) ->
llama.cpp server running Qwen3-8B (OpenAI-compatible, tool calling) -> Kokoro TTS -> speakers. Barge-in cancels audio
and the LLM request. Python 3.11, everything runs in `.venv`.

## Run
```powershell
.\setup.ps1                                          # one time: venv, deps, binaries, ~5.5 GB models
.\run.ps1                                            # start whisper-server and llama-server
.\.venv\Scripts\python.exe -m src.main --chat        # typed chat, no audio hardware needed
.\.venv\Scripts\python.exe -m unittest discover tests # live tests skip when the servers are down
```
`src/venv_guard.py` refuses to run outside `.venv`. Thresholds live in `config.yaml` only; do not hardcode them in Python.

## Map
- `src/main.py`: entry point and modes (`--talk`, `--listen`, `--chat`, `--speak`).
- `src/conversation.py`: the orchestrator and its state machine (idle, listening, thinking, speaking).
- `src/audio/`: capture (`io.py`), VAD and endpointing (`vad.py`), playback queue (`output.py`), barge-in (`barge_in.py`), wake word, turn model.
- `src/services/`: clients for whisper-server, llama-server (streaming, cancellable) and Kokoro.
- `src/tools/`: the 11 tools. `registry.py` registers them, validates arguments against the declared schema and refuses unknown names.
- `src/config.py`: YAML loading, deep-merged with the gitignored `config.local.yaml`.
- `tests/`: unit tests; the live ones skip when the servers are down.

## Adding a tool
1. Create the handler in `src/tools/` and register it in `registry.py` with a JSON schema. Group related actions under one tool with an `action` enum.
2. Keep the exposed tool count under `tools.max_exposed` (an 8B model degrades beyond roughly 10-15 tools). `Registry.schemas()` refuses to exceed it.
3. If the action is destructive or can inject input, gate it behind a spoken yes: declare `confirm_actions=("close",)` on the tool (see `src/tools/apps.py`) or set `requires_confirmation=True` (see `src/tools/media.py`).
4. Write a test in `tests/test_tools.py` style.

## Safety rules (follow these)
1. **No arbitrary code execution from the model.** Never add a shell passthrough, `eval`, or a tool that runs free-form commands. `app_control` resolves names against an allowlist; keep it that way.
2. **Keep the registry strict.** Unregistered tool names are refused, arguments are schema-validated, extra arguments are dropped.
3. **Keep confirmation gating** for `close`, `send_hotkey` and `forget_all`. Anything that is not clearly a yes counts as a no.
4. **URL validation stays:** https only, no credentials, no ports, no IP literals, no `file://` or `javascript:`. Validate URL shape, never subject matter.
5. **No cloud inference and no telemetry.** The only network calls are `web_search`, the location lookup in `system_info`, and `setup.ps1` downloads.
6. **Never commit** models, binaries (`*.gguf`, `*.onnx`, `*.bin`, `*.zip`), `logs/`, `.env`, `config.local.yaml` or `memory.json` (it holds personal facts).
7. Discord is driven by keyboard shortcuts only. Do not import a Discord library or automate the client's API.

## Style
Readable Python with type hints and `from __future__ import annotations`. Match the surrounding code, and explain measured results in comments when you tune a value.
