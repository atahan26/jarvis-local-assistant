# JARVIS — Local Voice Assistant Build Spec

## 0. Context for the implementer

Target machine:

- **OS:** Windows (10/11)
- **GPU:** AMD Radeon RX 6700 XT — 12GB VRAM, architecture `gfx1031`
- **CPU:** AMD Ryzen 5 X3D (6 cores, large L3 cache)
- **RAM:** 32GB
- **Language:** English only (assistant speaks and listens in English)

The critical hardware fact: `gfx1031` is **not** on ROCm's official support list, and llama.cpp's ROCm/HIP backend is Linux-only anyway. Every GPU-accelerated component in this project must therefore run on the **Vulkan** backend. Vulkan is roughly 60–80% of ROCm's speed but works out of the box on Windows.

Two consequences that must not be forgotten:

1. **Do not use `faster-whisper` or anything built on CTranslate2.** CTranslate2 has no ROCm/Vulkan path. On this machine it silently falls back to CPU and destroys the latency budget. Use `whisper.cpp` with the Vulkan build instead.
2. **Do not use NVIDIA NeMo / Parakeet.** CUDA-only.

---

## 1. Non-negotiables

These are hard requirements. Do not substitute them for convenience.

| #   | Requirement                                                                                                                                                     |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Everything runs in a **Python virtual environment** (`.venv` in the project root). Never `pip install` into the system Python. Never use `--user`.              |
| 2   | Everything is **free and local**. No paid APIs, no cloud inference, no API keys required for the core loop.                                                     |
| 3   | **Barge-in must work.** The user must be able to interrupt the assistant mid-sentence and be heard. This is the primary success criterion of the whole project. |
| 4   | The assistant must **not cut the user off** mid-thought during natural pauses.                                                                                  |
| 5   | Tool calls that touch the OS must be **explicitly registered and permissioned** — no arbitrary code execution from the LLM.                                     |

---

## 2. Architecture

Cascaded streaming pipeline. Native speech-to-speech models (Qwen3-Omni, Moshi) are explicitly rejected: they are either too slow to self-host in real time or English/French-only research artifacts. Cascaded streaming is the correct architecture here.

```
Microphone (16kHz mono, 20ms frames, ALWAYS OPEN)
    |
    v
[Silero VAD v5]  --> is there speech right now?
    |
    v
[Turn detector]  --> has the user finished their turn?
    |
    v
[whisper.cpp server, Vulkan]  --> transcript
    |
    v
[llama-server, Vulkan, OpenAI-compatible]  --> streaming text + tool calls
    |                                              |
    |                                              v
    |                                        [Tool registry]
    |                                              |
    v                                              v
[Kokoro-82M TTS, CPU]  <-- sentence-chunked, cancellable
    |
    v
Speaker / headphones
```

The mic stream **never closes**, including while TTS is playing. That is what makes barge-in possible.

---

## 3. Component choices

| Layer          | Choice                                                                       | Why                                                                                                                               | Runs on          |
| -------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Orchestration  | **Pipecat** (`pipecat-ai`)                                                   | Highest-scoring open framework for interruption handling; pure Python; pip-installable; has `LocalAudioTransport` for desktop use | CPU              |
| Audio I/O      | `sounddevice` (PortAudio)                                                    | Low-latency callback API, cancellable output stream                                                                               | CPU              |
| VAD            | **Silero VAD v5** (ONNX)                                                     | ~1ms per frame, industry default                                                                                                  | CPU              |
| Turn detection | **Pipecat SmartTurn** (local ONNX)                                           | Learned model that distinguishes backchannel ("mhm", "yeah") from real barge-in, instead of a dumb energy threshold               | CPU              |
| Wake word      | **openWakeWord** — pretrained `hey_jarvis` model                             | Free, ONNX, and the model literally already exists for this wake phrase                                                           | CPU              |
| STT            | **whisper.cpp** `small.en` (Q5_K), Vulkan build, run as `whisper-server.exe` | Only Whisper implementation with a working AMD path on Windows. English-only model is smaller and faster at equal accuracy        | **GPU (Vulkan)** |
| LLM            | **Qwen3-8B Q4_K_M** via `llama-server.exe`, Vulkan build                     | Strong tool-calling for its size; fits the VRAM budget with room for context                                                      | **GPU (Vulkan)** |
| TTS            | **Kokoro-82M** via `kokoro-onnx`                                             | Apache-2.0, ~327MB, natural output, fast enough on CPU that it never competes for VRAM                                            | CPU              |

### VRAM budget (12GB total)

| Component                                  | Approx. VRAM                       |
| ------------------------------------------ | ---------------------------------- |
| whisper.cpp `small.en` Q5                  | ~0.6 GB                            |
| Qwen3-8B Q4_K_M, 16k ctx, KV cache at q8_0 | ~6.5 GB                            |
| Kokoro                                     | 0 (CPU)                            |
| **Total**                                  | **~7.1 GB** — comfortable headroom |

Do not upgrade to a 14B model. It fits, but time-to-first-token is what matters in a voice loop, not throughput, and 14B makes TTFT worse.

### Latency budget (target)

| Stage                             | Target             |
| --------------------------------- | ------------------ |
| Endpoint detection → STT complete | ~300 ms            |
| LLM time-to-first-token           | ~250 ms            |
| TTS first audio chunk             | ~150 ms            |
| **Total round trip**              | **900 ms – 1.5 s** |
| TTS flush on barge-in             | **< 60 ms**        |
| LLM cancel on barge-in            | **< 40 ms**        |

---

## 4. Environment setup

Use **Python 3.11** — best wheel availability for this stack. Verify before proceeding.

```powershell
cd C:\projects\jarvis
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Every subsequent command in this project runs with `.venv` active. Add a check at the top of the entry point that refuses to run if `sys.prefix == sys.base_prefix`.

Create `requirements.txt`. **Verify current package names and extras against the Pipecat docs before pinning** — extras names have changed across releases:

```
pipecat-ai[silero,local]
sounddevice
numpy
onnxruntime
openwakeword
kokoro-onnx
httpx
pydantic
python-dotenv
pywinauto
uiautomation
pynput
psutil
pycaw
pyperclip
ddgs
```

`.gitignore` must include `.venv/`, `models/`, `*.gguf`, `*.onnx`, `*.bin`.

---

## 5. External binaries (not pip-installable)

These are downloaded separately into `bin/` and `models/`. Write a `setup.ps1` that fetches them.

### whisper.cpp (Vulkan, Windows x64)

- Download the **Vulkan** Windows release from the whisper.cpp GitHub releases page. Confirm the asset name at build time — do not hardcode a guessed filename.
- Download `ggml-small.en-q5_1.bin` from the whisper.cpp models repo on Hugging Face.
- Launch as a background HTTP service:

```powershell
.\bin\whisper\whisper-server.exe -m .\models\ggml-small.en-q5_1.bin --port 8081 -t 4
```

Wrapping the server over HTTP avoids Python binding compilation hell on Windows. This is deliberate.

### llama.cpp (Vulkan, Windows x64)

- Download `llama-<build>-bin-win-vulkan-x64.zip` from the llama.cpp releases page.
- Download `Qwen3-8B-Q4_K_M.gguf`.
- Launch as an OpenAI-compatible server:

```powershell
.\bin\llama\llama-server.exe `
  -m .\models\Qwen3-8B-Q4_K_M.gguf `
  -ngl 999 `
  -c 16384 `
  -fa `
  --cache-type-k q8_0 --cache-type-v q8_0 `
  --host 127.0.0.1 --port 8080
```

Verify `-ngl 999` actually offloads all layers to the GPU — check the startup log for Vulkan device detection. If it reports CPU-only, the Vulkan runtime or driver is the problem, not the flags.

**Fallback:** if the llama.cpp Vulkan build misbehaves, LM Studio's Vulkan backend is the lower-friction Windows path for AMD and also exposes an OpenAI-compatible endpoint. The rest of the code is unaffected — only the base URL changes.

### Kokoro

- `kokoro-v1.0.onnx` and `voices-v1.0.bin` into `models/kokoro/`.
- Default voice: pick one and expose it in config.

### openWakeWord

- Downloads its own ONNX models on first run, including `hey_jarvis`.

---

## 6. Project structure

```
jarvis/
├── .venv/
├── bin/
│   ├── whisper/
│   └── llama/
├── models/
├── config.yaml
├── requirements.txt
├── setup.ps1
├── run.ps1                  # starts whisper-server, llama-server, then the agent
├── src/
│   ├── __init__.py
│   ├── main.py              # entry point, venv guard, service health checks
│   ├── pipeline.py          # Pipecat pipeline assembly
│   ├── services/
│   │   ├── whisper_stt.py   # Pipecat STTService -> whisper.cpp HTTP
│   │   ├── kokoro_tts.py    # Pipecat TTSService -> kokoro-onnx, chunked + cancellable
│   │   └── llm.py           # OpenAI-compatible client -> llama-server
│   ├── audio/
│   │   ├── io.py            # sounddevice streams
│   │   └── barge_in.py      # interruption policy
│   ├── tools/
│   │   ├── registry.py      # schema generation + dispatch + permissions
│   │   ├── system.py
│   │   ├── apps.py
│   │   ├── media.py
│   │   └── web.py
│   └── config.py
└── tests/
```

---

## 7. Conversation loop requirements

**This is the hard part. Everything else is plumbing.** The previous attempts at this project failed here specifically: the assistant would cut the user off mid-sentence, ignore the user while speaking, and refuse to accept a new command until it had finished its answer. All three of those are turn-detection failures.

### 7.1 Acoustic echo cancellation

Without AEC, the microphone hears the assistant's own TTS output and interrupts itself in a loop. Echo cancellation must be handled on the client side, at the speaker.

**Primary approach: assume headphones.** Document this in the README as the supported configuration. It eliminates the problem entirely and costs nothing.

**Speaker fallback (implement as a config-gated mode):** while TTS is playing, require _all_ of the following before accepting a barge-in:

- Silero VAD probability > 0.6
- Sustained speech for ≥ 250 ms continuously
- Frame energy above a calibrated threshold

This is not real AEC. It is a heuristic that trades some barge-in responsiveness for immunity to self-triggering. Make the thresholds configurable.

### 7.2 Endpointing (do not cut the user off)

- Base silence threshold: **700 ms**, not the 500 ms most tutorials use.
- Feed the audio to the SmartTurn analyzer. If it classifies the turn as incomplete, extend the wait up to a 3000 ms ceiling.
- Backchannel utterances ("mhm", "yeah", "okay", "right") detected during assistant speech must be classified as **backchannel, not barge-in** — the assistant keeps talking. This is exactly what a raw energy threshold gets wrong.
- Expose `base_silence_ms`, `max_wait_ms`, and `vad_threshold` in `config.yaml`. These will need tuning against the actual microphone.

### 7.3 Barge-in (accept interruptions mid-answer)

Turn detection stays **fully active during TTS playback**. On a confirmed barge-in:

1. Clear the TTS audio queue and stop the output stream within **60 ms**. The queue must be flushable — do not synthesize the whole response into one blob.
2. Cancel the in-flight LLM stream within **40 ms**.
3. Append whatever the assistant actually spoke to conversation history, marked as interrupted. Do **not** append text that was generated but never played — the user never heard it, so the model must not act as if they did.
4. Hand control back to STT immediately.

### 7.4 Tool calls during interruption

If a tool call is in flight when the user interrupts, the default policy is: **let the tool finish in the background, store the result in conversation state, surface it on the next turn if still relevant.** Never lose work, never block the next turn on the previous turn's tool. Cancel the tool only when the new user intent directly contradicts it.

### 7.5 State machine

```
IDLE ──wake word──> LISTENING
LISTENING ──turn complete──> THINKING
THINKING ──first token──> SPEAKING
SPEAKING ──barge-in──> LISTENING   (flush TTS, cancel LLM)
SPEAKING ──finished──> IDLE or LISTENING (config: follow-up window)
```

Wake word required in IDLE only. After a response, stay in an open follow-up window (default 8s, configurable) so the user does not have to say "hey Jarvis" for every turn in a back-and-forth.

### 7.6 Metrics

Log these per turn. They are the debugging surface when it feels wrong:

- barge-in success rate (target > 96%)
- false barge-in rate (target < 2%)
- TTS flush latency
- LLM cancel latency
- turn-taking gap, P95 (target 200–450 ms)

---

## 8. Tool layer

### 8.1 Design

Local models degrade badly at function calling as the tool count grows. A 8B model handles roughly **10–15 well-described tools** reliably and starts hallucinating arguments beyond that.

Therefore:

- Keep the exposed tool set small. Group related actions under one tool with an `action` enum rather than exposing many near-identical tools.
- Add an **intent router** as a pre-filter once the registry exceeds ~15 tools: a cheap first pass that narrows to a relevant subset before the tool schemas are sent to the LLM.
- Every tool declares a JSON schema, a one-line natural description, and a `requires_confirmation` flag.
- Destructive tools (delete, close-without-save, send message) must be gated behind spoken confirmation.

### 8.2 Windows control APIs

| Need                             | Library                     |
| -------------------------------- | --------------------------- |
| Global hotkey injection          | `pynput`                    |
| Window focus / enumerate / click | `pywinauto`, `uiautomation` |
| Launch / list / kill processes   | `psutil`, `subprocess`      |
| System + per-app volume          | `pycaw`                     |
| Clipboard                        | `pyperclip`                 |
| Web search                       | `ddgs` (free, no key)       |

### 8.3 Initial tool set (v1)

1. `system_volume` — get/set/mute system or per-app volume
2. `app_control` — launch, focus, close a named application
3. `send_hotkey` — inject a keyboard shortcut to the focused window
4. `discord_control` — mute/deafen/join/leave via hotkeys (see 8.4)
5. `web_search` — DuckDuckGo query, return top results
6. `clipboard` — read/write clipboard
7. `timer` — set/cancel a named timer
8. `window_info` — what is currently focused, what windows are open
9. `screenshot_describe` — capture and describe (stub for now, needs a VLM later)
10. `conversation_control` — stop talking, repeat that, never mind

### 8.4 Discord: important constraint

**There is no supported API for controlling your own Discord client.** Self-botting violates Discord's Terms of Service and risks a permanent account ban. Do not attempt it, and do not use any library that does.

The correct approach is **keyboard shortcut injection** against the focused Discord window:

- Toggle mute: `Ctrl+Shift+M`
- Toggle deafen: `Ctrl+Shift+D`
- Disconnect from voice: `Ctrl+Shift+D` is deafen — disconnect has no default binding, so this requires either a user-assigned Discord keybind or UI automation via `uiautomation`.

Document that the user must set custom Discord keybinds for any action without a default shortcut. This same pattern (focus window → inject hotkey) is the general answer for most desktop apps, not just Discord.

---

## 9. Build order

Do not build this top to bottom. Build it in vertical slices, each independently testable.

| Milestone | Deliverable                                                 | Done when                                                                    |
| --------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **M0**    | venv + `setup.ps1` + both servers launching, health-checked | `run.ps1` starts everything and reports Vulkan GPU detected for both servers |
| **M1**    | Mic → Silero VAD → whisper.cpp → printed transcript         | Speaking into the mic prints accurate text with < 400ms lag                  |
| **M2**    | Text → llama-server → streaming text out                    | Streaming tokens appear in terminal, TTFT measured and logged                |
| **M3**    | Text → Kokoro → audio out, sentence-chunked                 | Assistant speaks; audio queue is verifiably flushable mid-sentence           |
| **M4**    | Full loop, **no** barge-in yet                              | End-to-end conversation works, round-trip latency logged                     |
| **M5**    | **Barge-in + turn detection**                               | All acceptance tests in §10 pass. Budget the most time here.                 |
| **M6**    | Wake word + follow-up window                                | "Hey Jarvis" activates; follow-ups need no wake word                         |
| **M7**    | Tool registry + first 3 tools                               | Volume, app launch, web search work by voice                                 |
| **M8**    | Remaining tools + confirmation gating                       | Full v1 tool set                                                             |

---

## 10. Acceptance tests

Barge-in is either right or it isn't, and it is not testable by vibes. Write these as scripted tests first (same phrase, same timestamp, same expected cancel behaviour — this makes regressions obvious), then validate with unscripted human testing.

| Test                                                             | Expected                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------ |
| User speaks a long sentence with a 600ms mid-sentence pause      | Assistant does **not** respond during the pause              |
| User says "uh, hmm, so..." then continues                        | Assistant waits                                              |
| User says "mhm" while assistant is speaking                      | Assistant **continues** speaking (backchannel, not barge-in) |
| User says "actually, stop" while assistant is speaking           | Audio stops within 60ms, assistant listens                   |
| User interrupts, assistant stops, user gives new command         | New command is heard and executed                            |
| User interrupts mid-tool-call                                    | Tool completes in background, result available next turn     |
| Assistant speaks through speakers (no headphones)                | Assistant does not interrupt itself                          |
| Assistant is interrupted, then user asks "what were you saying?" | Assistant references only what it actually spoke aloud       |

---

## 11. Known risks

| Risk                                         | Mitigation                                                                                                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pipecat wheels fail on Windows / Python 3.11 | Try 3.11 first; if a hard blocker appears, fall back to a hand-rolled asyncio loop implementing §7 directly. The §7 spec is framework-independent by design. |
| llama.cpp Vulkan doesn't detect the 6700 XT  | Update AMD drivers; verify Vulkan runtime; fall back to LM Studio's Vulkan backend (same OpenAI-compatible interface)                                        |
| Qwen3-8B tool calling is unreliable          | Reduce tool count, tighten descriptions, add few-shot examples, enforce structured output via GBNF grammar                                                   |
| Latency exceeds 2s and feels sluggish        | Drop to Qwen3-4B; TTFT matters more than model quality for perceived responsiveness                                                                          |
| Kokoro CPU synthesis blocks the event loop   | Run synthesis in a thread pool; never synthesize on the main async loop                                                                                      |
| Self-interruption on speakers                | Ship headphone mode as the default; speaker mode is explicitly degraded                                                                                      |

---

## 12. Implementation notes

- Verify every library version and API against current docs before writing against it. This stack moves fast and several of these packages have renamed extras or changed service class signatures across releases. Do not write code from memory.
- Log aggressively during M5. Timestamped events for VAD state changes, turn decisions, flush calls, and cancel calls are the only way to debug turn-taking.
- Keep `config.yaml` authoritative for every threshold. Tuning will be empirical and iterative — hardcoded constants will waste hours.
