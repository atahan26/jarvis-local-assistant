# JARVIS

Local, free, English-only voice assistant. Cascaded streaming pipeline:
mic → VAD → turn detection → whisper.cpp → llama-server → Kokoro TTS → speakers.

Built against [JARVIS_SPEC.md](JARVIS_SPEC.md). Milestone order is spec §9.

**Status: M8 complete. All milestones in §9 are done.** Say "hey jarvis", then
talk normally. Eleven tools, barge-in that stops in 22 ms, backchannels that do
not interrupt it, a wake word, and a follow-up window.

```powershell
.\run.ps1                                                  # start the servers
.\.venv\Scripts\python.exe -m src.main --talk              # M4: talk to it
.\.venv\Scripts\python.exe -m src.main --listen            # M1: speak, see text
.\.venv\Scripts\python.exe -m src.main --chat              # M2: type, see the reply stream
.\.venv\Scripts\python.exe -m src.main --speak             # M3: type, hear it
.\.venv\Scripts\python.exe -m src.main --speak --flush-test # M3: measure interrupt latency
```

`--listen` prints a per-turn latency breakdown and a median/p95 summary on
Ctrl+C. `--meter` adds a live VAD confidence bar, `--devices` lists microphones,
`--device 51` picks one. In `--chat`, Ctrl+C mid-answer exercises the
cancellation path that barge-in depends on. In `--speak`, `/stop` interrupts
playback and `/voices` lists all 54 Kokoro voices.

### Measured against the spec's budgets

| Stage | Budget (§3) | Measured | |
| --- | --- | --- | --- |
| STT, endpoint to text | ~300 ms | ~640 ms | over |
| LLM time-to-first-token | ~250 ms | **125-141 ms** | met |
| TTS first audio chunk | ~150 ms | ~750 ms | **unreachable, see below** |
| **Full round trip** | 900-1500 ms | **~1450-1750 ms** | slightly over |
| TTS flush on barge-in | < 60 ms | **22 ms** to silence | met |
| LLM cancel on barge-in | < 40 ms | **< 40 ms** (asserted in tests) | met |
| Turn-taking gap P95 | 200-450 ms | ~2600 ms | **contradicts §7.2, see below** |

§7.6's 200-450 ms turn-taking gap target is not reachable, and not because of
this implementation: §7.2 mandates a 700 ms silence wait before the turn is even
considered over, SmartTurn adds 125-200 ms to confirm it, then STT, the LLM and
Kokoro's ~300 ms floor follow. The 700 ms alone exceeds the ceiling. The two
sections of the spec are asking for incompatible things; §7.2 is the one that
protects the user from being cut off, so it wins.

Round trip is measured endpoint decision to first spoken word, over three turns
driven through the real pipeline. The two costs that dominate are STT (~640 ms)
and Kokoro's first chunk (~750 ms); the LLM leg is nearly free. Switching
`whisper.model` to the already-downloaded `ggml-base.en-q5_1.bin` would take
roughly 290 ms off and land inside the target.

The first LLM request after an idle period costs ~830 ms; every one after that is
~125 ms, at 23-39 tok/s.

**§3's 150 ms first-audio budget cannot be met with Kokoro on CPU.** Synthesis
has a fixed floor of roughly 300 ms regardless of how little text it is given
("Sure." = 5 chars takes 312 ms) and grows from there: 28 chars 593 ms, 51 chars
797 ms, 196 chars 2906 ms. The first chunk is capped at 60 chars
(`tts.first_chunk_chars`), landing around 950 ms standalone. In the real loop TTS
overlaps LLM generation, so the perceived figure at M4 will be lower than this.

Note the perceived STT delay is dominated by the deliberate 700 ms endpoint wait
(§7.2, so the user is not cut off), not by whisper. Halving STT would move
"stopped speaking to text" from ~1300 ms to only ~1000 ms.

## Supported configuration

**Wear headphones.** This is the supported setup, per spec §7.1. There is no
acoustic echo cancellation. On speakers the assistant hears its own TTS through
the microphone and interrupts itself in a loop.

A degraded speaker mode is planned for M5 (`barge_in.echo_mode: speakers` in
`config.yaml`): it requires simultaneous high VAD probability, sustained speech,
and above-threshold frame energy before accepting a barge-in. That is a
heuristic, not real AEC, and it trades barge-in responsiveness for immunity to
self-triggering.

## Layout

```
assistant/
├── .venv/                  Python 3.11. Everything installs here, never system Python.
├── config.yaml             Authoritative for every threshold (spec §12).
├── config.local.yaml       Optional, gitignored, deep-merged over config.yaml.
├── setup.ps1               Fetches binaries + models. Idempotent.
├── run.ps1                 Starts both servers, health-checks them.
├── logs/                   Server logs, PID files, metrics.
├── src/
│   ├── venv_guard.py       Refuses to run outside .venv.
│   ├── config.py           YAML loading + path resolution.
│   ├── health.py           Service health checks.
│   ├── conversation.py     M4 orchestrator: the full loop and its state machine.
│   ├── listen.py           M1 loop: mic to transcript, with timings.
│   ├── chat.py             M2 loop: typed text to streaming reply.
│   ├── speak.py            M3 loop: text to speech, plus the flush test.
│   ├── audio/io.py         Always-open sounddevice capture.
│   ├── audio/vad.py        Silero wrapper + endpointing state machine (§7.2).
│   ├── audio/output.py     Flushable playback queue, tracks what was spoken.
│   ├── services/whisper_stt.py   whisper.cpp HTTP client.
│   ├── services/llm.py     Streaming, cancellable llama-server client.
│   ├── services/kokoro_tts.py    Kokoro + streaming sentence chunker.
│   └── main.py             Entry point.
└── tests/                  75 tests; the live ones skip if servers are down.
```

Binaries and models live **outside** the project at `C:\jarvis-data` because the
project sits in a OneDrive folder and the models are ~5.5 GB. Change
`paths.data_dir` in `config.yaml` to move them.

```
C:\jarvis-data\
├── bin\llama\              llama.cpp Vulkan build
├── bin\whisper\            whisper.cpp OpenBLAS build
├── models\                 whisper + Qwen3 GGUF
├── models\kokoro\          Kokoro ONNX + voices
└── downloads\              Release zips, kept for re-extraction
```

## Setup

```powershell
.\setup.ps1          # venv, pip install, download binaries + ~5.5 GB of models
.\run.ps1            # start both servers, health-check, leave them running
.\run.ps1 -Stop      # shut them down
.\run.ps1 -CheckOnly # health-check whatever is already running
.\run.ps1 -StaticOnly# files + GPU only, no servers needed
```

`setup.ps1` resolves GitHub release asset names from the API at run time rather
than hardcoding them, so it keeps working as upstream tags move. Pin a specific
build with `-LlamaTag b10269` / `-WhisperTag v1.9.2`.

## Where this build deviates from the spec

Verified 2026-08-04. Each of these is a case where the spec describes something
that does not exist or has changed upstream.

| Spec | Reality | What was done |
| --- | --- | --- |
| §5 "download the Vulkan Windows release" of whisper.cpp | whisper.cpp has **never** published a Windows Vulkan asset. Checked every release v1.5.0 through v1.9.2. Windows assets are CPU, CPU+OpenBLAS, and cuBLAS only. | Using the OpenBLAS CPU build. **STT runs on CPU, not GPU.** Escalation if M1 misses the ~300 ms budget: build from source with `-DGGML_VULKAN=ON` (VS 2022 + LunarG Vulkan SDK). |
| §4 `pipecat-ai[silero,local]` | pipecat-ai 1.7.0 publishes no `silero` extra. Silero VAD is in the core package; `onnxruntime` is a core dependency pinned `~=1.24.3`. | `pipecat-ai[local,kokoro]==1.7.0`, and `onnxruntime` is deliberately absent from requirements.txt so pipecat's pin wins. |
| §5 launch line passes a bare `-fa` | Build b10269 rejects it: `error while handling argument "-fa": expected value for argument`. It now takes `on\|off\|auto`. | `llm.flash_attn: "on"` in config.yaml; run.ps1 also maps a YAML boolean to on/off. |
| §5 "check the startup log for Vulkan device detection" | At llama-server's default verbosity threshold (3) the offload summary is **not logged at all**: the log is 13 lines and mentions neither Vulkan nor layers. | run.ps1 passes `-lv 4` (`llm.log_verbosity`), which restores `offloaded 37/37 layers to GPU` and the buffer-size lines without the ~2600 lines of per-layer spam that `-lv 5` adds. |
| §0 "llama.cpp's ROCm/HIP backend is Linux-only" | llama.cpp now ships `llama-<build>-bin-win-hip-radeon-x64.zip`. | Still using Vulkan as specced. The HIP build is unlikely to include a `gfx1031` target. Noted only because the premise aged. |
| §0 target CPU "Ryzen 5 X3D, 6 cores" | Actual CPU is a **Ryzen 7 5800X3D, 8c/16t**. | `whisper.threads: 8` in config.yaml. More CPU headroom for CPU-side whisper and Kokoro than the spec assumed. |

Unchanged and still correct: no faster-whisper / CTranslate2 anywhere, no NeMo,
Vulkan for the LLM, Kokoro on CPU. Spec §3's "SmartTurn (local ONNX)" is also
accurate: `LocalSmartTurnAnalyzerV3` ships a bundled 8.7 MB
`smart-turn-v3.2-cpu.onnx` and imports with no torch installed, despite what the
`local-smart-turn` extra's dependency list suggests.

## M7/M8: tools

Eleven tools, the whole §8.3 v1 set plus `system_info`:

| tool | actions |
| --- | --- |
| `system_volume` | get, set, up, down, mute, unmute |
| `app_control` | launch, focus, close, status; apps **and websites** |
| `send_hotkey` | any allowlisted key combination |
| `discord_control` | toggle mute, toggle deafen |
| `web_search` | DuckDuckGo |
| `clipboard` | read, write, clear |
| `timer` | set, cancel, cancel_all, list |
| `window_info` | focused, list |
| `screenshot_describe` | stub, admits it cannot see |
| `conversation_control` | sleep, stop_talking, repeat, forget |
| `system_info` | time, date, location, battery, performance |

Grouped by an `action` enum rather than split into many tools, because §8.1 warns
an 8B model degrades past roughly 10-15 of them. `Registry.schemas()` refuses to
build beyond `tools.max_exposed` rather than degrading quietly.

**Websites are first-class, not "open a browser".** "Open YouTube" used to
resolve to Chrome and launch an empty browser, because the tool only understood
applications. `app_control` handles apps and sites together now:

| said | opens |
| --- | --- |
| "Spotify" | the desktop app |
| "Spotify on the web" | open.spotify.com |
| "YouTube", "Instagram" | the site |
| "Pinterest", "Duolingo" | guessed as `pinterest.com`, `duolingo.com` |
| "example.com" | that domain |
| "please", "the thing" | refused |

An installed app wins unless the user says "web", "online" or "in the browser".
An unknown name is treated as a brand and tried as a `.com`, because "open
Instagram" failing outright was worse than a reasonable guess. Filler words are
excluded from guessing so "open please" does not become `please.com`.

Every URL, guessed or spoken, goes through the same validation: https only, no
credentials, no ports, no IP literals, no `file://` or `javascript:`. A URL is
one of the easier ways to smuggle something past an allowlist, and §5 rules that
out.

**The tool layer validates URL shape, never subject matter.** Qwen3 refused to
open an ordinary adult site, phrased as an inability ("I cannot open that
website"), and did it inconsistently: 10/10 fine with no history, 5/10 refusals
once a couple of turns had accumulated. That is the model's own reflex, not
anything in this code. This is a single-user assistant on that user's own PC, and
opening a URL is what a browser address bar does. The system prompt now says so
explicitly, which took it back to 10/10. Confirmation gating for destructive
actions is untouched, and a test asserts no content blocklist creeps into the
tool layer.

**Location is answered honestly rather than confidently.** There is no GPS on a
desktop, so it is IP geolocation, which reports the *network's* exit point. On
the author's machine ip-api reported a different country than the Windows clock and locale
said, because a VPN was up. The tool checks for an active VPN or tunnel
adapter and for a timezone mismatch, and says so instead of naming a city as
fact.

**Timers announce themselves.** An elapsed timer speaks even from IDLE; a silent
timer is a broken timer.

**Memory outlives the process** (`memory` tool, `memory.json` in the data dir).
Not in the spec; added because an assistant that forgets you on every restart is
annoying in a way no latency number captures.

* Facts are stored, not transcripts. Whole conversations would bury the useful
  parts and eat the 16k context.
* Facts live in the **system prompt**, not the message history. History is
  trimmed as a conversation grows, so anything stored there would silently fall
  out of the window exactly when the conversation got long enough to need it.
* There is a cap (60 facts). The list is prepended to every request, so
  unbounded memory is unbounded cost per turn, not just disk.
* `forget_all` is gated behind spoken confirmation.

One thing that needed fixing after testing: asked to "remember my name is X and
I prefer the volume at thirty", the model stored only the volume, dropped the
name, and then said it had noted both. The tool now takes a `facts` array and
the description insists every mentioned fact goes in it.

**"Go to sleep" works** (`conversation_control`). Tools are not handed a
reference to the Assistant, which would let any tool reach into the loop.
A control tool returns `{"control": ...}` in its result data and only
`_apply_control` reads it, so the coupling stays one-way. Sleep is applied after
the acknowledgement finishes speaking, or the reply would cut off its own
command.

**Qwen3-8B tool calling is reliable here**, contrary to the §11 risk. Measured
8/8 on the current descriptions: it picks the right tool, fills the arguments,
and leaves general knowledge alone. The one early failure was searching the web
for "capital of France"; tightening the `web_search` description to "only for
current events, live data, or facts you genuinely do not know" fixed it.

Latency is the real cost, because a tool turn is two LLM round trips plus the
tool itself:

| turn | TTFT | round trip |
| --- | --- | --- |
| no tool ("capital of France") | 125 ms | 1.5 s |
| volume | 890 ms | 2.3 s |
| web search | 6.0 s | 7.9 s |

Web search is dominated by `ddgs` itself, measured at 2.2-4.5 s for three
results. Nothing in this codebase can make that faster.

**§5 is enforced at the registry, not by trusting the model.** Unregistered tool
names are refused rather than fuzzy-matched, arguments are validated against the
declared schema, hallucinated extra arguments are dropped rather than forwarded,
and `app_control` resolves names against a fixed allowlist so "launch" can never
become a shell passthrough. A handler that raises is turned into a failed
`ToolResult` instead of killing the turn.

**Destructive actions are gated behind a spoken yes** (§8.1). `close` is listed
in `confirm_actions`, so the assistant asks first and only runs the tool after a
clear confirmation. Anything that is not clearly a yes counts as a no, including
an unrelated reply.

`pycaw`'s API has moved: `AudioUtilities.GetSpeakers()` returns an `AudioDevice`
with no `.Activate()`, so the widely-copied
`devices.Activate(IAudioEndpointVolume._iid_, ...)` snippet raises
`AttributeError` on 20251023. The device exposes `.EndpointVolume` directly now.

**Discord is driven by keyboard shortcuts only** (§8.4). There is no supported
API for controlling your own client and self-botting risks a permanent ban, so
`discord_control` focuses the window and sends `Ctrl+Shift+M` / `Ctrl+Shift+D`.
No Discord library is imported, and a test asserts that stays true. Disconnecting
from voice has no default keybind, so the tool refuses rather than guessing a
shortcut that would do something else.

`send_hotkey` is restricted to an allowlist of modifiers, letters, digits and
function keys, capped at four keys. An unrestricted key injector is arbitrary
code execution with extra steps: it could type anything into anything, including
a terminal. It also requires confirmation.

## Known issue: the Vulkan device is occasionally lost

Twice during testing, llama-server died mid-request with:

```
decode() failed: vk::Queue::submit: ErrorDeviceLost
```

It is intermittent and it is **not** correlated with prompt size or tool count:
28 consecutive requests with the full 12-tool schema (~1800 tokens) all
succeeded. No TDR or display-driver-reset events appeared in the Windows event
log either, so it is not a classic driver timeout. Treat it as llama.cpp Vulkan
instability on this GPU.

The recovery is `.\run.ps1 -Restart`.

What was fixed on this side: **llama-server reports these failures inside an
HTTP 200 stream**, as `data: {"error":{...}}`. The client ignored that shape and
returned an empty generation, so a GPU crash looked like the assistant simply
having nothing to say. It now surfaces the error, says so out loud, and prints
the restart command.

## M6: wake word

`openWakeWord` ships a pretrained `hey_jarvis` model, which is why the spec chose
that phrase. Measured peak scores with the ONNX model, at 0.5-0.9 ms per 32 ms
frame (cheap enough to run on every frame, unlike SmartTurn):

| audio | peak score |
| --- | --- |
| "Hey Jarvis" | 0.994 |
| "Hey Jarvis, what's the weather?" | 0.998 |
| "Hey Jarvis" (different voice) | 0.971 |
| bare "Jarvis" | 0.480 |
| "Hey there" | 0.001 |
| unrelated speech, silence | 0.000 |

Two things the defaults get wrong for this project:

* **ONNX is mandatory, not a preference.** openWakeWord defaults to tflite, but
  neither `tflite_runtime` nor `tensorflow` installs on Windows + Python 3.11.
  Only `onnxruntime` is present, so the framework and the ONNX feature models
  are passed explicitly.
* **`download_models()` defaults into `.venv/Lib/site-packages`,** which a venv
  rebuild destroys. Models go to `models/openwakeword/` instead
  (`wake_word.models_dir`).

State machine, per §7.5:

```
IDLE ──"hey jarvis"──> LISTENING ──turn──> THINKING ──> SPEAKING
                          ^                                │
                          └────── follow-up window ────────┘
                          │
                          └── window expires (8s) ──> IDLE
```

Every route back to LISTENING reopens the window, so a reply, an interruption
and a discarded turn all leave the floor open for the same time. Set
`wake_word.enabled: false` to skip IDLE entirely and always listen, which is how
M1-M5 behaved.

**Buffer while asleep, or the first words are lost.** Detection latency is not
constant: it gets worse as the score drops. On clean synthesized speech scoring
0.99 the model fires ~40 ms after the phrase ends, but a real microphone scoring
0.82 fires much later, and everything spoken in between was being thrown away
because IDLE frames went to the detector and nowhere else. Reported from a real
session as "hey jarvis, what do you think about the Roman Empire" arriving as
just `"about the Roman Empire"`.

`wake_word.preroll_ms` (default 1500) keeps a rolling buffer while asleep and
replays it into the turn on wake. Measured on the same utterance: **2.69 s of
audio captured without it, 3.97 s with it.** The wake phrase is replayed too and
removed from the transcript afterwards (`strip_wake_phrase`), since trimming it
out of the audio would mean knowing exactly where it ended. A transcript that is
*only* the wake phrase is dropped rather than answered.

## M5: how barge-in actually works here

Spec §3 says SmartTurn is the component that "distinguishes backchannel ('mhm',
'yeah') from real barge-in". **It does not.** Measured directly against the
bundled `smart-turn-v3.2-cpu.onnx`:

| said during assistant speech | SmartTurn verdict |
| --- | --- |
| "yeah" | COMPLETE |
| "okay" | COMPLETE |
| "right" | COMPLETE |
| "actually stop" | COMPLETE |
| "mhm" | INCOMPLETE |
| "uh huh" | INCOMPLETE |

SmartTurn predicts whether an utterance is *grammatically finished*, not whether
the speaker meant to take the floor. Those are different questions, and it is
useless as a backchannel classifier. It is still used for what it is good at:
deciding whether to extend the endpoint wait past 700 ms (§7.2), where it scored
5/6 on a small fixture set, at 156-203 ms per call.

The backchannel discriminator is **duration** instead, which is free and
available immediately:

* Speech during playback that stops on its own before `barge_in.confirm_ms`
  (default 350 ms) **never touches the audio**. The assistant genuinely keeps
  talking, which is what §10 requires. Measured backchannels run 0.4-0.8 s.
* Speech still going at `confirm_ms` is a confirmed barge-in. Only then does the
  60 ms flush clock start, so §7.3 is measured from confirmation, not from the
  first syllable.
* A transcript veto runs afterwards as a second line of defence. It cannot delay
  the flush because the audio has already stopped.

Verified end to end, three consecutive runs: confirmed at ~328 ms of speech,
**silent 22 ms later**, 24 of 219 generated characters actually spoken, and
history recording only `"I can't provide detailed [interrupted by user]"`.

**Never run the turn probe per frame.** SmartTurn costs 125-200 ms and runs on
the capture thread. Probing on every 32 ms frame is roughly 4x real time: the
microphone queue backs up, frames are dropped, and the silence counter drifts
behind the wall clock because it counts frames rather than time. A real session
recorded `silence_wait=9281ms` against a 3000 ms ceiling, and fired several
spurious barge-ins while replaying the backlog. `turn.smart_turn_interval_ms`
(default 300) now probes once at base silence and then at most every 300 ms:
measured 30+ calls per turn down to **1**, and the wait back to 719 ms.

**Do not mark interruptions in-band.** §7.3 says an interrupted reply should
enter history "marked as interrupted", and the obvious reading is to append
something like `[interrupted by user]` to the assistant message. Do not. The
model reads its own history, learns the pattern, and starts emitting the marker
in later replies, at which point Kokoro reads it aloud. Reported from a real
session where the assistant spoke the words "interrupted by user" on a turn the
user had not interrupted at all.

History now stores only the spoken text; the truncation is the signal, and the
system prompt explains it. Independently, the TTS path strips bracketed asides
(`strip_asides`), because a local model produces stage directions on its own
regardless of what the prompt says.

**The stall that nearly ruined it.** Kokoro's `create()` cannot be cancelled and
is lock-serialized. The LLM normally finishes well before playback does, so the
turn is usually sitting in the "wait for synthesis to drain" join when an
interrupt lands, and a plain `join()` added the in-flight chunk's synthesis
(measured 1000-1234 ms) directly onto the reply to the interruption. That wait is
now polled and abandoned on cancel: **1234 ms -> 0 ms**. `tts.max_chunk_chars`
was also lowered to 100, which bounds the worst-case in-flight chunk at ~1.7 s
instead of ~2.4 s.

## M3: the output device is a barge-in problem, not an audio-quality problem

Clearing the playback queue is nearly free (0.00 ms). What actually determines
when the user stops hearing the assistant is the **sound device's own buffer**,
because that audio is already committed to hardware. Measured on this machine:

| Host API | Buffer | Verdict for §7.3's 60 ms budget |
| --- | --- | --- |
| MME (PortAudio's Windows default) | ~192 ms | fails outright |
| WASAPI shared | **22 ms** | comfortable |
| WASAPI exclusive | ~32 ms | works, but locks the device from other apps |

So `audio.output_device: null` does **not** hand the choice to PortAudio, which
would pick MME. It resolves the system default output on the fastest available
host API. WASAPI shared mode only accepts the device's native rate, so Kokoro's
24 kHz is resampled to 48 kHz on the synthesis thread; soxr costs ~0.3 ms per
second of audio.

This was worth catching early: the naive measurement reports a 0.00 ms flush and
looks like a comfortable pass, while the user would still hear ~200 ms of speech
after interrupting.

**Known limitation for M5:** Kokoro's `create()` cannot be interrupted and
synthesis is serialized behind a lock, so a chunk already being synthesized when
a barge-in lands runs to completion. It is never played, but it does delay the
next response by up to one chunk's synthesis time. Keeping chunks small bounds
that; it is the reason `max_chunk_chars` exists as a tunable.

## M1 gotchas worth remembering

Two pipecat/Silero traps, both of which fail **silently** rather than loudly:

1. **`SileroVADAnalyzer(sample_rate=16000)` does not set the sample rate.** The
   constructor argument only populates `_init_sample_rate`; the real
   `_sample_rate` stays `0` until `set_sample_rate()` is called, which normally
   only happens inside a pipecat transport. Without it the ONNX model rejects
   every frame, and `voice_confidence()` catches the exception and returns
   `0.0`, so the VAD simply never fires and the assistant appears deaf.
   `src/audio/vad.py` calls `set_sample_rate()` explicitly and then asserts
   `num_frames_required() == 512`.
2. **`voice_confidence()` returns a shape-`(1,)` array, not a float.** numpy 2.x
   raises `TypeError: only 0-dimensional arrays can be converted to Python
   scalars` on `float()`. Unwrap with `np.asarray(v).reshape(-1)[0]`.

Also: **Silero v5 requires exactly 512 samples per frame at 16 kHz (32 ms)**, so
spec §2's 20 ms frames are not achievable. 512-sample frames are used throughout.

## Known issue: nltk's CWD import guard vs. an in-project .venv

nltk 3.9+ installs a meta-path finder (`nltk/inisec.py`) that blocks any import
**nltk initiates** when the target module resolves to a path inside the current
working directory. Spec §1 requires `.venv` to live in the project root, so with
`cwd == project root` every package in `site-packages` is "inside the CWD" and the
guard fires on nltk's own `import regex`:

```
ImportError: Blocked import of regex from current working directory for security reasons.
```

pipecat reaches nltk through `pipecat.services.tts_service` →
`pipecat.utils.string`, so this will hit any pipecat TTS or LLM service import.
It is a false positive: the blocked module is a legitimate venv package, not an
attacker-planted file.

M0 does not import pipecat and is unaffected. The fix, to be applied when M3
lands: `chdir` out of the project root before importing pipecat and put the
project root on `sys.path` explicitly. `NLTK_DISABLE_IMPORT_SECURITY=1` is the
documented escape hatch but disables a real mitigation, so it is the fallback,
not the default.

## Verified environment

| Thing | Value |
| --- | --- |
| Python | 3.11.7 in `.venv` |
| GPU | AMD Radeon RX 6700 XT, 12272 MiB |
| Vulkan | 1.4.315, AMD proprietary driver |
| llama.cpp | b10269, `llama-b10269-bin-win-vulkan-x64.zip` |
| whisper.cpp | v1.9.2, `whisper-blas-bin-x64.zip` |
| LLM | Qwen3-8B-Q4_K_M, 16k ctx, KV cache q8_0 |
| STT | `ggml-small.en-q5_1.bin` |
| TTS | Kokoro-82M v1.0 |

### Measured VRAM (M0)

`load_tensors: offloaded 37/37 layers to GPU`, i.e. 4.35 GiB weights + 1.20 GiB KV
cache on Vulkan0, confirmed independently by the `GPU Process Memory` perf
counter at 5.37 GB dedicated for the llama-server process. Spec §3 budgeted
~6.5 GB, so there is more headroom than planned.

Note: `llama-server --list-devices` reports *per-process* free VRAM, so it always
shows ~11.4 GiB free regardless of what is loaded. It is not usable as a
"did the model land on the GPU" check. The log line and the perf counter are.
