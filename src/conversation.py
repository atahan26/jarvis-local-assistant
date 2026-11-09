"""M5: the full loop WITH barge-in. This is the actual project (spec §7).

The microphone never stops being read, and turn detection stays fully active
while the assistant is speaking. That is the whole point: §1 calls barge-in the
primary success criterion.

Three things had to be true at once, and they pull against each other:

  §7.2  do not cut the user off      -> 700 ms base silence, extended to 3000 ms
                                        when SmartTurn says the turn is unfinished
  §7.2  ignore backchannels          -> speech that stops before confirm_ms never
                                        interrupts playback at all
  §7.3  stop fast when interrupted   -> flush TTS < 60 ms, cancel LLM < 40 ms

Ordering matters in the interrupt path. The audio is flushed FIRST, because that
is what the user perceives, and only then is the LLM cancelled and history
written. History records only what was actually played (§7.3 item 3); text that
was generated but never reached the speaker is dropped, because the user never
heard it.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .audio.barge_in import BargeInPolicy, BargeInState, is_backchannel
from .audio.io import AudioFormat, MicrophoneStream, describe_device, resolve_device
from .audio.output import SpeakerOutput, describe_output_device, resolve_output_device
from .audio.turn_model import SmartTurnModel
from .audio.vad import Endpointer, TurnEvent, TurnState, Utterance, VoiceActivityDetector
from .audio.wake_word import WakeWordDetector, is_wake_phrase_only, strip_wake_phrase
from .config import Config
from .metrics import MetricsLog, TurnRecord
from .services.kokoro_tts import KokoroTTS, StreamingChunker
from .services.llm import Conversation, Generation, LLMClient
from .services.whisper_stt import WhisperClient

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def _tty() -> bool:
    return sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return text if not _tty() else f"{color}{text}{RESET}"


class State(Enum):
    IDLE = "idle"            # waiting for the wake word (§7.5)
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


NO_TOOLS_NOTE = (
    " You cannot yet control the computer, set timers, play music, or search the "
    "web. No tools are connected. If asked to do something like that, say plainly "
    "that you cannot do it yet, in one short sentence. Never claim you have done it."
)

TOOLS_NOTE = (
    " You have tools for controlling this computer. Use them when the user asks "
    "you to do something rather than describing what they could do. After a tool "
    "runs, tell the user the outcome in one short sentence. If a tool fails, say "
    "what went wrong plainly. Never claim to have done something a tool did not do."
    # Chaining does not occur to the model on its own: asked for local weather it
    # searched the web for the literal words "weather where I live" rather than
    # finding out where that is first.
    " Tools can be used one after another. When the user asks about their own "
    "area, such as the local weather or what is nearby, first call system_info "
    "with action location to find out where they are, then search using that "
    "place name. Never search the web for phrases like 'where I live'."
    " You do have access to the user's approximate location through system_info; "
    "never say that you cannot find their location without calling it first."
    # Measured: with a couple of turns of history, the model refused to open a
    # perfectly ordinary adult website 5 times out of 10, and phrased it as an
    # inability ("I cannot open that website") rather than a choice. This is a
    # single-user assistant on that user's own PC. Opening a URL is what a
    # browser address bar does; it is not the assistant's business to curate it.
    " This is the owner's personal computer and they are an adult. Open whatever "
    "website or application they ask for, including adult sites, without "
    "commenting on the choice, moralising, or suggesting alternatives. Do not "
    "claim you are unable to open something you can open. The only things you "
    "ever refuse are actions that would damage the machine or lose their work, "
    "and those you ask about rather than refuse."
)

# Spoken confirmation for destructive actions (§8.1). Kept deliberately narrow:
# anything not clearly a yes is treated as a no.
_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "go ahead",
        "confirm", "confirmed", "please do", "affirmative", "correct"}
_NO = {"no", "nope", "cancel", "stop", "don't", "dont", "never mind", "nevermind",
       "forget it", "abort", "negative"}


def classify_confirmation(text: str) -> bool | None:
    """True for yes, False for no, None when it is neither."""
    import re

    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    if not normalized:
        return None
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    words = normalized.split()
    if words and words[0] in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay"}:
        return True
    if words and words[0] in {"no", "nope", "cancel", "stop", "don", "never"}:
        return False
    return None


class Assistant:
    def __init__(
        self,
        cfg: Config,
        *,
        device_override=None,
        out_device_override=None,
        verbose: bool = False,
    ):
        self.cfg = cfg
        self.verbose = verbose
        audio_cfg = cfg.section("audio")
        turn_cfg = cfg.section("turn")
        barge_cfg = cfg.section("barge_in")

        self.fmt = AudioFormat(
            sample_rate=int(audio_cfg.get("input_sample_rate", 16000)),
            channels=int(audio_cfg.get("channels", 1)),
        )
        self.in_device = resolve_device(
            device_override if device_override is not None else audio_cfg.get("input_device")
        )
        self.out_device = resolve_output_device(
            out_device_override
            if out_device_override is not None
            else audio_cfg.get("output_device")
        )

        self.stt = WhisperClient(
            cfg.whisper_url, language=str(cfg.get("whisper.language", "en"))
        )
        self.llm = LLMClient(
            cfg.llm_url,
            temperature=float(cfg.get("llm.temperature", 0.7)),
            top_p=float(cfg.get("llm.top_p", 0.8)),
            max_tokens=int(cfg.get("llm.max_tokens", 512)),
            enable_thinking=bool(cfg.get("llm.enable_thinking", False)),
        )
        self.tts = KokoroTTS(
            cfg.tts_model,
            cfg.tts_voices,
            voice=str(cfg.get("tts.voice", "af_heart")),
            speed=float(cfg.get("tts.speed", 1.0)),
            max_chunk_chars=int(cfg.get("tts.max_chunk_chars", 180)),
            first_chunk_chars=int(cfg.get("tts.first_chunk_chars", 60)),
        )
        self.speaker = SpeakerOutput(
            sample_rate=self.tts.sample_rate,
            device=self.out_device,
            blocksize=int(audio_cfg.get("output_blocksize", 256)),
            device_sample_rate=audio_cfg.get("output_sample_rate", None),
        )

        self.vad = VoiceActivityDetector(
            sample_rate=self.fmt.sample_rate,
            confidence=float(turn_cfg.get("vad_threshold", 0.5)),
        )
        self.smart_turn: SmartTurnModel | None = None
        if bool(turn_cfg.get("smart_turn_enabled", True)):
            self.smart_turn = SmartTurnModel(sample_rate=self.fmt.sample_rate)

        self.endpointer = Endpointer(
            self.vad,
            vad_threshold=float(turn_cfg.get("vad_threshold", 0.5)),
            base_silence_ms=int(turn_cfg.get("base_silence_ms", 700)),
            max_wait_ms=int(turn_cfg.get("max_wait_ms", 3000)),
            min_speech_ms=int(turn_cfg.get("min_speech_ms", 200)),
            frame_ms=self.fmt.frame_ms,
            probe_interval_ms=int(turn_cfg.get("smart_turn_interval_ms", 300)),
            on_event=self._on_turn_event,
        )
        if self.smart_turn is not None:
            self.endpointer.turn_complete_probe = self._turn_complete_probe

        self.barge = BargeInPolicy(
            confirm_ms=int(barge_cfg.get("confirm_ms", 350)),
            vad_threshold=float(turn_cfg.get("vad_threshold", 0.5)),
            frame_ms=self.fmt.frame_ms,
            echo_mode=str(barge_cfg.get("echo_mode", "headphones")),
            speaker_vad_threshold=float(cfg.get("barge_in.speakers.vad_threshold", 0.6)),
            speaker_sustained_ms=int(cfg.get("barge_in.speakers.sustained_speech_ms", 250)),
            speaker_energy_dbfs=float(cfg.get("barge_in.speakers.energy_threshold_dbfs", -35.0)),
        )
        self.barge_enabled = bool(barge_cfg.get("enabled", True))
        self.transcript_veto = bool(barge_cfg.get("transcript_veto", True))
        self.flush_budget_ms = float(barge_cfg.get("tts_flush_budget_ms", 60))
        self.cancel_budget_ms = float(barge_cfg.get("llm_cancel_budget_ms", 40))
        self.backchannel_phrases = list(turn_cfg.get("backchannel_phrases", []))

        tools_cfg = cfg.section("tools")
        memory_cfg = cfg.section("memory")
        self.tools_enabled = bool(tools_cfg.get("enabled", False))

        self.memory = None
        if bool(memory_cfg.get("enabled", True)):
            from .tools.memory import MemoryStore

            self.memory = MemoryStore(
                cfg.data_dir / str(memory_cfg.get("file", "memory.json"))
            )

        self.registry = None
        if self.tools_enabled:
            from .tools import build_registry

            self.registry = build_registry(
                max_exposed=int(tools_cfg.get("max_exposed", 15)),
                memory_store=self.memory,
            )
        self.require_confirmation = bool(
            tools_cfg.get("require_confirmation_default", True)
        )
        if self.tools_enabled:
            # A timer that elapses has to interrupt whatever is happening and say
            # so; there is no other way for the user to find out.
            from .tools.media import TIMERS

            TIMERS.on_elapsed = self._on_timer_elapsed
        # A tool call waiting on a spoken yes/no (§8.1).
        self._pending_confirmation: tuple[Any, dict] | None = None
        # Set by conversation_control; acted on once the reply has been spoken.
        self._sleep_requested = False
        self._repeat_requested = False
        self._last_spoken_reply = ""

        self.convo = Conversation()
        self.convo.system_prompt += TOOLS_NOTE if self.tools_enabled else NO_TOOLS_NOTE
        self._base_system_prompt = self.convo.system_prompt
        # The refreshes happen at the end of __init__: the vocabulary hint reads
        # self.wake, which is not set up yet at this point.

        self.metrics = MetricsLog(
            cfg.log_dir / str(cfg.get("logging.metrics_file", "metrics.jsonl")),
            enabled=bool(cfg.get("logging.turn_events", True)),
        )

        # §7.5: the wake word gates IDLE only. With it disabled we start
        # listening immediately, which is how M1-M5 behaved.
        wake_cfg = cfg.section("wake_word")
        self.wake_enabled = bool(wake_cfg.get("enabled", True))
        self.wake: WakeWordDetector | None = None
        if self.wake_enabled:
            self.wake = WakeWordDetector(
                cfg.models_dir / str(wake_cfg.get("models_dir", "openwakeword")),
                name=str(wake_cfg.get("model", "hey_jarvis")),
                threshold=float(wake_cfg.get("threshold", 0.5)),
                refractory_ms=int(wake_cfg.get("refractory_ms", 2000)),
            )
        self.follow_up_window_ms = int(turn_cfg.get("follow_up_window_ms", 8000))
        self._follow_up_until = 0.0
        # Rolling buffer of audio captured while asleep, replayed on wake.
        self._wake_preroll: list[bytes] = []
        self._wake_preroll_max = max(
            1, int(int(wake_cfg.get("preroll_ms", 1500)) / self.fmt.frame_ms)
        )

        self.state = State.IDLE if self.wake_enabled else State.LISTENING
        self._state_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._work: queue.Queue[Utterance] = queue.Queue()
        self._stop = threading.Event()
        self._cancel = threading.Event()

        self._smart_turn_ms = 0.0
        self._turn_index = 0
        # Frames captured while the assistant was talking, so a confirmed
        # barge-in does not lose the beginning of what the user said.
        self._preroll: list[bytes] = []
        self._preroll_max = max(1, int(2000 / self.fmt.frame_ms))

        # Everything they depend on now exists.
        self._refresh_memory_prompt()
        self._refresh_stt_vocabulary()

    # -- plumbing ----------------------------------------------------------

    def _say(self, text: str) -> None:
        with self._print_lock:
            print(text)

    def _set_state(self, state: State) -> None:
        with self._state_lock:
            if self.state is state:
                return
            previous = self.state
            self.state = state
        if state is State.LISTENING:
            # Every route back to LISTENING reopens the follow-up window, so a
            # reply, an interruption and a discarded turn all leave the floor
            # open for the same amount of time.
            self._arm_follow_up()
        self.metrics.event("state", **{"from": previous.value, "to": state.value})
        if self.verbose:
            self._say(c(f"  [{previous.value} -> {state.value}]", MAGENTA))

    def _run_tools(self, gen: Generation) -> bool:
        """Execute the tools the model asked for. Returns False if cancelled.

        §7.4: a tool already running when the user interrupts is allowed to
        finish rather than being killed, and its result stays in conversation
        state. Nothing here aborts on _cancel; the check is only about whether
        to keep going to the next tool.
        """
        if self.registry is None:
            return True

        self.convo.add_tool_calls(gen.tool_calls)
        for call in gen.tool_calls:
            tool = self.registry.get(call.name)
            arguments: dict[str, Any] = {}
            if tool is not None:
                try:
                    arguments = self.registry.validate(
                        tool, self.registry.parse_arguments(tool, call.arguments)
                    )
                except Exception:
                    arguments = {}

            # §8.1: destructive actions need a spoken yes before they run.
            if (
                tool is not None
                and self.require_confirmation
                and tool.needs_confirmation(arguments)
            ):
                self._pending_confirmation = (call, arguments)
                message = self._confirmation_question(call, arguments)
                self.convo.add_tool_result(call, f"Waiting for the user to confirm: {message}")
                self.metrics.event(
                    "tool_confirm_requested", tool=call.name, arguments=arguments
                )
                self._say(c(f"  ? {message}", YELLOW))
                self._speak_now(message)
                return False

            # The reply is being streamed inline; start tool logging on its own
            # line so the two do not run together.
            print()
            self._say(c(f"  > {call.name}({call.arguments or '{}'})", MAGENTA))
            result = self.registry.call(call.name, call.arguments)
            self.metrics.event(
                "tool_call",
                tool=call.name,
                arguments=call.arguments,
                ok=result.ok,
                duration_ms=round(result.duration_ms, 1),
                summary=result.summary,
            )
            self._say(
                c(
                    f"  < {'ok' if result.ok else 'failed'} "
                    f"({result.duration_ms:.0f}ms) {result.summary[:70]}",
                    DIM,
                )
            )
            self.convo.add_tool_result(call, result.for_model())
            self._apply_control(result)
            if call.name == "memory":
                # A fact stored mid-turn should be in scope from the next
                # request onward, not only after a restart. A newly remembered
                # name also has to reach whisper, or it keeps being misheard.
                self._refresh_memory_prompt()
                self._refresh_stt_vocabulary()
        return True

    def _refresh_stt_vocabulary(self) -> None:
        """Feed proper nouns to whisper so it stops mangling them.

        Whisper reliably gets names wrong ("Atahan" came back as "Adehan", and
        "Atahan here" as "add a hand here"). An initial prompt containing the
        names fixes it. Sources: the config list, the app allowlist, the wake
        phrase, and any capitalised words already in memory, so a name the user
        asked to be remembered starts being heard correctly too.
        """
        words: list[str] = list(self.cfg.get("whisper.vocabulary", []) or [])

        if self.wake is not None:
            words.extend(w.capitalize() for w in self.wake.name.split("_"))

        if self.tools_enabled:
            from .tools.apps import DEFAULT_APPS

            words.extend(app.name for app in DEFAULT_APPS)

        if self.memory is not None:
            import re

            for fact in self.memory.all():
                # Proper nouns are the part whisper gets wrong; ordinary words
                # in a fact are not worth prompt budget.
                words.extend(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", fact.text))

        self.stt.set_vocabulary(words)

    def _refresh_memory_prompt(self) -> None:
        """Fold remembered facts into the system prompt.

        They go here rather than into the message history because history is
        trimmed as a conversation grows, so anything stored there would silently
        fall out of the window exactly when the conversation got long enough to
        need it.
        """
        if self.memory is None:
            return
        self.convo.system_prompt = self._base_system_prompt + self.memory.as_prompt()

    def _apply_control(self, result) -> None:
        """Act on a directive from conversation_control.

        Tools are not given a reference to the Assistant; a control tool returns
        `{"control": ...}` in its data and only this method reads it, so the
        coupling stays one-way.
        """
        directive = (result.data or {}).get("control")
        if not directive:
            return

        from .tools.conversation_control import FORGET, REPEAT, SLEEP, STOP_SPEAKING

        self.metrics.event("control", directive=directive)
        if directive == SLEEP:
            self._sleep_requested = True
        elif directive == STOP_SPEAKING:
            flush = self.speaker.flush()
            self.metrics.flush_latencies_ms.append(flush.audible_stop_ms)
            self._cancel.set()
        elif directive == FORGET:
            self.convo.reset()
        elif directive == REPEAT:
            self._repeat_requested = True

    def _speak_now(self, text: str) -> None:
        """Say a short line that did not come from the LLM stream.

        Used for confirmation questions, which must be spoken before the model
        has anything to say about the tool's outcome.
        """
        for chunk_text in self.tts.split(text):
            if self._cancel.is_set() or self._stop.is_set():
                return
            try:
                chunk = self.tts.synthesize(chunk_text)
            except Exception:  # noqa: BLE001
                return
            if chunk.samples.size and not self._cancel.is_set():
                self._set_state(State.SPEAKING)
                self.speaker.enqueue(chunk.text, chunk.samples)

    def _resolve_confirmation(self, text: str, record: TurnRecord) -> bool:
        """Handle a yes/no answer to a pending destructive tool call (§8.1).

        Returns True when this turn was consumed as an answer.
        """
        call, arguments = self._pending_confirmation
        answer = classify_confirmation(text)

        if answer is None:
            # Not a yes or a no. Drop the pending call rather than guessing:
            # §8.1 gates destructive actions, and silence is not consent.
            self._pending_confirmation = None
            self.metrics.event("tool_confirm_unclear", tool=call.name, text=text)
            self._say(c("  . not a yes or no, cancelling that action", DIM))
            self.convo.add_tool_result(call, "The user did not confirm, so it was not done.")
            self._speak_now("I won't do that then.")
            self._set_state(State.LISTENING)
            return True

        self._pending_confirmation = None
        if not answer:
            self.metrics.event("tool_confirm_denied", tool=call.name)
            self.convo.add_tool_result(call, "The user declined, so it was not done.")
            self._say(c(f"  . {call.name} cancelled by the user", DIM))
            self._speak_now("Okay, cancelled.")
            self._set_state(State.LISTENING)
            return True

        self.metrics.event("tool_confirm_granted", tool=call.name)
        self._say(c(f"  > {call.name}({call.arguments or '{}'}) [confirmed]", MAGENTA))
        result = self.registry.call(call.name, call.arguments)
        record.tools_used += 1
        self.metrics.event(
            "tool_call",
            tool=call.name,
            arguments=call.arguments,
            ok=result.ok,
            duration_ms=round(result.duration_ms, 1),
            summary=result.summary,
            confirmed=True,
        )
        self._say(c(f"  < {'ok' if result.ok else 'failed'} {result.summary[:70]}", DIM))
        self.convo.add_tool_result(call, result.for_model())
        self._speak_now(result.summary)
        self.speaker.wait_until_idle(timeout=30)
        self._set_state(State.LISTENING)
        return True

    def _confirmation_question(self, call, arguments: dict) -> str:
        target = arguments.get("name") or arguments.get("query") or ""
        action = arguments.get("action", call.name.replace("_", " "))
        if target:
            return f"Do you want me to {action} {target}?"
        return f"Do you want me to {action}?"

    def _on_timer_elapsed(self, name: str) -> None:
        """A timer finished. Announce it, from whatever thread fired it."""
        if self._stop.is_set():
            return
        self.metrics.event("timer_elapsed", name=name)
        message = f"Your {name} is up." if name != "timer" else "Your timer is up."
        self._say(c(f"  * {message}", YELLOW))
        # Announce even while asleep: the user asked for this before going idle,
        # and a silent timer is a broken timer.
        was_idle = self.state is State.IDLE
        self._speak_now(message)
        if was_idle:
            self.speaker.wait_until_idle(timeout=30)
            self._set_state(State.IDLE)

    def _go_to_sleep(self, reason: str) -> None:
        """Drop back to IDLE, requiring the wake word again (§7.5)."""
        if not self.wake_enabled:
            # Without a wake word there is no IDLE to return to, so honour the
            # request as far as it can be honoured and say so.
            self._say(c("  . wake word is off, so I can't sleep; still listening", YELLOW))
            self._set_state(State.LISTENING)
            return
        self._follow_up_until = 0.0
        if self.wake is not None:
            self.wake.reset()
        self._wake_preroll.clear()
        self.endpointer.reset()
        self.barge.reset()
        self.metrics.event("sleep", reason=reason)
        self._say(c(f"  . sleeping ({reason}), say \"{self.wake.name.replace('_', ' ')}\"", DIM))
        self._set_state(State.IDLE)

    def _arm_follow_up(self) -> None:
        """(Re)open the window during which no wake word is needed (§7.5)."""
        if self.wake_enabled:
            self._follow_up_until = time.monotonic() + self.follow_up_window_ms / 1000.0

    def _on_turn_event(self, ev: TurnEvent) -> None:
        self.metrics.event(
            "vad", event=ev.kind, detail=ev.detail, confidence=round(ev.confidence, 3)
        )
        if ev.kind == "speech_start" and self.state is State.LISTENING:
            self._say(c("  . hearing you", DIM))
        elif ev.kind == "turn_extended":
            self._say(c(f"  . waiting, turn sounds unfinished ({ev.detail})", DIM))

    def _turn_complete_probe(self, _audio: bytes) -> bool:
        """SmartTurn decides whether to fire now or keep waiting (§7.2)."""
        if self.smart_turn is None:
            return True
        verdict = self.smart_turn.is_complete()
        self._smart_turn_ms = verdict.latency_ms
        self.metrics.event(
            "smart_turn",
            complete=verdict.complete,
            latency_ms=round(verdict.latency_ms, 1),
            error=verdict.error,
        )
        return verdict.complete

    def close(self) -> None:
        self.stt.close()
        self.llm.close()
        self.speaker.close()
        if self.smart_turn is not None:
            self.smart_turn.close()
        self.metrics.close()

    # -- interruption ------------------------------------------------------

    def _interrupt_now(self, speech_ms: float, reason: str, onset_mono: float) -> None:
        """Confirmed barge-in. Audio first, everything else after (§7.3)."""
        interrupt_at = time.monotonic()

        # 1. Stop the sound. This is what the user is judging.
        flush = self.speaker.flush()
        self.metrics.flush_latencies_ms.append(flush.audible_stop_ms)
        onset_to_silence = (interrupt_at - onset_mono) * 1000.0 + flush.audible_stop_ms
        self.metrics.onset_to_silence_ms.append(onset_to_silence)

        # 2. Cancel generation. The worker notices and unwinds.
        self._cancel.set()

        self.metrics.barge_ins += 1

        over = " OVER BUDGET" if flush.audible_stop_ms > self.flush_budget_ms else ""
        self.metrics.event(
            "barge_in",
            reason=reason,
            speech_ms=round(speech_ms, 1),
            flush_ms=round(flush.audible_stop_ms, 2),
            spoken=flush.spoken_text,
            dropped=flush.dropped_text,
        )
        self._say(
            c(
                f"  <- interrupted, silent in {flush.audible_stop_ms:.0f}ms{over}",
                YELLOW if not over else RED,
            )
        )

    # -- the turn ----------------------------------------------------------

    def _handle(self, utt: Utterance) -> None:
        self._turn_index += 1
        record = TurnRecord(
            index=self._turn_index,
            silence_wait_ms=utt.silence_wait_ms,
            turn_extended=utt.extended,
            smart_turn_ms=self._smart_turn_ms,
        )
        self._smart_turn_ms = 0.0
        self._cancel.clear()

        self._set_state(State.THINKING)

        try:
            transcript = self.stt.transcribe_pcm(utt.audio, utt.sample_rate)
        except Exception as exc:  # noqa: BLE001
            self._say(c(f"  ! STT failed: {type(exc).__name__}: {exc}", RED))
            self._set_state(State.LISTENING)
            return

        record.stt_ms = transcript.latency_ms
        if transcript.is_empty:
            self._say(c("  . nothing recognised", DIM))
            self._set_state(State.LISTENING)
            return

        # The replayed wake audio puts "hey jarvis" at the front of the
        # transcript. It is not part of the request, and it has to come off
        # BEFORE the backchannel check: otherwise a bare "Hey Jarvis" with
        # nothing after it looks like a real request and produces an empty turn.
        text = transcript.text
        if self.wake is not None:
            if is_wake_phrase_only(text, self.wake.name):
                self._say(c("  . just the wake word, listening", DIM))
                self._set_state(State.LISTENING)
                return
            text = strip_wake_phrase(text, self.wake.name)

        # A turn that is nothing but "mhm" is an acknowledgement, not a request.
        if is_backchannel(text, self.backchannel_phrases):
            self.metrics.backchannels_ignored += 1
            self.metrics.event("backchannel_turn", text=text)
            self._say(c(f"  . ignoring backchannel: {text!r}", DIM))
            self._set_state(State.LISTENING)
            return

        record.transcript = text
        self._say(f"  {c('you', CYAN)}    {text}")

        # A destructive tool is waiting on a spoken yes/no (§8.1). This turn is
        # the answer, not a new request.
        if self._pending_confirmation is not None:
            if self._resolve_confirmation(text, record):
                return

        self.convo.add_user(text)

        chunker = StreamingChunker(
            max_chars=self.tts.max_chunk_chars,
            first_chunk_chars=self.tts.first_chunk_chars,
        )
        gen = Generation()
        llm_started = time.monotonic()
        first_audio_at: float | None = None

        synth_queue: queue.Queue[str | None] = queue.Queue()

        def synth_worker() -> None:
            nonlocal first_audio_at
            while True:
                chunk_text = synth_queue.get()
                if chunk_text is None:
                    return
                if self._cancel.is_set():
                    continue  # drain, do not synthesize
                try:
                    chunk = self.tts.synthesize(chunk_text)
                except Exception as exc:  # noqa: BLE001
                    self._say(c(f"  ! TTS failed: {type(exc).__name__}: {exc}", RED))
                    continue
                # Re-check: synthesis is not interruptible, so the world may have
                # changed while it ran. Never queue audio for a cancelled turn.
                if self._cancel.is_set():
                    continue
                if chunk.samples.size == 0:
                    # The chunk was nothing but a bracketed aside.
                    continue
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                    self._set_state(State.SPEAKING)
                self.speaker.enqueue(chunk.text, chunk.samples)

        worker = threading.Thread(target=synth_worker, name="tts", daemon=True)
        worker.start()

        self.speaker.reset_spoken()
        sys.stdout.write(f"  {c('jarvis', GREEN)} ")
        sys.stdout.flush()
        schemas = self.registry.schemas() if self.registry is not None else None
        stream = None
        # The model may ask for tools, receive their results, and only then
        # speak. Each pass is one LLM request; bounded so a model that keeps
        # calling tools cannot spin forever.
        MAX_TOOL_ROUNDS = 3
        first_gen: Generation | None = None
        reply_text = ""
        awaiting_confirmation = False
        # Time to the first token the user will actually HEAR. On a tool turn the
        # first round emits only tool calls and no content, so per-round ttft is
        # None there; reporting that as 0 ms made tool turns look instant when
        # they are in fact the slowest thing the assistant does.
        first_text_at: float | None = None

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                gen = Generation()
                stream = self.llm.stream(
                    self.convo.for_request(),
                    cancel=self._cancel,
                    into=gen,
                    tools=schemas,
                )
                for delta in stream:
                    if delta.is_reasoning:
                        continue
                    if self._cancel.is_set():
                        break
                    if first_text_at is None:
                        first_text_at = time.monotonic()
                    sys.stdout.write(delta.text)
                    sys.stdout.flush()
                    for chunk_text in chunker.feed(delta.text):
                        record.chunks += 1
                        synth_queue.put(chunk_text)
                stream.close()

                if first_gen is None:
                    first_gen = gen
                reply_text += gen.text

                if self._cancel.is_set() or gen.error:
                    break

                if not gen.wants_tools:
                    for chunk_text in chunker.flush():
                        record.chunks += 1
                        synth_queue.put(chunk_text)
                    break

                record.tools_used += len(gen.tool_calls)
                if not self._run_tools(gen):
                    # A destructive call is waiting on a spoken yes; the question
                    # has already been asked out loud.
                    awaiting_confirmation = True
                    break
        finally:
            # Only a stream that was still generating counts towards the §7.3
            # cancel budget. The LLM usually finishes long before playback does,
            # and recording those as 0.0 ms made the metric look perfect while
            # measuring nothing at all.
            was_generating = gen.finish_reason is None and gen.error is None
            cancel_started = time.monotonic()
            if stream is not None:
                stream.close()
            if self._cancel.is_set() and was_generating:
                self.metrics.cancel_latencies_ms.append(
                    (time.monotonic() - cancel_started) * 1000.0
                )
            synth_queue.put(None)
            # Wait for synthesis to drain, but stay interruptible while waiting.
            #
            # Kokoro's create() is uninterruptible and lock-serialized, so a chunk
            # caught mid-synthesis runs to completion (measured up to 1.7s). The
            # LLM usually finishes well before playback does, which means we are
            # normally sitting in THIS wait when a barge-in lands. A plain join
            # would add that stall straight onto the reply to the interruption,
            # the one interaction where latency matters most. So poll instead and
            # abandon the worker the moment we are cancelled: it discards its own
            # output because _cancel is set, so leaving it detached is harmless.
            deadline = time.monotonic() + 30
            while worker.is_alive() and time.monotonic() < deadline:
                if self._cancel.is_set() or self._stop.is_set():
                    break
                worker.join(timeout=0.02)
            print()

        if gen.error:
            self._say(c(f"  ! LLM failed: {gen.error}", RED))
            if "devicelost" in gen.error.lower().replace(" ", ""):
                # The GPU dropped out from under llama-server. It cannot recover
                # on its own; every later request fails the same way, so say so
                # once instead of leaving the user talking to a mute assistant.
                self._say(c("  ! The GPU was lost. Restart the servers: .\\run.ps1 -Restart", RED))
            self.metrics.event("llm_error", error=gen.error)
            self._speak_now("Something went wrong with the model. Check the terminal.")
            self._set_state(State.LISTENING)
            return

        record.ttft_ms = (
            (first_text_at - llm_started) * 1000.0 if first_text_at is not None else 0.0
        )
        record.reply = reply_text
        if first_audio_at is not None:
            record.first_audio_ms = max(
                0.0, (first_audio_at - llm_started) * 1000.0 - record.ttft_ms
            )

        # Wait for playback, but stay interruptible while waiting.
        while self.speaker.is_speaking and not self._cancel.is_set() and not self._stop.is_set():
            self.speaker.wait_until_idle(timeout=0.05)

        interrupted = self._cancel.is_set()
        record.interrupted = interrupted
        spoken = self.speaker.spoken_text()
        record.spoken_text = spoken

        if awaiting_confirmation:
            # The confirmation question was spoken through a separate path; the
            # tool call and question are already in history.
            self._set_state(State.LISTENING)
            return

        if interrupted:
            # §7.3 item 3: only what the user actually HEARD may enter history.
            # Text generated but never played is dropped outright, otherwise the
            # model behaves as though the user knows things they never heard.
            record.dropped_text = (
                reply_text[len(spoken):].strip() if reply_text.startswith(spoken) else ""
            )
            self.convo.add_assistant(spoken, interrupted=True)
        else:
            self.convo.add_assistant(spoken or reply_text)

        if not reply_text.strip() and not spoken:
            # Cancelled before a single token arrived. Recording this as a turn
            # would drag 0 ms TTFT and 0 ms first-audio into the medians and make
            # the latency summary lie.
            self.metrics.event("empty_turn", transcript=record.transcript)
            self._say(c("    (interrupted before it answered)", DIM))
            self._set_state(State.LISTENING)
            return

        if spoken:
            self._last_spoken_reply = spoken

        self.metrics.add_turn(record)
        self._say(
            c(
                f"    STT {record.stt_ms:.0f}ms | TTFT {record.ttft_ms:.0f}ms | "
                f"audio +{record.first_audio_ms:.0f}ms | trip {record.round_trip_ms:.0f}ms"
                + (f" | smart-turn {record.smart_turn_ms:.0f}ms" if record.smart_turn_ms else "")
                + (f" | {record.tools_used} tool" if record.tools_used else "")
                + (" | INTERRUPTED" if interrupted else ""),
                DIM,
            )
        )

        # "Go to sleep" only takes effect once the acknowledgement has actually
        # been spoken, otherwise the reply is cut off by its own command.
        if self._sleep_requested:
            self._sleep_requested = False
            self._go_to_sleep("as asked")
            return

        self._set_state(State.LISTENING)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utt = self._work.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handle(utt)
            except Exception as exc:  # noqa: BLE001
                self._say(c(f"  ! turn failed: {type(exc).__name__}: {exc}", RED))
                self.metrics.event("turn_error", error=f"{type(exc).__name__}: {exc}")
                self._set_state(State.LISTENING)

    # -- capture loop ------------------------------------------------------

    def _capture(self, mic: MicrophoneStream) -> None:
        """The only loop that touches the microphone. It must never block."""
        for frame in mic.frames():
            if self._stop.is_set():
                return

            confidence = self.vad.confidence(frame)

            # IDLE: nothing but the wake word matters. Cheap enough to run on
            # every frame (measured 0.5-0.9 ms), unlike the turn probe.
            if self.state is State.IDLE:
                if self.wake is None:
                    continue
                # Buffer while asleep. Detection latency is variable and gets
                # worse as the score drops: a clean 0.99 fires ~40 ms after the
                # phrase, but a real 0.82 fires much later, by which point the
                # user is already several words into their request. Without this
                # buffer those words are simply discarded and the transcript
                # starts mid-sentence.
                self._wake_preroll.append(frame)
                if len(self._wake_preroll) > self._wake_preroll_max:
                    self._wake_preroll.pop(0)

                event = self.wake.process(frame)
                if event is not None:
                    self.metrics.event("wake_word", score=round(event.score, 3))
                    self._say(c(f"  * awake (score {event.score:.2f})", GREEN))
                    self.endpointer.reset()
                    self.barge.reset()
                    self._preroll.clear()
                    self._arm_follow_up()
                    self._set_state(State.LISTENING)
                    # Replay what was said before and during detection. The wake
                    # phrase itself comes along; it is stripped from the
                    # transcript rather than from the audio, because cutting the
                    # audio accurately would need to know where the phrase ended.
                    if self.smart_turn is not None:
                        self.smart_turn.clear()
                    for buffered in self._wake_preroll:
                        conf = self.vad.confidence(buffered)
                        if self.smart_turn is not None:
                            self.smart_turn.append(
                                buffered, conf >= self.endpointer.vad_threshold
                            )
                        produced = self.endpointer.process(buffered)
                        if produced is not None:
                            self._work.put(produced)
                    self._wake_preroll.clear()
                continue

            busy = self.state not in (State.LISTENING,)

            if busy and self.barge_enabled:
                # Turn detection stays live during playback. This is §7.3.
                decision = self.barge.process(confidence, self.vad.dbfs(frame))
                self._preroll.append(frame)
                if len(self._preroll) > self._preroll_max:
                    self._preroll.pop(0)

                if decision.state is BargeInState.CONFIRMED:
                    onset = self.barge.onset_mono
                    self._interrupt_now(decision.speech_ms, decision.reason, onset)
                    self.barge.reset()
                    # Replay what was said during the interruption so the new turn
                    # starts from the first word, not from wherever we noticed.
                    self.endpointer.reset()
                    for buffered in self._preroll:
                        produced = self.endpointer.process(buffered)
                        if produced is not None:
                            self._work.put(produced)
                    self._preroll.clear()
                elif decision.reason.startswith("backchannel"):
                    self.metrics.backchannels_ignored += 1
                    self.metrics.event(
                        "backchannel", speech_ms=round(decision.speech_ms, 1)
                    )
                    self._say(
                        c(f"  . backchannel ({decision.speech_ms:.0f}ms), still talking", DIM)
                    )
                continue

            if busy:
                continue

            # LISTENING.
            self.barge.reset()
            self._preroll.clear()
            if self.smart_turn is not None:
                self.smart_turn.append(frame, confidence >= self.endpointer.vad_threshold)
            utt = self.endpointer.process(frame)
            if utt is not None:
                if self.smart_turn is not None:
                    self.smart_turn.clear()
                self._work.put(utt)
                continue

            # §7.5: the follow-up window keeps the floor open after a reply so a
            # back-and-forth does not need the wake word every turn. It only
            # expires while genuinely idle; a turn in progress holds it open.
            if (
                self.wake_enabled
                and self.endpointer.state is TurnState.QUIET
                and self._follow_up_until
                and time.monotonic() > self._follow_up_until
            ):
                self.metrics.event("follow_up_expired")
                self._go_to_sleep("follow-up window expired")

    # -- entry point -------------------------------------------------------

    def run(self) -> int:
        if not self.stt.is_up():
            print(c(f"whisper-server is not answering at {self.cfg.whisper_url}", RED))
            print("Start it with:  .\\run.ps1")
            return 1
        if not self.llm.is_up():
            print(c(f"llama-server is not answering at {self.cfg.llm_url}", RED))
            print("Start it with:  .\\run.ps1")
            return 1

        print()
        print(c("JARVIS M5: conversation with barge-in", BOLD))
        print(f"  mic       : {describe_device(self.in_device)}")
        print(f"  speaker   : {describe_output_device(self.out_device)}")
        print(f"  model     : {self.llm.model_name()}")
        print(f"  voice     : {self.tts.voice}")
        print(
            f"  endpoint  : {self.endpointer.base_silence_ms}ms base, "
            f"up to {self.endpointer.max_wait_ms}ms when unfinished"
        )
        print(
            f"  barge-in  : {'on' if self.barge_enabled else 'OFF'}, "
            f"confirm after {self.barge.confirm_ms}ms of speech, "
            f"echo mode {self.barge.echo_mode}"
        )
        if self.wake is not None:
            phrase = self.wake.name.replace("_", " ")
            print(
                f"  wake word : \"{phrase}\" at threshold {self.wake.threshold}, "
                f"then {self.follow_up_window_ms / 1000:.0f}s follow-up window"
            )
        else:
            print("  wake word : off, always listening")
        if self.registry is not None:
            print(f"  tools     : {', '.join(self.registry.names)}")
        else:
            print("  tools     : off")
        if self.memory is not None:
            print(f"  memory    : {len(self.memory)} facts from {self.memory.path}")
        warm = self.tts.warm_up()
        smart_ms = self.smart_turn.warm_up() if self.smart_turn else 0.0
        wake_ms = self.wake.warm_up() if self.wake else 0.0
        print(
            f"  warm-up   : TTS {warm:.0f}ms, SmartTurn {smart_ms:.0f}ms"
            + (f", wake {wake_ms:.0f}ms" if self.wake else "")
        )
        print(f"  metrics   : {self.metrics.path}")
        print()
        if self.barge.echo_mode == "headphones":
            print(c("  Wear headphones. On speakers it will hear itself (§7.1).", YELLOW))
        if self.wake is not None:
            print(c(f"  Say \"{self.wake.name.replace('_', ' ')}\" to wake it.", DIM))
        print(c("  Interrupt it whenever you like. Ctrl+C for the summary.", DIM))
        print()

        self.metrics.event("session_start", voice=self.tts.voice, model=self.llm.model_name())

        worker = threading.Thread(target=self._worker_loop, name="turn", daemon=True)
        worker.start()

        mic = MicrophoneStream(fmt=self.fmt, device=self.in_device)
        try:
            with self.speaker, mic:
                self._capture(mic)
        except KeyboardInterrupt:
            print("\n")
        finally:
            self._stop.set()
            self._cancel.set()
            worker.join(timeout=5)
            self.close()

        print()
        print(c("=" * 68, DIM))
        for line in self.metrics.summary_lines():
            print(line)
        print(c("=" * 68, DIM))
        return 0


def run(cfg: Config, *, device_override=None, out_device_override=None, verbose: bool = False) -> int:
    return Assistant(
        cfg,
        device_override=device_override,
        out_device_override=out_device_override,
        verbose=verbose,
    ).run()
