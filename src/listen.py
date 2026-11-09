"""M1: microphone -> Silero VAD -> whisper.cpp -> printed transcript.

Deliberately NOT a pipecat pipeline yet. M1 is a vertical slice, and a hand-rolled
loop makes the latency of each stage directly measurable, which is the whole
point of this milestone.

The capture loop never blocks on transcription. Whisper takes over a second on
CPU, and stalling the frame loop for that long would drop microphone audio and
make barge-in impossible later. Transcription runs on a worker thread while
frames keep flowing.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .audio.io import AudioFormat, MicrophoneStream, describe_device, resolve_device
from .audio.vad import Endpointer, TurnEvent, Utterance, VoiceActivityDetector
from .config import Config
from .services.whisper_stt import Transcript, WhisperClient

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _plain() -> bool:
    return not sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return text if _plain() else f"{color}{text}{RESET}"


@dataclass
class TurnMetrics:
    audio_s: float
    silence_wait_ms: float
    stt_ms: float
    queue_ms: float
    text: str

    @property
    def endpoint_to_text_ms(self) -> float:
        """The M1 target: endpoint decision -> transcript in hand."""
        return self.queue_ms + self.stt_ms

    @property
    def perceived_ms(self) -> float:
        """What the user feels: stopped talking -> transcript on screen."""
        return self.silence_wait_ms + self.endpoint_to_text_ms


@dataclass
class Session:
    turns: list[TurnMetrics] = field(default_factory=list)
    discarded: int = 0
    empty: int = 0

    def summary(self) -> str:
        if not self.turns:
            return "no turns captured"
        stt = sorted(t.stt_ms for t in self.turns)
        e2t = sorted(t.endpoint_to_text_ms for t in self.turns)
        perceived = sorted(t.perceived_ms for t in self.turns)

        def pct(values: list[float], p: float) -> float:
            if len(values) == 1:
                return values[0]
            k = max(0, min(len(values) - 1, int(round((len(values) - 1) * p))))
            return values[k]

        lines = [
            "",
            c("=" * 74, DIM),
            f"{len(self.turns)} turns transcribed"
            + (f", {self.empty} empty" if self.empty else "")
            + (f", {self.discarded} discarded as too short" if self.discarded else ""),
            "",
            f"  {'':<22}{'median':>10}{'p95':>10}{'max':>10}",
            f"  {'whisper STT':<22}{statistics.median(stt):>9.0f}ms{pct(stt, 0.95):>9.0f}ms{stt[-1]:>9.0f}ms",
            f"  {'endpoint -> text':<22}{statistics.median(e2t):>9.0f}ms{pct(e2t, 0.95):>9.0f}ms{e2t[-1]:>9.0f}ms",
            f"  {'stopped -> text':<22}{statistics.median(perceived):>9.0f}ms"
            f"{pct(perceived, 0.95):>9.0f}ms{perceived[-1]:>9.0f}ms",
            "",
        ]
        median_e2t = statistics.median(e2t)
        target = 400.0
        if median_e2t <= target:
            lines.append(c(f"  M1 target met: endpoint -> text median {median_e2t:.0f}ms <= {target:.0f}ms", GREEN))
        else:
            lines.append(
                c(f"  M1 target MISSED: endpoint -> text median {median_e2t:.0f}ms > {target:.0f}ms", YELLOW)
            )
            lines.append(
                c("  Lever: whisper.audio_ctx in config.yaml (see the measured table there).", DIM)
            )
        lines.append(c("=" * 74, DIM))
        return "\n".join(lines)


def run(cfg: Config, *, show_meter: bool = False, device_override: str | int | None = None) -> int:
    audio_cfg = cfg.section("audio")
    turn_cfg = cfg.section("turn")

    fmt = AudioFormat(
        sample_rate=int(audio_cfg.get("input_sample_rate", 16000)),
        channels=int(audio_cfg.get("channels", 1)),
    )
    if fmt.sample_rate != 16000:
        print(c(f"Silero VAD v5 requires 16 kHz; config says {fmt.sample_rate}.", RED))
        return 1

    device = resolve_device(
        device_override if device_override is not None else audio_cfg.get("input_device")
    )

    stt = WhisperClient(
        cfg.whisper_url,
        language=str(cfg.get("whisper.language", "en")),
    )
    if not stt.is_up():
        print(c(f"whisper-server is not answering at {cfg.whisper_url}", RED))
        print("Start it with:  .\\run.ps1")
        return 1

    vad = VoiceActivityDetector(
        sample_rate=fmt.sample_rate,
        confidence=float(turn_cfg.get("vad_threshold", 0.5)),
    )

    session = Session()
    print_lock = threading.Lock()

    def on_event(ev: TurnEvent) -> None:
        if ev.kind in ("speech_start", "turn_extended", "discarded"):
            with print_lock:
                label = {"speech_start": "listening", "turn_extended": "waiting (turn incomplete)",
                         "discarded": "ignored"}[ev.kind]
                print(c(f"  . {label} {ev.detail}", DIM))
            if ev.kind == "discarded":
                session.discarded += 1

    endpointer = Endpointer(
        vad,
        vad_threshold=float(turn_cfg.get("vad_threshold", 0.5)),
        base_silence_ms=int(turn_cfg.get("base_silence_ms", 700)),
        max_wait_ms=int(turn_cfg.get("max_wait_ms", 3000)),
        min_speech_ms=int(turn_cfg.get("min_speech_ms", 200)),
        frame_ms=fmt.frame_ms,
        on_event=on_event,
    )

    def transcribe(utt: Utterance, queued_at: float) -> None:
        queue_ms = (time.monotonic() - queued_at) * 1000.0
        try:
            result: Transcript = stt.transcribe_pcm(utt.audio, utt.sample_rate)
        except Exception as exc:  # noqa: BLE001 - surface anything the server does
            with print_lock:
                print(c(f"  ! STT failed: {type(exc).__name__}: {exc}", RED))
            return

        if result.is_empty:
            session.empty += 1
            with print_lock:
                print(c(f"  . no speech recognised ({result.latency_ms:.0f}ms)", DIM))
            return

        m = TurnMetrics(
            audio_s=result.audio_s,
            silence_wait_ms=utt.silence_wait_ms,
            stt_ms=result.latency_ms,
            queue_ms=queue_ms,
            text=result.text,
        )
        session.turns.append(m)
        with print_lock:
            print(f'  {c(">", GREEN)} {c(result.text, BOLD)}')
            print(
                c(
                    f"    {m.audio_s:.2f}s audio | STT {m.stt_ms:.0f}ms | "
                    f"endpoint->text {m.endpoint_to_text_ms:.0f}ms | "
                    f"stopped->text {m.perceived_ms:.0f}ms",
                    DIM,
                )
            )

    print()
    print(c("JARVIS M1: speech to text", BOLD))
    print(f"  mic       : {describe_device(device)}")
    print(f"  format    : {fmt.sample_rate} Hz mono, {fmt.frame_samples}-sample frames ({fmt.frame_ms:.0f} ms)")
    print(f"  whisper   : {cfg.whisper_url}")
    ac = int(cfg.get("whisper.audio_ctx", 0))
    print(f"  audio_ctx : {'default (full 30s window)' if ac == 0 else ac}")
    print(
        f"  endpoint  : {endpointer.base_silence_ms}ms silence, "
        f"threshold {endpointer.vad_threshold}, min speech {endpointer.min_speech_ms}ms"
    )
    print()
    print(c("  Speak. Ctrl+C to stop and print the latency summary.", DIM))
    print()

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")
    mic = MicrophoneStream(fmt=fmt, device=device)
    last_meter = 0.0

    try:
        with mic:
            for frame in mic.frames():
                utt = endpointer.process(frame)
                if utt is not None:
                    executor.submit(transcribe, utt, time.monotonic())

                if show_meter:
                    now = time.monotonic()
                    if now - last_meter > 0.1:
                        last_meter = now
                        conf = vad.confidence(frame)
                        level = vad.dbfs(frame)
                        bar = "#" * int(min(1.0, conf) * 30)
                        with print_lock:
                            sys.stdout.write(
                                f"\r    vad {conf:4.2f} {bar:<30} {level:6.1f} dBFS  "
                                f"[{endpointer.state.value}]   "
                            )
                            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n")
    finally:
        executor.shutdown(wait=True)
        stt.close()
        if mic.dropped_frames or mic.overflow_count:
            print(
                c(
                    f"  audio warnings: {mic.dropped_frames} frames dropped, "
                    f"{mic.overflow_count} input overflows",
                    YELLOW,
                )
            )

    print(session.summary())
    return 0
