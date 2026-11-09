"""M3: text -> Kokoro -> speakers, sentence-chunked and flushable.

Two modes:

  --speak            type text, hear it spoken
  --speak --flush-test   scripted proof that the queue flushes inside 60 ms

M3's acceptance criterion is not "it makes noise", it is "the audio queue is
verifiably flushable mid-sentence" (§9). The flush test is therefore the real
deliverable: it starts a long utterance, interrupts it at a known moment, and
reports the measured flush latency against the 60 ms budget from §3.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .audio.output import SpeakerOutput, describe_output_device, resolve_output_device
from .config import Config
from .services.kokoro_tts import KokoroTTS

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

FLUSH_BUDGET_MS = 60.0     # spec §3
FIRST_AUDIO_BUDGET_MS = 150.0


def _tty() -> bool:
    return sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return text if not _tty() else f"{color}{text}{RESET}"


def _build(cfg: Config, device_override=None) -> tuple[KokoroTTS, SpeakerOutput]:
    tts = KokoroTTS(
        cfg.tts_model,
        cfg.tts_voices,
        voice=str(cfg.get("tts.voice", "af_heart")),
        speed=float(cfg.get("tts.speed", 1.0)),
        max_chunk_chars=int(cfg.get("tts.max_chunk_chars", 180)),
        first_chunk_chars=int(cfg.get("tts.first_chunk_chars", 60)),
    )
    device = resolve_output_device(
        device_override if device_override is not None else cfg.get("audio.output_device")
    )
    speaker = SpeakerOutput(
        sample_rate=tts.sample_rate,
        device=device,
        blocksize=int(cfg.get("audio.output_blocksize", 256)),
        device_sample_rate=cfg.get("audio.output_sample_rate", None),
    )
    return tts, speaker


def _speak(
    tts: KokoroTTS,
    speaker: SpeakerOutput,
    pool: ThreadPoolExecutor,
    text: str,
    cancel: threading.Event | None = None,
):
    """Start speaking `text`, returning as soon as the first chunk is queued.

    Only the first chunk is synthesized inline; the rest continue on the pool
    while that first chunk is already playing. Blocking here until the whole
    response is synthesized would serialize TTS behind the LLM at M4 and destroy
    the latency budget.

    Returns (ms until first chunk was playable, future for the remainder).
    """
    chunks = tts.split(text)
    started = time.monotonic()
    if not chunks:
        return 0.0, None

    first = tts.synthesize(chunks[0])
    first_audio_ms = (time.monotonic() - started) * 1000.0
    if cancel is not None and cancel.is_set():
        return first_audio_ms, None
    if first.samples.size:
        speaker.enqueue(first.text, first.samples)

    def synthesize_rest() -> None:
        for chunk_text in chunks[1:]:
            if cancel is not None and cancel.is_set():
                return
            chunk = tts.synthesize(chunk_text)
            if cancel is not None and cancel.is_set():
                return
            if chunk.samples.size:
                speaker.enqueue(chunk.text, chunk.samples)

    future = pool.submit(synthesize_rest) if len(chunks) > 1 else None
    return first_audio_ms, future


def run_flush_test(cfg: Config, device_override=None, rounds: int = 5) -> int:
    tts, speaker = _build(cfg, device_override)

    print()
    print(c("JARVIS M3: flush latency test", BOLD))
    print(f"  voice   : {tts.voice} at {tts.sample_rate} Hz")
    print(f"  speaker : {describe_output_device(speaker.device)}")

    warm_ms = tts.warm_up()
    print(f"  warm-up : {warm_ms:.0f} ms (first synthesis pays ONNX setup)")
    print()

    long_text = (
        "This is a deliberately long sentence that keeps going for a while, "
        "because the whole point of this test is to interrupt it somewhere in "
        "the middle and measure how quickly the audio actually stops. "
        "If you can still hear me talking after the flush, then the queue is "
        "not being cleared properly and barge-in will never work. "
        "I will keep talking until somebody stops me."
    )

    audible: list[float] = []
    with speaker, ThreadPoolExecutor(max_workers=1) as pool:
        device_ms = speaker.latency_ms
        print(f"  device rate    : {speaker.device_sample_rate} Hz (Kokoro 24000 Hz resampled)")
        print(f"  device buffer  : {device_ms:.1f} ms of audio already committed to hardware")
        if device_ms > FLUSH_BUDGET_MS:
            print(
                c(
                    f"  WARNING: the device buffer alone exceeds the {FLUSH_BUDGET_MS:.0f} ms budget. "
                    "Barge-in cannot meet spec on this host API.",
                    RED,
                )
            )
        print()
        for round_index in range(1, rounds + 1):
            speaker.reset_spoken()
            cancel = threading.Event()
            chunks = tts.split(long_text)
            first_ms, future = _speak(tts, speaker, pool, long_text, cancel)
            print(
                f"  round {round_index}: {len(chunks)} chunks "
                f"(first is {len(chunks[0])} chars), first audio in {first_ms:.0f} ms"
            )

            # Let it actually start speaking, then interrupt mid-sentence.
            time.sleep(1.2)
            cancel.set()
            result = speaker.flush()

            # Kokoro serializes synthesis behind a lock and create() cannot be
            # interrupted, so a chunk already in flight runs to completion. Drain
            # it here, otherwise the next round's first chunk queues behind it and
            # its first-audio number measures lock contention instead of synthesis.
            if future is not None:
                future.result()
            audible.append(result.audible_stop_ms)

            verdict = (
                c("OK", GREEN) if result.audible_stop_ms <= FLUSH_BUDGET_MS else c("OVER", RED)
            )
            print(
                f"           queue cleared in {result.latency_ms:.2f} ms, "
                f"+{result.device_latency_ms:.1f} ms device buffer "
                f"= {result.audible_stop_ms:.1f} ms until silence [{verdict}]"
            )
            print(c(f"           dropped {result.dropped_chunks} queued chunks", DIM))
            print(c(f"           heard   : {result.spoken_text[:84]!r}", DIM))
            print(c(f"           dropped : {result.dropped_text[:84]!r}", DIM))

            # Confirm silence: nothing may be queued after a flush.
            time.sleep(0.3)
            if speaker.is_speaking:
                print(c("           STILL SPEAKING AFTER FLUSH", RED))
            print()

    median = statistics.median(audible)
    worst = max(audible)
    print(c("=" * 72, DIM))
    print(f"  {len(audible)} flushes | time to silence: median {median:.1f} ms | max {worst:.1f} ms")
    print(c("  (queue clear + device buffer, which is what the user actually hears)", DIM))
    if worst <= FLUSH_BUDGET_MS:
        print(c(f"  M3 target met: silence within the {FLUSH_BUDGET_MS:.0f} ms budget", GREEN))
    else:
        print(c(f"  M3 target MISSED: {worst:.1f} ms > {FLUSH_BUDGET_MS:.0f} ms budget", RED))
    print(c("=" * 72, DIM))
    return 0 if worst <= FLUSH_BUDGET_MS else 1


def run(cfg: Config, device_override=None) -> int:
    tts, speaker = _build(cfg, device_override)

    print()
    print(c("JARVIS M3: text to speech", BOLD))
    print(f"  voice   : {tts.voice} at {tts.sample_rate} Hz")
    print(f"  speaker : {describe_output_device(speaker.device)} at {speaker.device_sample_rate} Hz")
    warm_ms = tts.warm_up()
    print(f"  warm-up : {warm_ms:.0f} ms")
    print()
    print(c("  Type something and press Enter to hear it.", DIM))
    print(c("  /stop interrupts playback, /voices lists voices, /quit exits.", DIM))
    print()

    first_audio: list[float] = []
    try:
        with speaker, ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                try:
                    text = input(c("say > ", CYAN)).strip()
                except EOFError:
                    break
                if not text:
                    continue
                if text in ("/quit", "/exit"):
                    break
                if text == "/voices":
                    print(c("  " + ", ".join(tts.voices()), DIM))
                    continue
                if text == "/stop":
                    result = speaker.flush()
                    print(
                        c(
                            f"  silence in {result.audible_stop_ms:.1f} ms "
                            f"({result.latency_ms:.2f} ms queue + "
                            f"{result.device_latency_ms:.1f} ms device buffer)",
                            YELLOW,
                        )
                    )
                    continue

                speaker.reset_spoken()
                chunks = tts.split(text)
                print(c(f"  {len(chunks)} chunk(s): {chunks}", DIM))
                ms, future = _speak(tts, speaker, pool, text)
                first_audio.append(ms)
                mark = "" if ms <= FIRST_AUDIO_BUDGET_MS else f" (over {FIRST_AUDIO_BUDGET_MS:.0f}ms budget)"
                print(c(f"  first audio in {ms:.0f} ms{mark}", DIM))
                if future is not None:
                    future.result()
                speaker.wait_until_idle(timeout=120)
    except KeyboardInterrupt:
        print()
    finally:
        speaker.close()

    if first_audio:
        print()
        print(f"  first-audio median {statistics.median(first_audio):.0f} ms over {len(first_audio)} utterances")
    return 0
