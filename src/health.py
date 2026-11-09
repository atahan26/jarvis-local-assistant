"""Service health checks for M0.

Every check returns a Check with an explicit pass/fail and a human-readable
detail line. Nothing here guesses: the llama-server contract used below was read
off the current tools/server README (GET /health -> 503 while loading, 200
{"status":"ok"} when ready) and `llama-server --list-devices` output was verified
on this machine.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True

    @property
    def symbol(self) -> str:
        if self.ok:
            return "PASS"
        return "FAIL" if self.fatal else "WARN"


# ---------------------------------------------------------------------------
# Static checks: files on disk
# ---------------------------------------------------------------------------


def check_artifacts(cfg: Config) -> list[Check]:
    expected: list[tuple[str, Path, int]] = [
        ("llama-server.exe", cfg.llama_exe, 0),
        ("whisper-server.exe", cfg.whisper_exe, 0),
        ("whisper model", cfg.whisper_model, 150),
        ("llm model", cfg.llm_model, 4000),
        ("kokoro model", cfg.tts_model, 250),
        ("kokoro voices", cfg.tts_voices, 20),
    ]
    checks = []
    for label, path, min_mb in expected:
        if not path.exists():
            checks.append(Check(label, False, f"missing: {path}"))
            continue
        mb = path.stat().st_size / (1024 * 1024)
        if min_mb and mb < min_mb:
            checks.append(
                Check(label, False, f"truncated: {mb:.1f} MB, expected >= {min_mb} MB ({path})")
            )
            continue
        checks.append(Check(label, True, f"{mb:.1f} MB  {path}"))
    return checks


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

_DEVICE_RE = re.compile(r"^\s*(?P<id>\w+\d+):\s*(?P<name>.+?)\s*\((?P<total>\d+)\s*MiB", re.M)


def check_gpu(cfg: Config) -> Check:
    """Ask llama-server itself which backends/devices it can see.

    This needs no model, so it is a cheap pre-flight before paying the 5 GB load.
    """
    if not cfg.llama_exe.exists():
        return Check("vulkan gpu", False, f"llama-server.exe missing: {cfg.llama_exe}")
    try:
        proc = subprocess.run(
            [str(cfg.llama_exe), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("vulkan gpu", False, f"--list-devices failed: {exc}")

    out = (proc.stdout or "") + (proc.stderr or "")
    devices = _DEVICE_RE.findall(out)
    vulkan = [d for d in devices if d[0].lower().startswith("vulkan")]
    if not vulkan:
        return Check(
            "vulkan gpu",
            False,
            "no Vulkan device reported. Update the AMD driver / Vulkan runtime. "
            f"Raw output: {out.strip()[:400]!r}",
        )
    dev_id, name, total = vulkan[0]
    return Check("vulkan gpu", True, f"{dev_id}: {name} ({int(total)} MiB)")


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------


def wait_for_llm(cfg: Config, timeout_s: float | None = None) -> Check:
    timeout_s = timeout_s if timeout_s is not None else cfg.get("llm.startup_timeout_s", 180)
    url = f"{cfg.llm_url}/health"
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                elapsed = timeout_s - (deadline - time.monotonic())
                return Check("llama-server /health", True, f"ready after {elapsed:.1f}s")
            last = f"HTTP {r.status_code} {r.text.strip()[:120]}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(0.5)
    return Check("llama-server /health", False, f"not ready within {timeout_s}s (last: {last})")


def check_llm_props(cfg: Config) -> Check:
    try:
        r = httpx.get(f"{cfg.llm_url}/props", timeout=10.0)
        r.raise_for_status()
        props = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Check("llama-server /props", False, f"{type(exc).__name__}: {exc}")
    build = props.get("build_info", "?")
    model = Path(str(props.get("model_path", "?"))).name
    return Check("llama-server /props", True, f"model={model} build={build}")


_OFFLOAD_RE = re.compile(r"offloaded\s+(?P<on>\d+)\s*/\s*(?P<total>\d+)\s+layers to GPU")
# Fallback for higher verbosity levels, which print per-layer assignment instead
# of (as well as) the summary line.
_LAYER_RE = re.compile(r"load_tensors:\s*layer\s+(?P<n>\d+)\s+assigned to device\s+(?P<dev>\S+)")


def check_llm_offload(cfg: Config, log_dir: Path) -> Check:
    """Confirm -ngl 999 actually offloaded every layer (spec §5).

    llama-server prints the offload summary at load time and there is no HTTP
    endpoint that exposes it, so its log is the source of truth. Windows cannot
    redirect stdout and stderr of one process to the same file, so run.ps1 writes
    two files and both are searched.
    """
    require_gpu = bool(cfg.get("llm.require_gpu", True))
    logs = sorted(log_dir.glob("llama-server*.log"))
    if not logs:
        return Check(
            "llm gpu offload", False, f"no llama-server*.log in {log_dir}", fatal=require_gpu
        )
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)

    on = total = None
    match = None
    for match in _OFFLOAD_RE.finditer(text):
        pass  # keep the last occurrence, the most recent load
    if match:
        on, total = int(match.group("on")), int(match.group("total"))
    else:
        # Higher -lv prints one line per layer instead of a summary.
        layers = _LAYER_RE.findall(text)
        if layers:
            total = len(layers)
            on = sum(1 for _, dev in layers if not dev.upper().startswith("CPU"))

    if on is None:
        return Check(
            "llm gpu offload",
            False,
            "no offload report in the server log. llama-server suppresses it below "
            "-lv 4; check llm.log_verbosity in config.yaml",
            fatal=require_gpu,
        )
    if on < total:
        return Check(
            "llm gpu offload",
            False,
            f"only {on}/{total} layers on GPU; the rest are running on CPU",
            fatal=require_gpu,
        )

    extra = []
    vram_match = re.search(r"Vulkan\d+ model buffer size\s*=\s*([\d.]+)\s*MiB", text)
    if vram_match:
        extra.append(f"{float(vram_match.group(1)) / 1024:.2f} GiB weights")
    kv_match = re.search(r"Vulkan\d+ KV buffer size\s*=\s*([\d.]+)\s*MiB", text)
    if kv_match:
        extra.append(f"{float(kv_match.group(1)) / 1024:.2f} GiB KV cache")
    suffix = f" ({', '.join(extra)} on Vulkan0)" if extra else ""
    return Check("llm gpu offload", True, f"{on}/{total} layers on GPU{suffix}")


# ---------------------------------------------------------------------------
# whisper-server
# ---------------------------------------------------------------------------


def wait_for_whisper(cfg: Config, timeout_s: float | None = None) -> Check:
    """whisper-server has no /health route; any HTTP reply proves it is serving.

    404 from the static-file handler is a perfectly good liveness signal here.
    """
    timeout_s = timeout_s if timeout_s is not None else cfg.get("whisper.startup_timeout_s", 60)
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{cfg.whisper_url}/", timeout=3.0)
            elapsed = timeout_s - (deadline - time.monotonic())
            return Check(
                "whisper-server listening", True, f"HTTP {r.status_code} after {elapsed:.1f}s"
            )
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(0.5)
    return Check("whisper-server listening", False, f"not ready within {timeout_s}s (last: {last})")


def check_whisper_inference(cfg: Config) -> Check:
    """Round-trip a tiny WAV through /inference.

    Proves the model is loaded and the endpoint contract holds, and gives an
    early read on CPU decode cost before M1 measures it properly.
    """
    import io
    import struct
    import wave

    sample_rate = 16000
    seconds = 1.0
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<%dh" % int(sample_rate * seconds), *([0] * int(sample_rate * seconds))))
    payload = buf.getvalue()

    started = time.monotonic()
    try:
        r = httpx.post(
            f"{cfg.whisper_url}/inference",
            files={"file": ("probe.wav", payload, "audio/wav")},
            data={"response_format": "json", "temperature": "0.0"},
            timeout=60.0,
        )
        r.raise_for_status()
        body = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Check("whisper /inference", False, f"{type(exc).__name__}: {exc}")
    elapsed_ms = (time.monotonic() - started) * 1000
    text = str(body.get("text", "")).strip()
    return Check(
        "whisper /inference",
        True,
        f"1s of silence round-tripped in {elapsed_ms:.0f} ms (text={text!r})",
    )
