"""Wake word detection (spec §3, §7.5).

openWakeWord ships a pretrained `hey_jarvis` model, which is the entire reason
the spec picked this wake phrase.

Two things had to be pinned down, both verified rather than assumed:

* **ONNX is mandatory here, not optional.** openWakeWord defaults to tflite, but
  neither `tflite_runtime` nor `tensorflow` installs on Windows/Python 3.11.
  Only `onnxruntime` is available, so `inference_framework="onnx"` is forced and
  the ONNX feature models are passed explicitly.
* **Models must not live in site-packages.** `download_models()` defaults to
  `.venv/Lib/site-packages/openwakeword/resources/models`, which is destroyed
  whenever the venv is rebuilt. They are downloaded into the project's models
  directory instead.

Measured on this machine with the ONNX model: 0.5-0.9 ms per 32 ms frame, so
unlike SmartTurn this is cheap enough to run on every frame. Peak scores were
0.994 for "Hey Jarvis", 0.000 for unrelated speech and silence, and 0.480 for a
bare "Jarvis", which sits just under the default 0.5 threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_FILES = {
    "hey_jarvis": "hey_jarvis_v0.1.onnx",
    "alexa": "alexa_v0.1.onnx",
    "hey_mycroft": "hey_mycroft_v0.1.onnx",
    "hey_rhasspy": "hey_rhasspy_v0.1.onnx",
}

MELSPEC = "melspectrogram.onnx"
EMBEDDING = "embedding_model.onnx"


@dataclass
class WakeEvent:
    score: float
    at_mono: float


def _wake_pattern(name: str) -> str:
    import re

    words = [w for w in name.split("_") if w]
    return r"^\W*" + r"[\s,.\-]*".join(re.escape(w) for w in words) + r"\s*[,.!?]*\s*"


def is_wake_phrase_only(text: str, name: str = "hey_jarvis") -> bool:
    """True when the user said nothing but the wake phrase.

    Worth its own check: the replayed audio means a bare "Hey Jarvis" arrives as
    a complete transcript, and without this it becomes a turn with no request in
    it and the model answers something invented.
    """
    import re

    if not name.split("_"):
        return False
    remainder = re.sub(_wake_pattern(name), "", text.strip(), count=1, flags=re.IGNORECASE)
    return not re.sub(r"[^\w]", "", remainder)


def strip_wake_phrase(text: str, name: str = "hey_jarvis") -> str:
    """Remove a leading wake phrase from a transcript.

    Audio captured before the detector fired is replayed into the turn, so the
    wake phrase itself usually lands at the front of the transcript. It is not
    part of the request, so it is trimmed before the LLM ever sees it.
    """
    import re

    if not [w for w in name.split("_") if w]:
        return text.strip()
    # Allows punctuation and whitespace between the words, and a trailing comma:
    # "Hey Jarvis," / "hey, jarvis" / "Hey Jarvis."
    stripped = re.sub(_wake_pattern(name), "", text, count=1, flags=re.IGNORECASE)
    return stripped.strip() or text.strip()


def model_path_for(models_dir: Path, name: str) -> Path:
    filename = MODEL_FILES.get(name, f"{name}.onnx")
    return models_dir / filename


def ensure_models(models_dir: Path, name: str = "hey_jarvis") -> Path:
    """Download the wake model and its feature models if missing."""
    import openwakeword.utils

    models_dir.mkdir(parents=True, exist_ok=True)
    target = model_path_for(models_dir, name)
    needed = [target, models_dir / MELSPEC, models_dir / EMBEDDING]
    if all(p.exists() for p in needed):
        return target

    openwakeword.utils.download_models(
        model_names=[name], target_directory=str(models_dir)
    )
    if not target.exists():
        raise FileNotFoundError(
            f"wake word model {name!r} not found at {target} after download. "
            f"Available: {sorted(p.name for p in models_dir.glob('*.onnx'))}"
        )
    return target


class WakeWordDetector:
    def __init__(
        self,
        models_dir: Path,
        *,
        name: str = "hey_jarvis",
        threshold: float = 0.5,
        refractory_ms: int = 2000,
    ):
        from openwakeword.model import Model

        self.models_dir = Path(models_dir)
        self.name = name
        self.threshold = threshold
        # After firing, ignore further detections briefly. The score stays high
        # across several frames, and without this one utterance wakes it
        # repeatedly.
        self.refractory_ms = refractory_ms

        path = ensure_models(self.models_dir, name)
        self._model = Model(
            wakeword_models=[str(path)],
            inference_framework="onnx",
            melspec_model_path=str(self.models_dir / MELSPEC),
            embedding_model_path=str(self.models_dir / EMBEDDING),
        )
        self._last_fire_mono = 0.0
        self.last_score = 0.0

    def reset(self) -> None:
        """Start listening for the phrase from scratch.

        This clears the refractory timer as well as the model state. Without
        that, a short interaction (wake, quick exchange, follow-up window
        expires) would leave the detector deaf to the user's next "hey jarvis"
        for the remainder of the refractory period.
        """
        self._model.reset()
        self._last_fire_mono = 0.0
        self.last_score = 0.0

    def process(self, frame: bytes) -> WakeEvent | None:
        """Feed one int16 frame. Returns an event when the phrase is detected."""
        samples = np.frombuffer(frame, dtype=np.int16)
        scores = self._model.predict(samples)
        score = max(scores.values()) if scores else 0.0
        self.last_score = float(score)

        if score < self.threshold:
            return None

        now = time.monotonic()
        if (now - self._last_fire_mono) * 1000.0 < self.refractory_ms:
            return None
        self._last_fire_mono = now
        # Clear internal state so the next detection starts fresh rather than
        # riding the tail of this one.
        self._model.reset()
        return WakeEvent(score=float(score), at_mono=now)

    def warm_up(self) -> float:
        """First inference pays ONNX setup."""
        started = time.monotonic()
        self.process(b"\x00\x00" * 512)
        self.reset()
        return (time.monotonic() - started) * 1000.0
