"""Text-to-speech via Kokoro-82M (ZeroGPU) with pyttsx3 fallback for local dev."""

from __future__ import annotations

import io
import os

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **_):
            return fn if fn is not None else lambda f: f

_USE_KOKORO = os.getenv("USE_KOKORO", "1") == "1"
_pipeline = None


def _load_kokoro():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a")  # American English
    return _pipeline


@spaces.GPU(duration=30)
def _kokoro_speak(text: str) -> bytes:
    import numpy as np
    import soundfile as sf

    pipeline = _load_kokoro()
    chunks = [audio for _, _, audio in pipeline(text, voice="af_heart", speed=0.9)]
    full = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, full, 24000, format="WAV")
    return buf.getvalue()


def _pyttsx3_speak(text: str) -> bytes:
    import tempfile, os as _os
    import pyttsx3
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    engine.save_to_file(text, tmp)
    engine.runAndWait()
    with open(tmp, "rb") as f:
        data = f.read()
    _os.unlink(tmp)
    return data


def speak(text: str) -> bytes:
    """Return WAV bytes. Uses Kokoro on HF Space, pyttsx3 locally."""
    if _USE_KOKORO:
        try:
            return _kokoro_speak(text)
        except Exception:
            pass
    return _pyttsx3_speak(text)
