"""Text-to-speech via VoxCPM2 (OpenBMB) with Kokoro fallback."""

from __future__ import annotations

import io
import os

_USE_VOXCPM = os.getenv("USE_VOXCPM", "0") == "1"
_vox_model = None
_kokoro_pipeline = None


def _load_voxcpm():
    global _vox_model
    if _vox_model is None:
        from voxcpm import VoxCPM
        _vox_model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    return _vox_model


def _voxcpm_speak(text: str) -> bytes:
    import soundfile as sf
    model = _load_voxcpm()
    wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
    buf = io.BytesIO()
    sf.write(buf, wav, model.tts_model.sample_rate, format="WAV")
    return buf.getvalue()


def _load_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def _kokoro_speak(text: str) -> bytes:
    import numpy as np
    import soundfile as sf
    pipeline = _load_kokoro()
    chunks = [audio for _, _, audio in pipeline(text, voice="af_heart", speed=0.9)]
    full = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, full, 24000, format="WAV")
    return buf.getvalue()


def _silent_wav() -> bytes:
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(24000 // 4, dtype=np.float32), 24000, format="WAV")
    return buf.getvalue()


def speak(text: str) -> bytes:
    """Return WAV bytes. Uses VoxCPM2 on HF Space, Kokoro as fallback."""
    if _USE_VOXCPM:
        try:
            return _voxcpm_speak(text)
        except Exception as exc:
            print(f"VoxCPM2 TTS failed: {exc} — falling back to Kokoro", flush=True)
    try:
        return _kokoro_speak(text)
    except Exception as exc:
        print(f"Kokoro TTS fallback failed: {exc}", flush=True)
        return _silent_wav()
