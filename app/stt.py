"""Speech-to-text via faster-whisper (CPU — fast enough for whisper-small)."""

from __future__ import annotations

import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> str:
    model = _load()
    segments, _ = model.transcribe(audio_path, language="en")
    return " ".join(s.text.strip() for s in segments).strip()
