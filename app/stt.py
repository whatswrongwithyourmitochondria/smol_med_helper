"""Speech-to-text via Cohere Transcribe 2B with faster-whisper fallback."""

from __future__ import annotations

import os

STT_MODEL = os.getenv("STT_MODEL", "CohereLabs/cohere-transcribe-03-2026")
WHISPER_FALLBACK = os.getenv("WHISPER_MODEL", "small")

_pipe = None
_whisper = None


def _load_cohere():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL,
            device=device,
            chunk_length_s=30,
        )
    return _pipe


def _load_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_FALLBACK, device="cpu", compute_type="int8")
    return _whisper


def _cohere_transcribe(audio_path: str) -> str:
    result = _load_cohere()(audio_path, return_timestamps=False)
    return result["text"].strip()


def _whisper_transcribe(audio_path: str) -> str:
    model = _load_whisper()
    segments, _ = model.transcribe(audio_path, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe(audio_path: str) -> str:
    try:
        return _cohere_transcribe(audio_path)
    except Exception as exc:
        print(f"Cohere Transcribe failed: {exc} — falling back to Whisper", flush=True)
        return _whisper_transcribe(audio_path)
