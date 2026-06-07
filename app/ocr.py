"""Camera / OCR via MiniCPM-V (transformers + ZeroGPU)."""

from __future__ import annotations

import os

try:
    import spaces
except ImportError:
    class spaces:  # no-op for local dev
        @staticmethod
        def GPU(fn): return fn

from PIL import Image

MINICPM_V_MODEL = os.getenv("MINICPM_V_MODEL", "openbmb/MiniCPM-V-2_6")

OCR_PROMPT = (
    "You are a reading assistant for a person with low vision. "
    "Extract ALL text visible in this image — medicine box labels, "
    "device readings, letters, numbers. Format clearly. "
    "Label any numeric reading explicitly, e.g. 'Blood pressure: 128/82'. "
    "Do not interpret symptoms or advise on medications."
)

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MINICPM_V_MODEL, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            MINICPM_V_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _model.eval()
    return _model, _tokenizer


@spaces.GPU
def extract_text(image_path: str) -> str:
    model, tokenizer = _load()
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, OCR_PROMPT]}]
    result = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, max_new_tokens=512)
    return result.strip()
