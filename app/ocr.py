"""Camera / OCR via MiniCPM-V 4.6 (transformers + ZeroGPU)."""

from __future__ import annotations

import os

MINICPM_V_MODEL = os.getenv("MINICPM_V_MODEL", "openbmb/MiniCPM-V-4.6")

OCR_PROMPT = (
    "You are a reading assistant for a person with low vision. "
    "Extract ALL text visible in this image — medicine box labels, "
    "device readings, letters, numbers. Format clearly. "
    "Label any numeric reading explicitly, e.g. 'Blood pressure: 128/82'. "
    "Do not interpret symptoms or advise on medications."
)

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        _processor = AutoProcessor.from_pretrained(MINICPM_V_MODEL, trust_remote_code=True)
        _model = AutoModelForImageTextToText.from_pretrained(
            MINICPM_V_MODEL,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        _model.eval()
    return _model, _processor


def extract_text(image_path: str) -> str:
    model, processor = _load()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    new_tokens = generated_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()
