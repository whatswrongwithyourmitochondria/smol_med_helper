"""Camera / OCR via MiniCPM-V 4.6 (transformers + ZeroGPU)."""

from __future__ import annotations

import os

MINICPM_V_MODEL = os.getenv("MINICPM_V_MODEL", "openbmb/MiniCPM-V-4.6")

OCR_PROMPT = (
    "You are a reading assistant for a person with low vision. "
    "Extract only the text that is visibly printed or handwritten in this image. "
    "Preserve short line breaks when helpful. "
    "Do not describe colors, background, alignment, layout, or formatting. "
    "Do not say whether numeric readings are present or absent. "
    "Do not add introductions such as 'The visible text is'. "
    "If no text is readable, return exactly: No readable text found. "
    "Do not interpret symptoms or advise on medications."
)

_model = None
_processor = None


def _clean_ocr_text(text: str) -> str:
    stripped = text.strip()
    prefixes = (
        "The visible text in the image is:",
        "Visible text in the image:",
        "The text in the image is:",
        "Extracted text:",
    )
    for prefix in prefixes:
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix):].strip()
            break

    dropped_prefixes = (
        "There are no numeric readings",
        "The text is presented",
        "The format is",
        "The background is",
        "The layout is",
    )
    lines = []
    for line in stripped.splitlines():
        clean = line.strip()
        if any(clean.lower().startswith(prefix.lower()) for prefix in dropped_prefixes):
            continue
        lines.append(line.rstrip())

    cleaned = "\n".join(lines).strip()
    return cleaned or "No readable text found."


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
    return _clean_ocr_text(processor.decode(new_tokens, skip_special_tokens=True))
