"""Camera / OCR via MiniCPM-V 4.6 (transformers + ZeroGPU)."""

from __future__ import annotations

import os

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **_):
            return fn if fn is not None else lambda f: f

from PIL import Image

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


@spaces.GPU(duration=90)
def extract_text(image_path: str) -> str:
    model, processor = _load()
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]
    downsample_mode = "16x"
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode=downsample_mode,
        max_slice_nums=36,
    ).to(model.device)
    generated_ids = model.generate(
        **inputs, downsample_mode=downsample_mode, max_new_tokens=512
    )
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
