"""Qwen3-4B text model inference via transformers + ZeroGPU, with Nemotron safety gate."""

from __future__ import annotations

import os
import re

MINICPM_TEXT_MODEL = os.getenv("TEXT_MODEL", "Qwen/Qwen3-4B")
SAFETY_MODEL = os.getenv("SAFETY_MODEL", "nvidia/Nemotron-3-Content-Safety")

SYSTEM_PROMPT = """You are a health log assistant helping an elderly stroke survivor track his health.

STRICT RULES - never break these:
- You NEVER diagnose any condition.
- You NEVER advise on medication doses, changes, or interactions.
- You NEVER interpret symptoms as a specific condition.
- You surface patterns as questions to raise, never as conclusions.

Your job: organise, summarise, and read back health log entries clearly."""

_model = None
_tokenizer = None
_safety_model = None
_safety_processor = None

_DEFLECTION = (
    "I'm not able to give medical advice on that. "
    "Please discuss this with your doctor or pharmacist."
)


def _load_safety():
    global _safety_model, _safety_processor
    if _safety_model is None:
        import transformers
        from transformers import AutoProcessor

        for class_name in (
            "Gemma3ForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
        ):
            safety_model_cls = getattr(transformers, class_name, None)
            if safety_model_cls is not None:
                break
        else:
            raise ImportError("No compatible Transformers class for Nemotron content safety")

        _safety_processor = AutoProcessor.from_pretrained(SAFETY_MODEL)
        _safety_model = safety_model_cls.from_pretrained(
            SAFETY_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
        _safety_model.eval()
    return _safety_model, _safety_processor


def _make_safety_messages(response: str, user_prompt: str | None = None) -> list[dict]:
    prompt = user_prompt or "The user asked a health assistant for help with health logging."
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt[:800]}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": response[:1600]}],
        },
    ]


def _is_safety_output_safe(output: str) -> bool:
    low = output.lower()
    response_match = re.search(r"response\s+safety\s*:\s*(safe|unsafe)", low)
    if response_match:
        return response_match.group(1) == "safe"
    if "unsafe" in low:
        return False
    return bool(re.search(r"\bsafe\b", low))


def safety_check(text: str, user_prompt: str | None = None) -> bool:
    """Return True if text is safe to speak, False if it must be deflected."""
    try:
        import torch

        model, processor = _load_safety()
        messages = _make_safety_messages(text, user_prompt)
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                request_categories="/no_categories",
            )
        except TypeError:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        if hasattr(inputs, "to"):
            try:
                inputs = inputs.to(model.device)
            except Exception:
                pass
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )
        answer = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return _is_safety_output_safe(answer)
    except Exception as exc:
        print(f"Safety check failed: {exc} - deflecting", flush=True)
        return False


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MINICPM_TEXT_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            MINICPM_TEXT_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
        _model.eval()
    return _model, _tokenizer


OCR_CLEAN_EXTRA = """The user has photographed a medicine box, device screen, or document.
Below is the raw text extracted from the image. Clean it up for someone who will hear it read aloud:
- Convert tables and columns into plain sentences
- Remove barcodes, batch numbers, legal boilerplate, and manufacturer addresses
- Keep drug names, dosages, instructions, warnings, and any numeric readings
- If a numeric reading is present (blood pressure, glucose, weight, temperature), start your response with "Reading: [value]"
- Do not add anything not present in the original text
- If there is nothing medically relevant, say so briefly"""


def clean_ocr(raw_text: str) -> str:
    return complete(raw_text, extra_system=OCR_CLEAN_EXTRA)


def complete(user_message: str, extra_system: str = "") -> str:
    import torch

    model, tokenizer = _load()
    system = SYSTEM_PROMPT + ("\n\n" + extra_system if extra_system else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return result if safety_check(result, user_message) else _DEFLECTION
