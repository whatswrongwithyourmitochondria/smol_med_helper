"""Qwen3-4B text model inference via transformers + ZeroGPU, with Nemotron safety gate."""

from __future__ import annotations

import os

MINICPM_TEXT_MODEL = os.getenv("TEXT_MODEL", "Qwen/Qwen3-4B")
SAFETY_MODEL = os.getenv("SAFETY_MODEL", "nvidia/Nemotron-3-Nano-4B-Instruct")

SYSTEM_PROMPT = """You are a health log assistant helping an elderly stroke survivor track his health.

STRICT RULES — never break these:
- You NEVER diagnose any condition.
- You NEVER advise on medication doses, changes, or interactions.
- You NEVER interpret symptoms as a specific condition.
- You surface patterns as questions to raise, never as conclusions.

Your job: organise, summarise, and read back health log entries clearly."""

_model = None
_tokenizer = None
_safety_model = None
_safety_tokenizer = None

_DEFLECTION = (
    "I'm not able to give medical advice on that. "
    "Please discuss this with your doctor or pharmacist."
)

_SAFETY_PROMPT = (
    "Does the following text contain a medical diagnosis, symptom interpretation "
    "as a specific condition, medication dosing advice, or drug interaction advice? "
    "Answer with only YES or NO.\n\nText:\n{text}"
)


def _load_safety():
    global _safety_model, _safety_tokenizer
    if _safety_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _safety_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
        _safety_model = AutoModelForCausalLM.from_pretrained(
            SAFETY_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
        _safety_model.eval()
    return _safety_model, _safety_tokenizer


def safety_check(text: str) -> bool:
    """Return True if text is safe to speak, False if it must be deflected."""
    try:
        import torch
        model, tokenizer = _load_safety()
        prompt = _SAFETY_PROMPT.format(text=text[:600])
        messages = [{"role": "user", "content": prompt}]
        inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inp, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()
        return not answer.startswith("YES")
    except Exception as exc:
        print(f"Safety check failed: {exc} — allowing through", flush=True)
        return True


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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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
    return result if safety_check(result) else _DEFLECTION
