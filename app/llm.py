"""MiniCPM text model inference via transformers + ZeroGPU."""

from __future__ import annotations

import os

MINICPM_TEXT_MODEL = os.getenv("MINICPM_TEXT_MODEL", "openbmb/MiniCPM3-4B")

SYSTEM_PROMPT = """You are a health log assistant helping an elderly stroke survivor track his health.

STRICT RULES — never break these:
- You NEVER diagnose any condition.
- You NEVER advise on medication doses, changes, or interactions.
- You NEVER interpret symptoms as a specific condition.
- When a symptom or medication question arises, always say: "That's a question for your doctor or pharmacist. I'll note it for your appointment."
- You surface patterns as questions to raise, never as conclusions.

Your job: organise, summarise, and read back health log entries clearly."""

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MINICPM_TEXT_MODEL, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MINICPM_TEXT_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
