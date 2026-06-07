# Voice-First Health Companion — HuggingFace "Build Small" Hackathon

**Submission deadline: 15 June 2026**

## What this is

A voice-first health companion designed for a stroke survivor with low vision and one impaired hand. He cannot type or read small text; speech is intact. The app lets him:

1. Speak daily check-ins (no keyboard)
2. Point the camera at medicine boxes / letters / device readings; the model OCRs and explains them, logging readings automatically
3. Generate a *change-focused* doctor brief before appointments (New / Changed / Resolved / Ongoing / Readings / Questions to raise — a diff, not a transcript dump)
4. Hear the brief read back; confirm or correct by voice before anything is shared

## Stack (all ≤32B, llama.cpp backend — no cloud LLM API calls)

| Role | Model |
|------|-------|
| Camera / OCR | MiniCPM-V 4.6 (OpenBMB) |
| Text / brief / Q&A | small MiniCPM text model (OpenBMB) |
| Speech-in | Whisper-small |
| Speech-out | VoxCPM2 |

Using MiniCPM models makes us eligible for the **OpenBMB $10k prize**.

## Safety rules — NON-NEGOTIABLE

- The model **never diagnoses** and **never advises on doses, interactions, or symptoms**
- Any symptom pattern → surface as a question for the doctor, never a conclusion
- Medication information → redirect to pharmacist or doctor
- Deflection phrases must appear in every relevant response
- LoRA fine-tune specifically locks this behaviour

## Module layout

```
app/
  main.py        — Gradio UI, tab layout, event wiring
  stt.py         — Whisper speech → text
  ocr.py         — MiniCPM-V camera / OCR / reading extraction
  llm.py         — MiniCPM text inference (brief gen, Q&A, summarise)
  tts.py         — VoxCPM2 text → speech
  log.py         — Health log CRUD (JSON flat file per day)
  brief.py       — Diff logic: compares last N days, produces structured brief
data/
  logs/          — One JSON file per date: {date, entries: [{type, content, ts}]}
finetune/
  generate_data.py   — Synthetic training examples (brief format + deflection)
  train_lora.py      — LoRA training on Modal
  eval.py            — Compare base vs LoRA on held-out examples
```

## Gradio UI tabs

1. **Check-in** — mic button → STT → log entry → TTS confirmation
2. **Camera / Read** — camera snapshot → OCR → spoken explanation → log reading
3. **Doctor Brief** — date-range picker → generate brief → TTS playback → voice confirm/correct
4. **History** — read-only log viewer (large text)

## Hackathon timeline

| Dates | Goal |
|-------|------|
| Jun 7–8 (this weekend) | De-risk full loop: STT → log → brief → TTS working end-to-end |
| Jun 9–11 (midweek) | Diff logic polish, accessible UI (large font, high contrast), LoRA data gen + train |
| Jun 13–15 (final weekend) | Polish, film him using it, demo video + social post, submit |

## Prize targets

- **OpenBMB $10k** — MiniCPM-V + MiniCPM text models used
- **Off the Grid** — local inference, no cloud API
- **Llama Champion** — (check eligibility, may swap one model)
- **Well-Tuned** — publish LoRA + training code + eval results
- **Field Notes** — short write-up on the accessibility use case

## Run commands

```bash
# Install dependencies
pip install -r requirements.txt

# Launch app (dev)
python app/main.py

# Run end-to-end smoke test
python -m pytest tests/ -v

# Generate synthetic fine-tune data
python finetune/generate_data.py --count 200 --out finetune/data/

# Train LoRA (run on Modal)
python finetune/train_lora.py --base minicpm-text --data finetune/data/ --out finetune/output/
```

## Key constraints

- All model inference via llama.cpp (llama-cpp-python) — no transformers runtime for main models
- Gradio Space runs on HF hardware; models loaded from HF Hub or local cache
- Log files are local JSON — no database, no external storage
- UI must work with single-hand operation: large tap targets, voice-first, minimal typing
- Brief output format is strictly structured (headers enforced by LoRA + prompt template)
