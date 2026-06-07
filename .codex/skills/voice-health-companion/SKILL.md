---
name: voice-health-companion
description: Use when working on this Build Small Hackathon Backyard AI voice-first health companion, including accessible Gradio UI, local small-model integration, OCR/readings logging, change-focused doctor briefs, medical safety deflection, LoRA evaluation, and submission prep.
---

# Voice Health Companion

## Start Here

Read `AGENTS.md` and `docs/implementation-plan.md` before making project decisions. Keep changes aligned with the hackathon deadline, the Backyard AI judging criteria, the <=32B model cap, and the local-first/no cloud LLM API constraint.

## Product Shape

The target user has low vision and impaired typing ability after a stroke. Optimize for voice in, camera in, large readable output, voice playback, and confirmation before sharing.

Primary loop:

1. Capture spoken check-ins.
2. OCR/explain documents, medicine boxes, or readings from camera input.
3. Confirm extracted facts.
4. Store structured events locally.
5. Generate a change-focused doctor brief.
6. Read it aloud and accept voice corrections.

## Safety Rules

Never implement behavior that diagnoses, recommends medication dose changes, evaluates drug interactions, or advises symptom treatment. For those requests, deflect and offer to record the concern as a clinician question.

Use this brief deflection when needed:

> I cannot safely advise on diagnosis, medication changes, interactions, or urgent symptoms. I can help record this clearly and turn it into a question for your doctor. If this might be urgent, contact a clinician or emergency service now.

Treat model output as draft until confirmed by the user.

## Implementation Preferences

- Use Gradio as the app shell.
- Keep model adapters isolated and provide mock fallbacks.
- Store structured data before summarizing it.
- Keep doctor briefs sectioned as `New`, `Changed`, `Resolved`, `Ongoing`, `Readings`, and `Questions`.
- Prefer deterministic diff logic before asking a text model to polish wording.
- Add tests for safety gates, brief formatting, and diff behavior.
- Do not commit real personal health information.

## Accessibility Preferences

- Large text and large controls by default.
- High contrast, restrained layout, and few steps per task.
- Voice alternatives for confirmation and correction.
- Avoid dense transcript dumps; show concise confirmed summaries.

## Model Notes

Planned models are MiniCPM-V 4.6 for camera/OCR, a small MiniCPM text model for brief generation/Q&A, Whisper-small for speech-to-text, and VoxCPM2 or a local fallback for text-to-speech. Verify exact model, quantization, license, parameter count, and runtime before locking the submission stack.
