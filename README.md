---
title: Voice Health Companion
emoji: 🎙️
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
license: apache-2.0
tags:
  - build-small-hackathon
  - health
  - voice
  - accessibility
  - openbmb
tracks:
  - backyard
badges:
  - offbrand
  - tiny
  - demo
---

# Voice-First Health Companion

A voice-first health companion for a stroke survivor with low vision and one impaired hand.
All models are ≤4B parameters and run entirely on-device — no cloud LLM calls.

**What it does:**
- Daily spoken check-ins (no typing required)
- Camera → OCR: reads medicine boxes, letters, device readings aloud
- Generates a change-focused doctor brief before appointments
- Reads the brief back by voice for confirmation

**Models (all ≤4B — qualifies for Tiny Titan):**
| Role | Model | Params |
|------|-------|--------|
| Vision / OCR | MiniCPM-V 4.6 (OpenBMB) | ~1.3B |
| Text / brief | MiniCPM3-4B (OpenBMB) | 4B |
| Speech-in | Whisper small | ~244M |
| Speech-out | Kokoro-82M | 82M |

**Safety:** The model never diagnoses and never advises on doses or interactions.
All symptom patterns surface as questions for the doctor.

Built for the HuggingFace "Build Small" hackathon — Backyard AI track · OpenBMB prize track.
