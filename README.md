---
title: Voice Health Companion
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Voice-First Health Companion

A voice-first health companion for a stroke survivor with low vision and one impaired hand.

**What it does:**
- Daily spoken check-ins (no typing required)
- Camera → OCR: reads medicine boxes, letters, device readings aloud
- Generates a change-focused doctor brief before appointments
- Reads the brief back by voice for confirmation

**Models:** MiniCPM-V (vision/OCR) · MiniCPM text (brief/Q&A) · Whisper (speech-in) · Kokoro (speech-out)

Built for the HuggingFace "Build Small" hackathon — OpenBMB prize track.
