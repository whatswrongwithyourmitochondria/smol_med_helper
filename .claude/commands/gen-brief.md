# /gen-brief

Generate a test doctor brief from the current log files and display it.

Steps:
1. List available log dates in `data/logs/`
2. Call `brief.py` `generate_brief()` over the most recent available window (up to 30 days)
3. Print the brief to console with section headers clearly visible
4. Run the safety check: confirm no diagnostic conclusions present
5. Optionally pass through TTS and report audio duration

Use this to manually inspect brief quality and section completeness before submitting.
