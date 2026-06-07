# /test-loop

Smoke-test the full pipeline end-to-end using a synthetic audio + image fixture.

Steps:
1. Run STT on `tests/fixtures/checkin.wav` — confirm transcript is returned
2. Pass transcript through log.py `add_entry()` — confirm JSON written to `data/logs/`
3. Run OCR on `tests/fixtures/med_box.jpg` — confirm reading extracted
4. Call `brief.py` `generate_brief()` over the last 3 days of logs — confirm all six sections present (New / Changed / Resolved / Ongoing / Readings / Questions)
5. Pass brief text through TTS — confirm audio bytes returned
6. Print pass/fail for each step with timing

Run with: `python -m pytest tests/test_loop.py -v`
