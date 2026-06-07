"""Generate a change-focused doctor brief from the health log."""

from __future__ import annotations

import json
from datetime import date, timedelta

from app import log as log_module
from app.llm import complete

BRIEF_SYSTEM = """Generate a structured doctor appointment brief. Use EXACTLY these six sections:

## New
(conditions, symptoms, or concerns that appeared for the first time in this period)

## Changed
(anything that got better or worse compared to baseline)

## Resolved
(issues that were present before but are no longer mentioned)

## Ongoing
(stable, unchanged conditions and symptoms)

## Readings
(all numeric measurements: blood pressure, glucose, weight, temperature, etc. with dates)

## Questions to raise
(patterns worth asking the doctor about — phrased as questions, never conclusions)

If a section is empty, write "Nothing to report."
Never include diagnostic conclusions. Never advise on medications."""


def generate_brief(days: int = 30, end: date | None = None) -> str:
    end = end or date.today()
    start = end - timedelta(days=days)

    all_dates = log_module.list_log_dates()
    window = [d for d in all_dates if start <= d <= end]

    if not window:
        return "No log entries found for this period."

    entries_by_date = {}
    for d in window:
        entries = log_module.get_entries(d)
        if entries:
            entries_by_date[d.isoformat()] = entries

    log_text = json.dumps(entries_by_date, indent=2, ensure_ascii=False)
    prompt = (
        f"Here are the health log entries from {start.isoformat()} to {end.isoformat()}:\n\n"
        f"{log_text}\n\n"
        "Generate the doctor brief now."
    )

    return complete(prompt, extra_system=BRIEF_SYSTEM)
