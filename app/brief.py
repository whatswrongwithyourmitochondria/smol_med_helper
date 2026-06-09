"""Generate a change-focused doctor brief from the health log."""

from __future__ import annotations

import re
from datetime import date, timedelta

from app import log as log_module

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

READING_RE = re.compile(
    r"\b(?:\d{2,3}\s*/\s*\d{2,3}|\d+(?:\.\d+)?\s*(?:mg/dl|mmhg|bpm|kg|lb|lbs|f|c|%|over\s+\d+))\b",
    re.IGNORECASE,
)
QUESTION_KEYWORDS = (
    "pain",
    "dizzy",
    "dizziness",
    "fall",
    "falls",
    "chest",
    "breath",
    "breathing",
    "swelling",
    "medicine",
    "medication",
    "dose",
    "tablet",
    "pill",
    "glucose",
    "blood pressure",
)


def _section(title: str, items: list[str]) -> str:
    body = "\n".join(f"- {item}" for item in items) if items else "Nothing to report."
    return f"## {title}\n{body}"


def _summarize_entry(d: date, entry: dict) -> str:
    content = " ".join(str(entry.get("content", "")).split())
    return f"{d.isoformat()}: {content}"


def _looks_like_reading(content: str) -> bool:
    return bool(READING_RE.search(content))


def _question_for(content: str) -> str:
    trimmed = " ".join(content.split())
    if len(trimmed) > 140:
        trimmed = trimmed[:137].rstrip() + "..."
    return f"Should we discuss this at the appointment: {trimmed}"


def generate_brief(days: int = 30, end: date | None = None) -> str:
    end = end or date.today()
    start = end - timedelta(days=days)

    all_dates = log_module.list_log_dates()
    window = [d for d in all_dates if start <= d <= end]

    if not window:
        return "No log entries found for this period."

    dated_entries: list[tuple[date, dict]] = []
    for d in window:
        entries = log_module.get_entries(d)
        dated_entries.extend((d, entry) for entry in entries if entry.get("content"))

    readings = [
        _summarize_entry(d, entry)
        for d, entry in dated_entries
        if entry.get("type") == "reading" or _looks_like_reading(str(entry.get("content", "")))
    ]
    narrative = [
        _summarize_entry(d, entry)
        for d, entry in dated_entries
        if entry.get("type") != "reading" and not _looks_like_reading(str(entry.get("content", "")))
    ]
    questions = []
    for _, entry in dated_entries:
        content = str(entry.get("content", ""))
        low = content.lower()
        if any(keyword in low for keyword in QUESTION_KEYWORDS):
            question = _question_for(content)
            if question not in questions:
                questions.append(question)

    # This deterministic MVP avoids diagnostic conclusions. It lists confirmed
    # log facts and leaves change/resolution interpretation for the clinician.
    return "\n\n".join(
        [
            _section("New", narrative[:8]),
            _section("Changed", []),
            _section("Resolved", []),
            _section("Ongoing", narrative[8:16]),
            _section("Readings", readings[:12]),
            _section("Questions to raise", questions[:8]),
        ]
    )
