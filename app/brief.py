"""Generate a change-focused doctor brief from the health log."""

from __future__ import annotations

import re
from datetime import date, timedelta

from app import log as log_module

_COMPRESS_SYSTEM = """Condense this health log brief for a doctor's appointment.
Keep EXACTLY these six section headers: ## New, ## Changed, ## Resolved, ## Ongoing, ## Readings, ## Questions to raise
- Deduplicate repeated records, including repeated text attached to different photos
- Merge repeated mentions; keep all dates and numbers that add new information
- Keep any "(see image)" marker exactly where it appears — never drop it
- Plain spoken English — this will be read aloud to the patient
- If a section has nothing, write exactly: Nothing to report.
- NEVER diagnose conditions or advise on medications"""


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
SECTION_TITLES = ("New", "Changed", "Resolved", "Ongoing", "Readings", "Questions to raise")
SECTION_ALIASES = {
    "new": "New",
    "changed": "Changed",
    "resolved": "Resolved",
    "ongoing": "Ongoing",
    "readings": "Readings",
    "questions": "Questions to raise",
    "questions to raise": "Questions to raise",
}
NOTHING = "Nothing to report."


def _section(title: str, items: list[str]) -> str:
    body = "\n".join(f"- {item}" for item in items) if items else NOTHING
    return f"## {title}\n{body}"


def _entry_key(d: date, entry: dict) -> tuple[str, str, str]:
    content = re.sub(r"\s+", " ", str(entry.get("content", "")).strip()).lower()
    entry_type = str(entry.get("type", ""))
    if entry_type != "reading" and _looks_like_reading(content):
        entry_type = "reading"
    return d.isoformat(), entry_type, content


def _dedupe_entries(dated_entries: list[tuple[date, dict]]) -> list[tuple[date, dict]]:
    seen = set()
    deduped = []
    for d, entry in dated_entries:
        key = _entry_key(d, entry)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((d, entry))
    return deduped


def _canonical_section_title(line: str) -> str | None:
    title = line.strip().lstrip("#").strip().rstrip(":")
    return SECTION_ALIASES.get(title.lower())


def _normalise_item(line: str) -> str:
    item = line.strip()
    while item.startswith(("-", "*", "\u2022")):
        item = item[1:].strip()
    return " ".join(item.split())


def _item_key(item: str) -> str:
    item = item.replace("(see image)", "").strip()
    return re.sub(r"\s+", " ", item).lower()


def _post_process_brief(brief: str) -> str:
    sections: dict[str, list[str]] = {title: [] for title in SECTION_TITLES}
    current: str | None = None

    for raw_line in brief.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        title = _canonical_section_title(line)
        if title:
            current = title
            continue
        if current is None:
            continue
        item = _normalise_item(line)
        if not item or item == NOTHING:
            continue
        sections[current].append(item)

    rendered = []
    for title in SECTION_TITLES:
        seen = set()
        items = []
        for item in sections[title]:
            key = _item_key(item)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
        rendered.append(_section(title, items))
    return "\n\n".join(rendered)


def _summarize_entry(d: date, entry: dict) -> str:
    content = " ".join(str(entry.get("content", "")).split())
    result = f"{d.isoformat()}: {content}"
    if entry.get("photo"):
        result += " (see image)"
    return result


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
    dated_entries = _dedupe_entries(dated_entries)

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

    raw = "\n\n".join(
        [
            _section("New", narrative),
            _section("Changed", []),
            _section("Resolved", []),
            _section("Ongoing", []),
            _section("Readings", readings),
            _section("Questions to raise", questions),
        ]
    )
    brief = _post_process_brief(_compress_brief(raw))

    # Append a Photos section with clickable links — built directly (not via the
    # LLM) so the URLs are never mangled by the compression pass.
    photos = [(d, entry["photo"]) for d, entry in dated_entries if entry.get("photo")]
    if photos:
        seen_photos = set()
        deduped_photos = []
        for d, path in photos:
            if path in seen_photos:
                continue
            seen_photos.add(path)
            deduped_photos.append((d, path))
        links = "\n".join(
            f"- [{d.isoformat()} — view photo](/gradio_api/file={path})"
            for d, path in deduped_photos
        )
        brief = f"{brief}\n\n## Photos\n{links}"

    return brief


def _compress_brief(raw: str) -> str:
    try:
        from app.llm import postprocess_brief

        return postprocess_brief(raw, _COMPRESS_SYSTEM)
    except Exception as exc:
        print(f"[brief] LLM compression failed: {exc}", flush=True)
        return raw
