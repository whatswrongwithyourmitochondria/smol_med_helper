"""Health log: one JSON file per date in data/logs/YYYY-MM-DD.json"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Literal

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EntryType = Literal["checkin", "reading", "ocr", "note"]


def _log_path(d: date) -> Path:
    return LOG_DIR / f"{d.isoformat()}.json"


def _load(d: date) -> dict:
    p = _log_path(d)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"date": d.isoformat(), "entries": []}


def _save(d: date, data: dict) -> None:
    _log_path(d).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_entry(
    content: str,
    entry_type: EntryType = "checkin",
    d: date | None = None,
    photo: str | None = None,
) -> dict:
    d = d or date.today()
    data = _load(d)
    entry: dict = {"type": entry_type, "content": content, "ts": datetime.utcnow().isoformat()}
    if photo:
        entry["photo"] = photo
    data["entries"].append(entry)
    _save(d, data)
    return entry


def get_entries(d: date) -> list[dict]:
    return _load(d)["entries"]


def list_log_dates() -> list[date]:
    dates = []
    for p in sorted(LOG_DIR.glob("*.json")):
        try:
            dates.append(date.fromisoformat(p.stem))
        except ValueError:
            pass
    return dates
