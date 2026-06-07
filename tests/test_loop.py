"""End-to-end smoke tests — run with: pytest tests/test_loop.py -v"""

import json
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import log as log_module


def test_log_add_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.setattr(log_module, "LOG_DIR", tmp_path)
    entry = log_module.add_entry("Blood pressure was 128 over 82", entry_type="reading", d=date(2026, 6, 7))
    assert entry["type"] == "reading"
    entries = log_module.get_entries(date(2026, 6, 7))
    assert len(entries) == 1
    assert "128" in entries[0]["content"]


def test_log_lists_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(log_module, "LOG_DIR", tmp_path)
    log_module.add_entry("Morning check-in", d=date(2026, 6, 6))
    log_module.add_entry("Evening check-in", d=date(2026, 6, 7))
    dates = log_module.list_log_dates()
    assert date(2026, 6, 6) in dates
    assert date(2026, 6, 7) in dates


def test_brief_no_entries_returns_message(tmp_path, monkeypatch):
    monkeypatch.setattr(log_module, "LOG_DIR", tmp_path)
    from app.brief import generate_brief
    result = generate_brief(days=7, end=date(2026, 6, 7))
    assert "No log entries" in result
