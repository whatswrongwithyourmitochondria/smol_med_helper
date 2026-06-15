"""HuggingFace Space entry point."""

import os

from app.main import CSS, CUSTOM_HEAD, THEME, demo

_BASE = os.path.dirname(os.path.abspath(__file__))
_ALLOWED = [os.path.join(_BASE, "data"), os.path.join(_BASE, "assets")]

demo.launch(css=CSS, theme=THEME, head=CUSTOM_HEAD, allowed_paths=_ALLOWED)
