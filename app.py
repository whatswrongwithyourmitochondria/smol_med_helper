"""HuggingFace Space entry point."""

from app.main import CSS, CUSTOM_HEAD, THEME, demo

demo.launch(css=CSS, theme=THEME, head=CUSTOM_HEAD)
