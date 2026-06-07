"""HuggingFace Space entry point."""

from app.main import demo, CSS, THEME, CUSTOM_HEAD

demo.launch(css=CSS, theme=THEME, head=CUSTOM_HEAD)
