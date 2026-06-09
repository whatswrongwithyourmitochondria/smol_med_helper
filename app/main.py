"""Gradio UI — voice-first health companion."""

from __future__ import annotations

import os
import tempfile
import traceback

os.environ.setdefault("GRADIO_SSR_MODE", "False")

import gradio as gr

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **_):
            return fn if fn is not None else lambda f: f

from app import log as log_module
from app.brief import generate_brief
from app.stt import transcribe
from app.tts import speak

# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_checkin(audio_path: str | None) -> str:
    if not audio_path:
        return ""
    transcript = transcribe(audio_path)
    log_module.add_entry(transcript, entry_type="checkin")
    return transcript


def _pil_from_image_value(value, image_module):
    if value is None:
        return None
    if isinstance(value, image_module.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        return image_module.open(value).convert("RGB")
    if hasattr(value, "astype"):
        return image_module.fromarray(value.astype("uint8")).convert("RGB")
    raise ValueError(f"Unsupported image input type: {type(value)!r}")


def _layer_bbox(layer, padding: int = 12):
    import numpy as np

    arr = np.asarray(layer)
    if arr.ndim == 2:
        mask = arr > 8
    elif arr.ndim == 3 and arr.shape[-1] >= 4:
        mask = arr[..., 3] > 8
    elif arr.ndim == 3:
        mask = arr.max(axis=-1) > 8
    else:
        return None
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = int(xs.max()) + padding + 1
    bottom = int(ys.max()) + padding + 1
    return left, top, right, bottom


def _image_from_editor_value(value, image_module):
    if isinstance(value, dict):
        background = _pil_from_image_value(value.get("background"), image_module)
        composite = _pil_from_image_value(value.get("composite"), image_module)
        base = background or composite
        if base is None:
            return None

        for layer in value.get("layers") or []:
            bbox = _layer_bbox(layer)
            if bbox is not None:
                left, top, right, bottom = bbox
                right = min(right, base.width)
                bottom = min(bottom, base.height)
                return base.crop((left, top, right, bottom))
        return composite or base
    return _pil_from_image_value(value, image_module)


def _editor_value_from_image(image):
    if image is None:
        return gr.update()
    return {"background": image, "layers": [], "composite": image}


def show_camera_capture():
    return gr.update(visible=True, value=None), gr.update(visible=False)


def load_camera_capture(image):
    if image is None:
        return gr.update(), gr.update()
    return gr.update(value=_editor_value_from_image(image), visible=True), gr.update(
        visible=False, value=None
    )


def load_uploaded_photo(file_path):
    if file_path is None:
        return gr.update(), gr.update()
    value = {"background": file_path, "layers": [], "composite": None}
    return gr.update(value=value, visible=True), gr.update(visible=False, value=None)


@spaces.GPU(duration=120)
def handle_ocr(image_editor_value) -> tuple[str, str]:
    if image_editor_value is None:
        return "", None
    import os as _os
    from PIL import Image as PILImage

    img = _image_from_editor_value(image_editor_value, PILImage)
    if img is None:
        return "", None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name
    img.save(img_path)

    from app.ocr import extract_text
    try:
        result = extract_text(img_path)
    except Exception:
        traceback.print_exc()
        result = (
            "I could not read this image. Please try another photo with clearer "
            "lighting and the text fully in frame."
        )
    finally:
        _os.unlink(img_path)

    log_module.add_entry(result, entry_type="ocr")
    audio_bytes = speak(result)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return result, tmp.name


def _speech_text_from_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("##"):
            heading = stripped.lstrip("#").strip()
            if heading:
                lines.append(f"{heading}.")
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        lines.append(stripped.replace("#", "").strip())
    return "\n".join(line for line in lines if line).strip()


@spaces.GPU(duration=120)
def handle_brief(days: int) -> tuple:
    brief_text = generate_brief(days=int(days))
    audio_bytes = speak(_speech_text_from_markdown(brief_text))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return brief_text, gr.update(visible=True, value=tmp.name)


def handle_history() -> str:
    dates = log_module.list_log_dates()
    if not dates:
        return "_No entries yet. Start with a daily check-in._"
    type_icon = {"checkin": "🎙️", "reading": "📊", "ocr": "📷", "note": "📝"}
    lines = []
    for d in reversed(dates[-7:]):
        lines.append(f"### {d.isoformat()}")
        for e in log_module.get_entries(d):
            icon = type_icon.get(e["type"], "•")
            lines.append(f"{icon}&nbsp; {e['content']}")
        lines.append("")
    return "\n".join(lines)


# ── Design ────────────────────────────────────────────────────────────────────

CSS = """
/* @import not allowed in constructable stylesheets — fonts loaded via head param instead */

/* ── Tokens + Gradio primary override ── */
:root {
    --bg:       #070b16;
    --surface:  #0d1526;
    --surface2: #111e35;
    --border:   #192e50;
    --text:     #dde6f5;
    --muted:    #4d6a8a;
    --cyan:     #00d2ff;
    --blue:     #3a7bd5;
    --purple:   #c471ed;
    --grad:     linear-gradient(270deg, #00d2ff, #3a7bd5, #c471ed, #00d2ff);

    /* Override Gradio's orange primary — affects tab underline, slider, focus rings */
    --primary-50:  #ecfeff;
    --primary-100: #cffafe;
    --primary-200: #a5f3fc;
    --primary-300: #67e8f9;
    --primary-400: #22d3ee;
    --primary-500: #00d2ff;
    --primary-600: #0891b2;
    --primary-700: #0e7490;
    --primary-800: #155e75;
    --primary-900: #164e63;
    --primary-950: #083344;
    --color-accent: #00d2ff;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }
.contain, .wrap, .svelte-1gfkn6j { background: transparent !important; }

/* ── Hero header ── */
.app-header {
    text-align: center;
    padding: 2.2rem 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.4rem;
}
.app-title {
    font-family: 'Tomorrow', monospace;
    font-size: clamp(1.9rem, 5vw, 2.8rem);
    font-weight: 700;
    letter-spacing: 3px;
    background: linear-gradient(95deg, #00d2ff 0%, #3a7bd5 48%, #c471ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
    line-height: 1.1;
}
.app-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--muted) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0;
}

/* ── Tabs — element-prefixed so Gradio's CSS parser keeps them ── */
div.tab-container {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 7px !important;
    margin-bottom: 14px !important;
    gap: 4px !important;
}
button.svelte-11gaq1 {
    font-family: 'Tomorrow', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border-radius: 12px !important;
    padding: 13px 20px !important;
    min-height: 46px !important;
    border: none !important;
    border-bottom: none !important;
    background: transparent !important;
    color: var(--muted) !important;
    transition: all 0.22s ease !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    line-height: 1 !important;
    vertical-align: middle !important;
}
/* Nudge emoji glyphs to sit on the same optical baseline as the label text */
button.svelte-11gaq1 > * { vertical-align: middle !important; }
button.svelte-11gaq1:hover:not(.selected) {
    background: var(--surface2) !important;
    color: var(--text) !important;
}
button.selected.svelte-11gaq1 {
    background: transparent !important;
    color: #f97316 !important;
    border: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    outline: none !important;
    text-shadow: none !important;
}
/* Kill the orange underline — (0,2,2) beats Gradio's (0,2,1) and has !important */
button.selected.svelte-11gaq1::after {
    background-color: transparent !important;
    background: transparent !important;
    display: none !important;
    height: 0 !important;
}

/* ── Cards / blocks ── */
.block, .gr-group, .panel, .form {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
}

/* ── Section hints ── */
.hint {
    text-align: center !important;
}
.hint p {
    text-align: center !important;
    color: var(--muted) !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
    margin: 0 0 0.5rem !important;
}

/* ── Animated primary button ── */
@keyframes aurora {
    0%   { background-position:   0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position:   0% 50%; }
}
@keyframes opalhue {
    0%   { filter: hue-rotate(  0deg) brightness(1.00) saturate(1.10); }
    25%  { filter: hue-rotate( 25deg) brightness(1.12) saturate(1.35); }
    50%  { filter: hue-rotate(-15deg) brightness(1.18) saturate(1.40); }
    75%  { filter: hue-rotate( 20deg) brightness(1.10) saturate(1.30); }
    100% { filter: hue-rotate(  0deg) brightness(1.00) saturate(1.10); }
}

button.primary {
    font-family: 'Tomorrow', monospace !important;
    font-weight: 700 !important;
    font-size: 1.08rem !important;
    letter-spacing: 0.8px !important;
    min-height: 62px !important;
    border-radius: 14px !important;
    border: none !important;
    background: linear-gradient(270deg, #92400e, #c2410c, #ea580c, #f59e0b, #fbbf24, #f59e0b, #ea580c, #c2410c, #92400e) !important;
    background-size: 300% 300% !important;
    animation: aurora 9s ease infinite !important;
    color: #fff !important;
    box-shadow: 0 4px 28px rgba(234,88,12,0.20), 0 1px 0 rgba(255,255,255,0.10) inset !important;
    cursor: pointer !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}
button.primary:hover {
    transform: translateY(-3px) !important;
    animation: aurora 9s ease infinite, opalhue 1.6s linear infinite !important;
    box-shadow: 0 12px 40px rgba(234,88,12,0.35),
                0 0 28px rgba(200,255,240,0.10),
                0 1px 0 rgba(255,255,255,0.15) inset !important;
}
button.primary:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 10px rgba(234,88,12,0.14) !important;
}

.source-actions {
    justify-content: center !important;
    gap: 12px !important;
    margin: 0 auto !important;
}
.source-actions button {
    min-height: 56px !important;
    min-width: min(260px, 45vw) !important;
}
.photo-start-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 30px 24px !important;
    margin-bottom: 14px !important;
}

/* ── Secondary button ── */
button.secondary {
    font-family: 'Tomorrow', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    min-height: 50px !important;
    border-radius: 12px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    border-color: rgba(0,210,255,0.4) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 16px rgba(0,210,255,0.1) !important;
}

/* ── Text inputs ── */
textarea, input[type=text], input[type=number], input[type=search] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
    color: var(--text) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input:focus {
    border-color: rgba(0,210,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,210,255,0.1) !important;
    outline: none !important;
}

/* ── Labels ── */
label > span, .label-wrap > span {
    font-family: 'Tomorrow', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── History markdown ── */
.prose p, .prose li { color: var(--text) !important; font-size: 1.05rem !important; line-height: 1.7 !important; }
.prose h3 {
    font-family: 'Tomorrow', monospace !important;
    color: var(--cyan) !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 5px !important;
    margin: 1.4rem 0 0.5rem !important;
}

/* ── Slider ── */
input[type=range] { accent-color: var(--cyan) !important; height: 6px !important; }

/* ── Audio ── */
.waveform-container, .waveform-container * { background: var(--bg) !important; }

/* ── Doctor Brief — compact scrollable box ── */
.brief-box textarea {
    min-height: 180px !important;
    max-height: 260px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }
"""

HEADER_HTML = """
<div class="app-header">
    <h1 class="app-title">Health Companion</h1>
    <p class="app-sub">Voice-first &nbsp;·&nbsp; Camera-assisted &nbsp;·&nbsp; Appointment-ready</p>
</div>
"""

# Fonts + accent-colour fix injected via <head> so @import and <script> both work.
CUSTOM_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Tomorrow:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script>
(function () {
    var CYAN = '#00d2ff';
    var PROPS = ['--color-accent', '--primary-500', '--primary-400', '--primary-600'];
    function fix() {
        PROPS.forEach(function (p) {
            document.documentElement.style.setProperty(p, CYAN);
        });
    }
    fix();
    new MutationObserver(fix).observe(
        document.documentElement,
        { attributes: true, attributeFilter: ['style'] }
    );
    [50, 200, 600, 1500].forEach(function (t) { setTimeout(fix, t); });
}());
</script>
"""

# ── UI ────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Tomorrow"),
    font_mono=gr.themes.GoogleFont("Tomorrow"),
)

with gr.Blocks(title="Health Companion") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── 🎙️ Check-in ───────────────────────────────────────────────────────
        with gr.Tab("🎙️  Check-in"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Press <strong style="color:#7a92aa;">Record</strong> and speak your check-in, '
                'then <strong style="color:#7a92aa;">Stop</strong>.</p>'
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'The app logs it and shows what it heard.</p>'
            )
            audio_in = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️  Your voice",
            )
            checkin_btn = gr.Button("⬆️  Log Check-in", variant="primary")
            transcript_out = gr.Textbox(
                label="📝  What I heard",
                lines=4,
                interactive=False,
            )
            checkin_btn.click(
                handle_checkin,
                inputs=audio_in,
                outputs=transcript_out,
            )

        # ── 📷 Camera & Read ──────────────────────────────────────────────────
        with gr.Tab("📷  Camera & Read"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Point the camera at a <strong style="color:#7a92aa;">medicine box</strong>, '
                'device screen, or letter.</p>'
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'The model reads the text aloud and logs any numeric readings.</p>'
            )
            with gr.Group(elem_classes=["photo-start-panel"]):
                with gr.Row(elem_classes=["source-actions"]):
                    take_photo_btn = gr.Button("📷  Take a photo", variant="secondary")
                    import_photo_btn = gr.UploadButton(
                        "🖼️  Import a photo",
                        file_types=["image"],
                        type="filepath",
                        variant="secondary",
                    )
            camera_capture = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="📷  Take a photo",
                height=260,
                visible=False,
            )
            camera_in = gr.ImageEditor(
                sources=(),
                type="numpy",
                image_mode="RGBA",
                transforms=("crop", "resize"),
                brush=gr.Brush(
                    default_size=40,
                    colors=["#00d2ff"],
                    default_color="#00d2ff",
                    color_mode="fixed",
                ),
                eraser=gr.Eraser(default_size=40),
                layers=False,
                buttons=["fullscreen"],
                label="✏️  Mark area",
                placeholder="Take or import a photo",
                height=360,
                canvas_size=(900, 700),
                visible=False,
            )
            ocr_btn = gr.Button("🔍  Read It to Me", variant="primary")
            with gr.Row(equal_height=True):
                ocr_out = gr.Textbox(
                    label="📄  Extracted text",
                    lines=7,
                    interactive=False,
                    scale=3,
                )
                ocr_audio_out = gr.Audio(
                    label="🔊  Reading",
                    autoplay=True,
                    interactive=False,
                    scale=2,
                )
            take_photo_btn.click(show_camera_capture, outputs=[camera_capture, camera_in])
            camera_capture.change(
                load_camera_capture,
                inputs=camera_capture,
                outputs=[camera_in, camera_capture],
            )
            import_photo_btn.upload(
                load_uploaded_photo,
                inputs=import_photo_btn,
                outputs=[camera_in, camera_capture],
            )
            ocr_btn.click(handle_ocr, inputs=camera_in, outputs=[ocr_out, ocr_audio_out])

        # ── 📋 Doctor Brief ───────────────────────────────────────────────────
        with gr.Tab("📋  Doctor Brief"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Generates a <strong style="color:#7a92aa;">change-focused brief</strong> '
                'with six sections.</p>'
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'New &nbsp;·&nbsp; Changed &nbsp;·&nbsp; Resolved &nbsp;·&nbsp; '
                'Ongoing &nbsp;·&nbsp; Readings &nbsp;·&nbsp; Questions to raise.</p>'
            )
            days_slider = gr.Slider(
                7, 90, value=30, step=1,
                label="📅  Days to include",
            )
            brief_btn = gr.Button("📋  Generate Brief", variant="primary")
            with gr.Row(equal_height=True):
                brief_out = gr.Textbox(
                    label="📄  Doctor brief",
                    lines=8,
                    interactive=False,
                    elem_classes=["brief-box"],
                    scale=3,
                )
                brief_audio_out = gr.Audio(
                    label="🔊  Brief read aloud",
                    autoplay=True,
                    interactive=False,
                    scale=2,
                    visible=False,
                )
            brief_btn.click(
                handle_brief,
                inputs=days_slider,
                outputs=[brief_out, brief_audio_out],
            )

        # ── 📖 History ────────────────────────────────────────────────────────
        with gr.Tab("📖  History"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">Last 7 days of log entries.</p>'
            )
            refresh_btn = gr.Button("🔄  Refresh", variant="secondary")
            history_out = gr.Markdown(value="_Press Refresh to load entries._")
            refresh_btn.click(handle_history, outputs=history_out)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, head=CUSTOM_HEAD)
