"""Gradio UI — voice-first health companion."""

from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

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

# ── Session state (server-side, keyed by Gradio session hash) ─────────────────
_session_photos: dict[str, str] = {}  # session_hash → full-res temp file path

# Persistent photo store — Gradio's upload temp files are deleted, so we copy
# attached photos here and keep a stable relative path in the log.
PHOTO_DIR = Path(__file__).parent.parent / "data" / "photos"
PHOTO_DIR.mkdir(parents=True, exist_ok=True)

# ── Handlers ──────────────────────────────────────────────────────────────────

def _persist_photo(src_path: str | None) -> str | None:
    """Copy an uploaded photo into data/photos/ and return its relative path."""
    if not src_path or not os.path.exists(src_path):
        return None
    ext = os.path.splitext(src_path)[1] or ".jpg"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dest = PHOTO_DIR / f"checkin-{stamp}{ext}"
    shutil.copyfile(src_path, dest)
    return str(dest.relative_to(PHOTO_DIR.parent.parent))


def handle_checkin(audio_path: str | None, photo_path: str | None) -> tuple[str, None, object]:
    if not audio_path:
        return "", photo_path, gr.update()
    transcript = transcribe(audio_path)
    saved_photo = _persist_photo(photo_path)
    log_module.add_entry(transcript, entry_type="checkin", photo=saved_photo)
    return transcript, None, gr.update(visible=False)


def _to_pil(source):
    from PIL import Image as PILImage
    if source is None:
        return None
    if isinstance(source, PILImage.Image):
        return source.convert("RGB")
    if isinstance(source, str):
        return PILImage.open(source).convert("RGB")
    if hasattr(source, "astype"):
        return PILImage.fromarray(source.astype("uint8")).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(source)!r}")


def _pil_to_b64(img) -> str:
    import base64
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_CANVAS_HTML = """
<div style="text-align:center;">
<div id="rsel-wrap" style="position:relative;display:inline-block;max-width:100%;touch-action:none;line-height:0;">
  <img id="rsel-img" src="{src}" draggable="false"
       style="display:block;max-width:100%;max-height:380px;width:auto;height:auto;
              user-select:none;-webkit-user-drag:none;">
  <canvas id="rsel-cvs" style="position:absolute;top:0;left:0;cursor:crosshair;touch-action:none;"></canvas>
</div>
<p style="margin:6px 0 0;font-size:0.83rem;color:#4d6a8a;text-align:center;line-height:1.4;">
  Drag to select an area &nbsp;·&nbsp; leave blank to read the whole image
</p>
</div>
"""


def _make_canvas_html(pil_img) -> str:
    thumb = pil_img.copy()
    thumb.thumbnail((900, 700))
    return _CANVAS_HTML.format(src=_pil_to_b64(thumb))


_EMPTY_AUDIO_HTML = (
    '<audio controls style="width:100%;height:36px;accent-color:#00d2ff;opacity:0.35;"></audio>'
)


def _make_audio_html(audio_bytes: bytes) -> str:
    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return (
        '<audio class="smc-autoplay" controls '
        'style="width:100%;height:36px;accent-color:#00d2ff;">'
        f'<source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">'
        '</audio>'
    )


def show_camera_capture():
    return (
        gr.update(visible=True, value=None),   # camera_capture
        gr.update(visible=False, value=""),    # canvas_selector
        gr.update(visible=False),              # clear_photo_btn
    )


def _save_photo(pil) -> str:
    """Save PIL image to a temp file and return the path."""
    tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    pil.save(tf.name, "JPEG", quality=95)
    tf.close()
    return tf.name


def load_camera_capture(image, request: gr.Request):
    if image is None:
        return gr.update(), gr.update(), gr.update()
    pil = _to_pil(image)
    path = _save_photo(pil)
    _session_photos[request.session_hash] = path
    print(f"[LOAD] webcam → {path}", flush=True)
    return (
        gr.update(visible=False, value=None),                   # camera_capture
        gr.update(value=_make_canvas_html(pil), visible=True),  # canvas_selector
        gr.update(visible=True),                                # clear_photo_btn
    )


def load_uploaded_photo(file_path, request: gr.Request):
    if file_path is None:
        return gr.update(), gr.update(), gr.update()
    pil = _to_pil(file_path)
    path = _save_photo(pil)
    _session_photos[request.session_hash] = path
    print(f"[LOAD] upload → {path}", flush=True)
    return (
        gr.update(value=_make_canvas_html(pil), visible=True),  # canvas_selector
        gr.update(visible=False, value=None),                   # camera_capture
        gr.update(visible=True),                                # clear_photo_btn
    )


def clear_photo_selection(request: gr.Request):
    _session_photos.pop(request.session_hash, None)
    return (
        gr.update(value="", visible=False),   # canvas_selector
        gr.update(value=None, visible=False), # camera_capture
        gr.update(visible=False),             # clear_photo_btn
        "",                                   # ocr_out
        _EMPTY_AUDIO_HTML,                    # ocr_audio_out (gr.HTML)
        "",                                   # crop_coords_box
    )


@spaces.GPU(duration=120)
def _do_ocr(image_path: str, crop_coords: str) -> tuple[str, str]:
    import os as _os
    from PIL import Image as PILImage

    if not image_path or not _os.path.exists(image_path):
        print("[OCR] no image — aborting", flush=True)
        return "", None

    img = PILImage.open(image_path).convert("RGB")
    print(f"[OCR] loaded image size={img.size}", flush=True)

    if crop_coords and crop_coords.strip():
        try:
            x1, y1, x2, y2 = [float(v) for v in crop_coords.split(",")]
            if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                w, h = img.size
                img = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
                print(f"[OCR] cropped to {img.size}", flush=True)
            else:
                print(f"[OCR] invalid crop coords {x1},{y1},{x2},{y2} — using full image", flush=True)
        except Exception:
            traceback.print_exc()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        ocr_path = f.name
    img.save(ocr_path)

    from app.ocr import extract_text
    from app.llm import clean_ocr
    try:
        result = extract_text(ocr_path)
        print(f"[OCR] raw={result!r}", flush=True)
    except Exception:
        traceback.print_exc()
        result = (
            "I could not read this image. Please try another photo with clearer "
            "lighting and the text fully in frame."
        )
    finally:
        _os.unlink(ocr_path)

    if not result:
        result = "No readable text found in this image."
    else:
        try:
            result = clean_ocr(result)
            print(f"[OCR] cleaned={result!r}", flush=True)
        except Exception:
            traceback.print_exc()

    log_module.add_entry(result, entry_type="ocr")
    audio_bytes = speak(result)
    return result, _make_audio_html(audio_bytes)


def handle_ocr(crop_coords: str, request: gr.Request) -> tuple[str, str]:
    image_path = _session_photos.get(request.session_hash, "")
    print(f"[OCR] image_path={image_path!r}  crop_coords={crop_coords!r}", flush=True)
    return _do_ocr(image_path, crop_coords)


_DATE_PREFIX_RE = __import__("re").compile(r"\d{4}-\d{2}-\d{2}[:\s]*")
_MD_LINK_RE = __import__("re").compile(r"\[([^\]]+)\]\([^)]+\)")


def _speech_text_from_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Never read photo links or the Photos section aloud.
        if "](" in stripped or "/gradio_api/file=" in stripped:
            continue
        if stripped.startswith("##"):
            heading = stripped.lstrip("#").strip()
            if heading and heading.lower() != "photos":
                lines.append(f"{heading}.")
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        stripped = _MD_LINK_RE.sub(r"\1", stripped)
        stripped = _DATE_PREFIX_RE.sub("", stripped)
        stripped = stripped.replace("#", "").strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(line for line in lines if line).strip()


@spaces.GPU(duration=120)
def handle_brief(days: int) -> str:
    return generate_brief(days=int(days))


@spaces.GPU(duration=60)
def handle_brief_read(brief_text: str) -> str:
    if not brief_text or brief_text.startswith("No log") or brief_text.startswith("_Press"):
        return _EMPTY_AUDIO_HTML
    audio_bytes = speak(_speech_text_from_markdown(brief_text))
    return _make_audio_html(audio_bytes)


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
            line = f"{icon}&nbsp; {e['content']}"
            if e.get("photo"):
                line += f' &nbsp; [📎 view photo](/gradio_api/file={e["photo"]})'
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


# ── Design ────────────────────────────────────────────────────────────────────

CSS = """
/* @import not allowed in constructable stylesheets — fonts loaded via head param instead */

/* ── Mobile overflow guard ── */
html, body {
    overflow-x: hidden !important;
    max-width: 100% !important;
}
/* Always reserve the vertical scrollbar so the viewport width never changes
   between tabs — otherwise the fixed/cover background re-fits and "jumps". */
html {
    overflow-y: scroll !important;
    scrollbar-gutter: stable !important;
}

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
    background-color: var(--bg) !important;
    background-image: url('/gradio_api/file=assets/bg-cover.png') !important;
    background-repeat: no-repeat !important;
    background-size: cover !important;
    background-position: center center !important;
    background-attachment: fixed !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    background-image: none !important;  /* pattern only on the page body, not the column */
    background-color: transparent !important;
}
.contain, .wrap, .svelte-1gfkn6j { background: transparent !important; }

/* ── Hero header ── */
.app-header {
    text-align: center;
    padding: 0.8rem 1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.app-title {
    font-family: 'Tomorrow', monospace;
    font-size: 4rem;
    font-weight: 700;
    letter-spacing: 0;
    background: linear-gradient(95deg, #00d2ff 0%, #3a7bd5 48%, #c471ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
    line-height: 1.1;
}
.app-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: var(--muted) !important;
    letter-spacing: 1px;
    text-transform: none;
    margin: 0;
}

/* ── Tabs — plain text, no boxes, no divider ── */
div.tabs.svelte-11gaq1,
div.tabs.svelte-11gaq1 > div.tab-wrapper.svelte-11gaq1,
div.tabs.svelte-11gaq1 div.tab-container.svelte-11gaq1,
.tab-wrapper.svelte-11gaq1,
div.tab-container.svelte-11gaq1,
.tabs.svelte-11gaq1,
.tabs, div.tabs,
.tabs > div, .tabs > div > div,
.tab-wrapper, div.tab-container,
[class*="tab-wrapper"], [class*="tab-container"],
.tab-nav, [role="tablist"],
[role="tabpanel"], .tabitem {
    background: transparent !important;
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
div.tabs.svelte-11gaq1 > div.tab-wrapper.svelte-11gaq1 > div.tab-container.svelte-11gaq1[role="tablist"],
.tabs [role="tablist"],
[role="tablist"] {
    margin-bottom: 14px !important;
    gap: 8px !important;
    justify-content: center !important;
}
.tabs::before, .tabs::after,
.tabs > div::before, .tabs > div::after,
.tabs > div > div::before, .tabs > div > div::after,
.tab-wrapper.svelte-11gaq1::before, .tab-wrapper.svelte-11gaq1::after,
.tab-wrapper::before, .tab-wrapper::after,
.tab-container::before, .tab-container::after,
[class*="tab-wrapper"]::before, [class*="tab-wrapper"]::after,
[class*="tab-container"]::before, [class*="tab-container"]::after,
[role="tablist"]::before, [role="tablist"]::after,
.tab-nav::before, .tab-nav::after { display: none !important; content: none !important; }
/* Keep every tab panel full width so the layout doesn't jump between tabs */
[role="tabpanel"], .tabitem { width: 100% !important; }
/* Tab buttons are plain text — no box, no border, no fill. Only the colour changes. */
[role="tablist"] button,
button.svelte-11gaq1[role="tab"],
[role="tab"] {
    font-family: 'Tomorrow', monospace !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border-radius: 0 !important;
    padding: 12px 18px !important;
    min-height: 0 !important;
    border: none !important;
    border-bottom: none !important;
    background: transparent !important;
    background-image: none !important;
    color: var(--muted) !important;
    transition: color 0.2s ease !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    line-height: 1 !important;
    vertical-align: middle !important;
}
/* Nudge emoji glyphs to sit on the same optical baseline as the label text */
[role="tablist"] button > *,
[role="tab"] > * { vertical-align: middle !important; }
[role="tablist"] button:hover:not(.selected),
[role="tab"]:hover:not(.selected) {
    background: transparent !important;
    color: var(--text) !important;
}
[role="tablist"] button:focus,
[role="tablist"] button:focus-visible,
[role="tab"]:focus,
[role="tab"]:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
[role="tablist"] button.selected,
[role="tablist"] button[aria-selected="true"],
[role="tab"].selected,
[role="tab"][aria-selected="true"] {
    background: transparent !important;
    color: #f97316 !important;
    border: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    outline: none !important;
    text-shadow: none !important;
}
/* Kill tab underlines and full-width separators, including Gradio pseudo-elements. */
[role="tablist"] button::before,
[role="tablist"] button::after,
button.svelte-11gaq1[role="tab"]::before,
button.svelte-11gaq1[role="tab"]::after,
[role="tab"]::before, [role="tab"]::after,
[role="tablist"] button.selected::before, [role="tablist"] button.selected::after,
[role="tablist"] button[aria-selected="true"]::before, [role="tablist"] button[aria-selected="true"]::after,
[role="tab"].selected::before, [role="tab"].selected::after,
[role="tab"][aria-selected="true"]::before, [role="tab"][aria-selected="true"]::after {
    background-color: transparent !important;
    background: transparent !important;
    display: none !important;
    height: 0 !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* ── Cards / blocks ── */
.block, .gr-group, .panel, .form {
    background: rgba(13, 21, 38, 0.72) !important;  /* semi-transparent so the heart pattern shows through */
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
}
/* NB: no backdrop-filter here — blur() breaks webcam / <canvas> / image painting
   in Chrome, which hid the captured photo in the Camera tab. */
/* Image / webcam components render on a solid surface so the feed + captured
   photo are always clearly visible. */
.image-container, .image-frame, [data-testid="image"] {
    background: var(--surface) !important;
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
    align-items: stretch !important;
    gap: 14px !important;
    margin: 0 0 16px !important;
}
.source-actions .source-button {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.source-actions .source-button button,
.source-actions button {
    width: 100% !important;
    min-height: 58px !important;
    min-width: 0 !important;
    border-radius: 14px !important;
}
.source-actions .clear-photo-button {
    flex: 0 0 92px !important;
    max-width: 92px !important;
}
.source-actions .clear-photo-button button {
    padding-left: 0 !important;
    padding-right: 0 !important;
    font-size: 1rem !important;
}
.camera-actions .clear-photo-button {
    flex: 1 1 0 !important;
    max-width: none !important;
}
.camera-actions .clear-photo-button button {
    font-size: 0.95rem !important;
    padding-left: 10px !important;
    padding-right: 10px !important;
}
.source-actions .upload-button,
.source-actions .file-preview,
.source-actions .wrap,
.source-actions .form,
.source-actions .block {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
@media (max-width: 640px) {
    .source-actions {
        flex-direction: column !important;
    }
    .source-actions .clear-photo-button {
        flex: 1 1 auto !important;
        max-width: none !important;
    }

    /* Tab bar: 2×2 grid so all four tabs fit without overflow */
    div.tab-container {
        flex-wrap: wrap !important;
        height: auto !important;
    }
    /* The wrapper's height comes from the invisible clone — let it grow too */
    .tab-wrapper.svelte-11gaq1,
    [class*="tab-wrapper"] {
        height: auto !important;
        overflow: visible !important;
    }
    [role="tablist"] button,
    button.svelte-11gaq1[role="tab"],
    [role="tab"] {
        flex: 1 1 calc(50% - 8px) !important;
        min-width: 0 !important;
        padding: 11px 8px !important;
        font-size: 0.9rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .app-title {
        font-size: 2.75rem;
    }
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
/* Days slider: NO box around the bar — strip the card off it and its wrappers */
.days-slider,
.days-slider.block, .days-slider .block, .days-slider .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
.days-slider { padding: 4px 16px 10px !important; margin: 0 0 4px !important; }
.days-slider .head, .days-slider .wrap { padding: 0 !important; }
/* Show ONLY the bar — hide the number readout box and the reset arrow */
.days-slider input[type=number],
.days-slider .number-input,
.days-slider button,
.days-slider .reset-button {
    display: none !important;
}
/* Neutralise the (now empty) head container so no leftover box shows */
.days-slider .head,
.days-slider .head > div,
.days-slider .wrap {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ── Read Aloud — identical to the "Attach photo" secondary button ── */
.read-aloud-btn button {
    font-family: 'Tomorrow', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    min-height: 58px !important;
    border-radius: 14px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.read-aloud-btn button:hover {
    border-color: rgba(0,210,255,0.4) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 16px rgba(0,210,255,0.1) !important;
}

/* ── Bare audio — no card box around the player ── */
.audio-bare, .audio-bare .block, .audio-bare .form,
.audio-bare .wrap, .audio-bare > * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Audio ── */
.waveform-container, .waveform-container * { background: var(--bg) !important; }
#checkin-audio {
    overflow: hidden !important;
    position: relative !important;
}
#checkin-audio .top-panel,
#checkin-audio .icon-button-wrapper.top-panel,
#checkin-audio .icon-button-wrapper.hide-top-corner {
    display: none !important;
}
#checkin-audio select,
#checkin-audio [role="combobox"],
#checkin-audio [aria-haspopup="listbox"] {
    display: none !important;
}
#checkin-audio .audio-container:has(.record-button),
#checkin-audio .audio-container:has(button[aria-label*="Record"]),
#checkin-audio .audio-container:has(button[title*="Record"]),
#checkin-audio.smc-record-idle .audio-container,
#checkin-audio .recording-container:has(.record-button),
#checkin-audio .recording-container:has(button[aria-label*="Record"]),
#checkin-audio .recording-container:has(button[title*="Record"]),
#checkin-audio.smc-record-idle .recording-container {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#checkin-audio .audio-container:has(.record-button),
#checkin-audio .audio-container:has(button[aria-label*="Record"]),
#checkin-audio .audio-container:has(button[title*="Record"]),
#checkin-audio.smc-record-idle .audio-container {
    min-height: 112px !important;
    position: relative !important;
}
#checkin-audio .record-button,
#checkin-audio .smc-record-button,
#checkin-audio button[aria-label*="Record"],
#checkin-audio button[title*="Record"] {
    margin: 0 auto !important;
    align-self: center !important;
    justify-self: center !important;
}
#checkin-audio.smc-record-idle .smc-record-button {
    position: absolute !important;
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: auto !important;
    min-width: max-content !important;
    z-index: 2 !important;
}
#checkin-audio.smc-record-idle .smc-record-noise {
    display: none !important;
}
#checkin-audio .controls[data-testid="waveform-controls"],
#checkin-audio .controls {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) !important;
    align-items: center !important;
    gap: 18px !important;
    width: 100% !important;
}
#checkin-audio .control-wrapper,
#checkin-audio .play-pause-wrapper,
#checkin-audio .settings-wrapper {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
}
#checkin-audio .control-wrapper {
    justify-self: start !important;
}
#checkin-audio .play-pause-wrapper {
    justify-self: center !important;
}
#checkin-audio .settings-wrapper {
    justify-self: end !important;
}
/* Keep the playback control icons (volume / speed / reset / trim) as plain
   transparent icon buttons — no bordered boxes. */
#checkin-audio .controls button,
#checkin-audio .control-wrapper button,
#checkin-audio .play-pause-wrapper button,
#checkin-audio .settings-wrapper button,
#checkin-audio button.action.icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 2.45rem !important;
    height: 2.45rem !important;
    min-height: 2.45rem !important;
    min-width: 2.45rem !important;
    padding: 0.48rem !important;
    font-size: 1.15rem !important;
    letter-spacing: 0 !important;
    line-height: 1.2 !important;
}
#checkin-audio .play-pause-button {
    width: 3rem !important;
    height: 3rem !important;
    min-width: 3rem !important;
    min-height: 3rem !important;
}
#checkin-audio .playback span {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
    line-height: 1 !important;
}
#checkin-audio button svg,
#checkin-audio .controls svg,
#checkin-audio .control-wrapper svg,
#checkin-audio .play-pause-wrapper svg,
#checkin-audio .settings-wrapper svg,
#checkin-audio button.action.icon svg {
    width: 1.45rem !important;
    height: 1.45rem !important;
    min-width: 1.45rem !important;
    min-height: 1.45rem !important;
}

/* ── History placeholder / entries — align text inside its box ── */
.history-box { padding: 4px 18px !important; }
.history-box p, .history-box li { text-align: left !important; }

/* ── Doctor Brief — compact scrollable box ── */
.brief-box textarea {
    min-height: 180px !important;
    max-height: 260px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}
/* Brief rendered as Markdown — one outer scroll area, no nested scrollbars */
.brief-box,
.brief-box > *,
.brief-box .prose,
.brief-box .md {
    max-height: none !important;
    overflow: visible !important;
}
.brief-box {
    min-height: 160px !important;
    max-height: 360px !important;
    overflow-y: auto !important;
    padding: 8px 18px !important;
}
.brief-box h2 {
    font-family: 'Tomorrow', monospace !important;
    color: var(--cyan) !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    margin: 1rem 0 0.3rem !important;
}
.brief-box p, .brief-box li { color: var(--text) !important; font-size: 1rem !important; line-height: 1.6 !important; }
.brief-box a { color: var(--cyan) !important; text-decoration: underline !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }

/* ── Hidden crop-coords textbox (must stay rendered for Gradio to track its value) ── */
#crop-coords-box { display: none !important; }
"""

HEADER_HTML = """
<div class="app-header">
    <h1 class="app-title">Patient Scribe</h1>
    <p class="app-sub">&middot; Get ready for the next appointment &middot;</p>
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

// ── Kill the divider/line under the tab buttons ─────────────────────────────
// CSS keeps losing the specificity/source-order battle against Gradio's scoped
// styles, so force it off inline with !important priority (beats any stylesheet).
(function () {
    var TAB_CSS = [
        '.tabs,.tabs>div,.tabs>div>div,.tab-wrapper,.tab-container,',
        '[class*="tab-wrapper"],[class*="tab-container"],[role="tablist"],.tab-nav{',
        'border:0!important;border-bottom:0!important;box-shadow:none!important;',
        'background:transparent!important;background-image:none!important;}',
        '.tabs::before,.tabs::after,.tabs>div::before,.tabs>div::after,',
        '.tabs>div>div::before,.tabs>div>div::after,.tab-wrapper::before,.tab-wrapper::after,',
        '.tab-container::before,.tab-container::after,[class*="tab-wrapper"]::before,',
        '[class*="tab-wrapper"]::after,[class*="tab-container"]::before,',
        '[class*="tab-container"]::after,[role="tablist"]::before,[role="tablist"]::after,',
        '.tab-nav::before,.tab-nav::after,[role="tablist"] button::before,',
        '[role="tablist"] button::after,[role="tab"]::before,[role="tab"]::after{',
        'display:none!important;content:none!important;',
        'height:0!important;border:0!important;box-shadow:none!important;background:transparent!important;}',
        '[role="tablist"] button.selected,[role="tablist"] button[aria-selected="true"],',
        '[role="tab"].selected,[role="tab"][aria-selected="true"]{',
        'color:#f97316!important;background:transparent!important;border:0!important;',
        'border-bottom:0!important;box-shadow:none!important;text-shadow:none!important;}'
    ].join('');
    var SEL = '.tabs, .tab-wrapper, .tab-container, [class*="tab-wrapper"], [class*="tab-container"], [role="tablist"], [role="tab"], .tab-nav';
    function installTabStyle() {
        var style = document.getElementById('smc-tabs-no-divider');
        if (!style) {
            style = document.createElement('style');
            style.id = 'smc-tabs-no-divider';
            document.head.appendChild(style);
        }
        if (style.textContent !== TAB_CSS) style.textContent = TAB_CSS;
    }
    function killDivider() {
        installTabStyle();
        var nodes = Array.from(document.querySelectorAll(SEL));
        document.querySelectorAll('[role="tablist"]').forEach(function (tablist) {
            [tablist.parentElement, tablist.parentElement && tablist.parentElement.parentElement]
                .forEach(function (el) { if (el) nodes.push(el); });
        });
        nodes.forEach(function (el) {
            el.style.setProperty('border', 'none', 'important');
            el.style.setProperty('border-bottom', 'none', 'important');
            el.style.setProperty('box-shadow', 'none', 'important');
            el.style.setProperty('background', 'transparent', 'important');
            el.style.setProperty('background-image', 'none', 'important');
        });
    }
    killDivider();
    new MutationObserver(killDivider).observe(document.documentElement, {
        childList: true, subtree: true
    });
    [50, 200, 600, 1500].forEach(function (t) { setTimeout(killDivider, t); });
}());

// Center the idle Check-in "Record" control without affecting playback controls
// after a recording exists. Gradio's generated classes change, so mark by text.
(function () {
    function markCheckinRecord() {
        var root = document.getElementById('checkin-audio');
        if (!root) return;
        var recordButton = null;
        root.querySelectorAll('button').forEach(function (button) {
            var text = (button.textContent || '').replace(/\\s+/g, ' ').trim();
            var isRecord = text === 'Record' || text.endsWith(' Record');
            button.classList.toggle('smc-record-button', isRecord);
            button.classList.toggle('smc-record-noise', !isRecord);
            if (isRecord) recordButton = button;
        });
        root.classList.toggle('smc-record-idle', !!recordButton);
        if (!recordButton) {
            root.querySelectorAll('.smc-record-noise').forEach(function (el) {
                el.classList.remove('smc-record-noise');
            });
        }
    }
    markCheckinRecord();
    new MutationObserver(markCheckinRecord).observe(document.documentElement, {
        childList: true, subtree: true, characterData: true
    });
    [50, 200, 600, 1500].forEach(function (t) { setTimeout(markCheckinRecord, t); });
}());

// ── Autoplay for audio injected via gr.HTML ─────────────────────────────────
(function () {
    new MutationObserver(function (mutations) {
        mutations.forEach(function (m) {
            m.addedNodes.forEach(function (node) {
                if (node.nodeType !== 1) return;
                var els = node.classList && node.classList.contains('smc-autoplay')
                    ? [node] : Array.from(node.querySelectorAll ? node.querySelectorAll('.smc-autoplay') : []);
                els.forEach(function (a) {
                    if (a._smc) return;
                    a._smc = true;
                    a.play().catch(function () {});
                });
            });
        });
    }).observe(document.documentElement, { childList: true, subtree: true });
}());

// ── Rectangle selector ──────────────────────────────────────────────────────
(function () {
    function initRsel() {
        var img = document.getElementById('rsel-img');
        var cvs = document.getElementById('rsel-cvs');
        if (!img || !cvs || cvs._rsel === img) return;
        cvs._rsel = img;  // mark as initialised for this img element

        var ctx = cvs.getContext('2d');
        var sx, sy, active = false, rx = 0, ry = 0, rw = 0, rh = 0;

        function resize() {
            cvs.width  = img.offsetWidth;
            cvs.height = img.offsetHeight;
            draw();
        }
        function draw() {
            ctx.clearRect(0, 0, cvs.width, cvs.height);
            if (rw > 4 && rh > 4) {
                ctx.strokeStyle = '#00d2ff';
                ctx.lineWidth   = 2;
                ctx.setLineDash([6, 3]);
                ctx.strokeRect(rx, ry, rw, rh);
                ctx.fillStyle = 'rgba(0,210,255,0.08)';
                ctx.fillRect(rx, ry, rw, rh);
            }
        }
        function pt(e) {
            var r = cvs.getBoundingClientRect();
            var t = e.touches ? e.touches[0] : e;
            return [t.clientX - r.left, t.clientY - r.top];
        }
        function onDown(e) { var p = pt(e); sx = p[0]; sy = p[1]; active = true; }
        function onMove(e) {
            if (!active) return;
            if (e.cancelable) e.preventDefault();
            var p = pt(e);
            rx = Math.min(sx, p[0]); ry = Math.min(sy, p[1]);
            rw = Math.abs(p[0] - sx); rh = Math.abs(p[1] - sy);
            draw();
        }
        function onUp() {
            if (!active) return;
            active = false;
            var tb = document.querySelector('#crop-coords-box textarea');
            if (!tb) return;
            if (rw < 4 || rh < 4) {
                tb.value = '';
            } else {
                tb.value = [rx/cvs.width, ry/cvs.height,
                            (rx+rw)/cvs.width, (ry+rh)/cvs.height]
                    .map(function (v) { return Math.max(0, Math.min(1, v)).toFixed(4); })
                    .join(',');
            }
            tb.dispatchEvent(new Event('input', { bubbles: true }));
        }

        cvs.addEventListener('mousedown',  onDown);
        cvs.addEventListener('mousemove',  onMove);
        cvs.addEventListener('mouseup',    onUp);
        cvs.addEventListener('touchstart', onDown, { passive: true });
        cvs.addEventListener('touchmove',  onMove, { passive: false });
        cvs.addEventListener('touchend',   onUp);

        if (img.complete && img.naturalWidth) resize();
        else img.addEventListener('load', resize);
        new ResizeObserver(resize).observe(img);
    }

    new MutationObserver(initRsel).observe(document.documentElement, {
        childList: true, subtree: true
    });
    initRsel();
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

with gr.Blocks(title="Patient Scribe") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── 🎙️ Check-in ───────────────────────────────────────────────────────
        with gr.Tab("🎙️  Check-in"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Press <strong style="color:#00d2ff;">Record</strong> and speak your check-in, '
                'then <strong style="color:#00d2ff;">Stop</strong>.</p>'
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'Optionally attach a photo.</p>'
            )
            audio_in = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️  Your voice",
                show_label=False,
                elem_id="checkin-audio",
            )
            with gr.Row(elem_classes=["source-actions"]):
                checkin_photo_btn = gr.UploadButton(
                    "📷  Attach photo (optional)",
                    file_types=["image"],
                    type="filepath",
                    variant="secondary",
                    elem_classes=["source-button"],
                )
                checkin_photo_clear_btn = gr.Button(
                    "🗑️",
                    variant="secondary",
                    visible=False,
                    elem_classes=["source-button", "clear-photo-button"],
                )
            checkin_photo_state = gr.State(None)
            checkin_photo_btn.upload(
                lambda p: (p, gr.update(visible=True)),
                inputs=checkin_photo_btn,
                outputs=[checkin_photo_state, checkin_photo_clear_btn],
            )
            checkin_photo_clear_btn.click(
                lambda: (None, gr.update(visible=False)),
                outputs=[checkin_photo_state, checkin_photo_clear_btn],
            )
            checkin_btn = gr.Button("⬆️  Log Check-in", variant="primary")
            transcript_out = gr.Textbox(
                label="📝  What I heard",
                lines=4,
                interactive=False,
            )
            checkin_btn.click(
                handle_checkin,
                inputs=[audio_in, checkin_photo_state],
                outputs=[transcript_out, checkin_photo_state, checkin_photo_clear_btn],
            )

        # ── 📷 Camera & Read ──────────────────────────────────────────────────
        with gr.Tab("📷  Camera"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Point the camera at or Import a '
                '<strong style="color:#00d2ff;">medicine box</strong>, '
                '<strong style="color:#00d2ff;">device screen</strong>, or '
                '<strong style="color:#00d2ff;">letter</strong>.</p>'
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'The model reads the text aloud and logs any numeric readings.</p>'
            )
            with gr.Row(elem_classes=["source-actions", "camera-actions"]):
                take_photo_btn = gr.Button(
                    "📷  Take a photo",
                    variant="secondary",
                    elem_classes=["source-button"],
                )
                import_photo_btn = gr.UploadButton(
                    "🖼️  Import a photo",
                    file_types=["image"],
                    type="filepath",
                    variant="secondary",
                    elem_classes=["source-button"],
                )
                clear_photo_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    visible=False,
                    elem_classes=["source-button", "clear-photo-button"],
                )
            camera_capture = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="📷  Take a photo",
                height=260,
                visible=False,
            )
            canvas_selector = gr.HTML(value="", visible=False)
            crop_coords_box = gr.Textbox(
                value="", elem_id="crop-coords-box", container=False, label="",
            )
            ocr_btn = gr.Button("🔍  Read It to Me", variant="primary")
            ocr_audio_out = gr.HTML(value=_EMPTY_AUDIO_HTML, elem_classes=["audio-bare"])
            ocr_out = gr.Textbox(
                label="📄  Extracted text",
                lines=7,
                interactive=False,
            )
            take_photo_btn.click(
                show_camera_capture,
                outputs=[camera_capture, canvas_selector, clear_photo_btn],
            )
            camera_capture.change(
                load_camera_capture,
                inputs=camera_capture,
                outputs=[camera_capture, canvas_selector, clear_photo_btn],
            )
            import_photo_btn.upload(
                load_uploaded_photo,
                inputs=import_photo_btn,
                outputs=[canvas_selector, camera_capture, clear_photo_btn],
            )
            clear_photo_btn.click(
                clear_photo_selection,
                outputs=[canvas_selector, camera_capture, clear_photo_btn,
                         ocr_out, ocr_audio_out, crop_coords_box],
            )
            ocr_btn.click(
                handle_ocr,
                inputs=[crop_coords_box],
                outputs=[ocr_out, ocr_audio_out],
            )

        # ── 📋 Doctor Brief ───────────────────────────────────────────────────
        with gr.Tab("📋  Brief"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Generates a <strong style="color:#00d2ff;">change-focused brief</strong> '
                'with six sections.</p>'
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'New &nbsp;·&nbsp; Changed &nbsp;·&nbsp; Resolved &nbsp;·&nbsp; '
                'Ongoing &nbsp;·&nbsp; Readings &nbsp;·&nbsp; Questions</p>'
            )
            days_slider = gr.Slider(
                7, 90, value=30, step=1,
                label="📅  Days to include",
                container=False,
                elem_classes=["days-slider"],
            )
            brief_btn = gr.Button("📋  Generate Brief", variant="primary")
            brief_out = gr.Markdown(
                value="_Press Generate Brief to build the doctor brief._",
                elem_classes=["brief-box"],
            )
            with gr.Row(elem_classes=["source-actions"]):
                read_brief_btn = gr.Button(
                    "🔊  Read Aloud", variant="secondary",
                    elem_classes=["source-button", "read-aloud-btn"],
                )
            brief_audio_out = gr.HTML(value=_EMPTY_AUDIO_HTML, elem_classes=["audio-bare"])
            brief_btn.click(
                handle_brief,
                inputs=days_slider,
                outputs=brief_out,
            )
            read_brief_btn.click(
                handle_brief_read,
                inputs=brief_out,
                outputs=brief_audio_out,
            )

        # ── 📖 History ────────────────────────────────────────────────────────
        with gr.Tab("📖  History"):
            gr.HTML(
                '<p style="text-align:center;color:#4d6a8a;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">Last 7 days of log entries.</p>'
            )
            refresh_btn = gr.Button("🔄  Refresh", variant="secondary")
            history_out = gr.Markdown(
                value="_Press Refresh to load entries._",
                elem_classes=["history-box"],
            )
            refresh_btn.click(handle_history, outputs=history_out)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, head=CUSTOM_HEAD, allowed_paths=["data", "assets"])
