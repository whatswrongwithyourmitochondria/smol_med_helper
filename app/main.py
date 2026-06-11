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

# ── Session state (server-side, keyed by Gradio session hash) ─────────────────
_session_photos: dict[str, str] = {}  # session_hash → full-res temp file path

# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_checkin(audio_path: str | None) -> str:
    if not audio_path:
        return ""
    transcript = transcribe(audio_path)
    log_module.add_entry(transcript, entry_type="checkin")
    return transcript


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


_AUDIO_WRAP = (
    'background:#0d1526;border:1px solid #192e50;border-radius:12px;padding:10px 14px;'
)

_EMPTY_AUDIO_HTML = (
    f'<div style="{_AUDIO_WRAP}">'
    '<audio controls style="width:100%;height:36px;accent-color:#00d2ff;opacity:0.35;"></audio>'
    '</div>'
)


def _make_audio_html(audio_bytes: bytes) -> str:
    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return (
        f'<div style="{_AUDIO_WRAP}">'
        '<audio class="smc-autoplay" controls '
        'style="width:100%;height:36px;accent-color:#00d2ff;">'
        f'<source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">'
        '</audio></div>'
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
    try:
        result = extract_text(ocr_path)
        print(f"[OCR] result={result!r}", flush=True)
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

    log_module.add_entry(result, entry_type="ocr")
    audio_bytes = speak(result)
    return result, _make_audio_html(audio_bytes)


def handle_ocr(crop_coords: str, request: gr.Request) -> tuple[str, str]:
    image_path = _session_photos.get(request.session_hash, "")
    print(f"[OCR] image_path={image_path!r}  crop_coords={crop_coords!r}", flush=True)
    return _do_ocr(image_path, crop_coords)


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

/* ── Mobile overflow guard ── */
html, body {
    overflow-x: hidden !important;
    max-width: 100% !important;
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
    justify-content: center !important;
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
    .tab-wrapper.svelte-11gaq1 {
        height: auto !important;
        overflow: visible !important;
    }
    button.svelte-11gaq1 {
        flex: 1 1 calc(50% - 8px) !important;
        min-width: 0 !important;
        padding: 11px 8px !important;
        font-size: 0.72rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
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

/* ── Hidden crop-coords textbox (must stay rendered for Gradio to track its value) ── */
#crop-coords-box { display: none !important; }
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
            with gr.Row(elem_classes=["source-actions"]):
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
            ocr_audio_out = gr.HTML(value=_EMPTY_AUDIO_HTML)
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
