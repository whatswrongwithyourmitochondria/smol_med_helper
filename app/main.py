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

def handle_checkin(audio_path: str | None, photo_path: str | None) -> tuple[str, None, object]:
    if not audio_path:
        return "", photo_path, gr.update()
    transcript = transcribe(audio_path)
    log_module.add_entry(transcript, entry_type="checkin", photo=photo_path or None)
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
<p style="margin:6px 0 0;font-size:0.83rem;color:#8a8064;text-align:center;line-height:1.4;">
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


def _speech_text_from_markdown(text: str) -> str:
    import re as _re
    lines = ["Doctor's brief."]  # spoken intro and TTS warm-up token
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("##"):
            heading = stripped.lstrip("#").strip()
            if heading:
                lines.append(f"\n{heading}.")
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        stripped = _DATE_PREFIX_RE.sub("", stripped)
        stripped = stripped.replace("#", "").strip()
        if stripped:
            lines.append(stripped)
    result = "\n".join(lines)
    return _re.sub(r"\n{3,}", "\n\n", result).strip()


@spaces.GPU(duration=120)
def handle_brief(days: int) -> str:
    return generate_brief(days=int(days))


@spaces.GPU(duration=60)
def handle_brief_read(brief_text: str) -> str:
    if not brief_text or brief_text.startswith("No log"):
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
            lines.append(f"{icon}&nbsp; {e['content']}")
        lines.append("")
    return "\n".join(lines)


# ── Design ────────────────────────────────────────────────────────────────────

CSS = """
/* Festa style — Fredoka + cream cards + chunky ink borders */

html, body { overflow-x: hidden !important; max-width: 100% !important; }

:root {
    --cream:  #FBF7EC;
    --ink:    #1E160C;
    --bg:     #cdbf86;
    --blue:   #2F6FE0;
    --red:    #D8362B;
    --mut:    #8a8064;
    --butter: #F7E27C;
}

*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Fredoka', sans-serif !important;
    color: var(--ink) !important;
}
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }
.contain, .wrap, .svelte-1gfkn6j { background: transparent !important; }

/* ── Hero header ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.app-title {
    font-family: 'Fredoka', sans-serif;
    font-size: clamp(2rem, 6vw, 3rem);
    font-weight: 700;
    color: var(--ink);
    margin: 0 0 6px;
    line-height: 1.1;
}
.app-sub {
    font-family: 'Fredoka', sans-serif;
    font-size: 0.9rem;
    color: var(--mut);
    letter-spacing: 0.5px;
    margin: 0;
}

/* ── Cards / blocks ── */
.block, .gr-group, .panel, .form {
    background: var(--cream) !important;
    border: 3px solid var(--ink) !important;
    border-radius: 18px !important;
    box-shadow: 5px 6px 0 var(--ink) !important;
}

/* ── Tabs — chunky chips ── */
div.tab-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 14px !important;
    gap: 9px !important;
    justify-content: center !important;
}
button.svelte-11gaq1 {
    font-family: 'Fredoka', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    padding: 9px 14px !important;
    min-height: 46px !important;
    border: 3px solid var(--ink) !important;
    background: var(--cream) !important;
    color: var(--ink) !important;
    box-shadow: 3px 4px 0 var(--ink) !important;
    transition: transform 0.13s ease, box-shadow 0.13s ease !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
}
button.svelte-11gaq1:hover:not(.selected) {
    transform: translateY(-2px) !important;
    box-shadow: 3px 6px 0 var(--ink) !important;
}
button.selected.svelte-11gaq1 {
    background: var(--red) !important;
    color: #fff !important;
    box-shadow: 3px 4px 0 var(--ink) !important;
}
button.selected.svelte-11gaq1::after { display: none !important; }

/* ── Primary button ── */
button.primary {
    font-family: 'Fredoka', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    min-height: 56px !important;
    border-radius: 44px !important;
    border: 3px solid var(--ink) !important;
    background: var(--red) !important;
    color: #fff !important;
    box-shadow: 5px 6px 0 var(--ink) !important;
    cursor: pointer !important;
    transition: transform 0.13s ease, box-shadow 0.13s ease !important;
}
button.primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 5px 9px 0 var(--ink) !important;
}
button.primary:active {
    transform: translateY(2px) !important;
    box-shadow: 2px 3px 0 var(--ink) !important;
}

/* ── Secondary button ── */
button.secondary {
    font-family: 'Fredoka', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    min-height: 48px !important;
    border-radius: 14px !important;
    background: var(--cream) !important;
    border: 3px solid var(--ink) !important;
    color: var(--ink) !important;
    box-shadow: 3px 4px 0 var(--ink) !important;
    transition: transform 0.13s ease, box-shadow 0.13s ease !important;
}
button.secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 3px 6px 0 var(--ink) !important;
}
button.secondary:active {
    transform: translateY(1px) !important;
    box-shadow: 1px 2px 0 var(--ink) !important;
}

/* ── Source-actions row ── */
.source-actions {
    justify-content: center !important;
    align-items: stretch !important;
    gap: 12px !important;
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
    min-height: 52px !important;
    min-width: 0 !important;
    border-radius: 14px !important;
}
.source-actions .clear-photo-button { flex: 0 0 80px !important; max-width: 80px !important; }
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

/* ── Text inputs ── */
textarea, input[type=text], input[type=number], input[type=search] {
    background: #fffdf6 !important;
    border: 2.5px solid var(--ink) !important;
    border-radius: 12px !important;
    font-family: 'Fredoka', sans-serif !important;
    font-size: 1rem !important;
    color: var(--ink) !important;
}
textarea:focus, input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(47,111,224,0.15) !important;
    outline: none !important;
}

/* ── Labels ── */
label > span, .label-wrap > span {
    font-family: 'Fredoka', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--mut) !important;
    text-transform: uppercase !important;
}

/* ── History markdown ── */
.prose p, .prose li { color: var(--ink) !important; font-size: 1rem !important; line-height: 1.7 !important; }
.prose h3 {
    font-family: 'Fredoka', sans-serif !important;
    color: var(--blue) !important;
    font-size: 1rem !important;
    border-bottom: 2px solid var(--ink) !important;
    padding-bottom: 4px !important;
    margin: 1.2rem 0 0.4rem !important;
}

/* ── Slider ── */
input[type=range] { accent-color: var(--red) !important; height: 6px !important; }

/* ── Audio ── */
.waveform-container, .waveform-container * { background: #fffdf6 !important; }

/* ── Doctor Brief compact box ── */
.brief-box textarea {
    min-height: 180px !important;
    max-height: 260px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--mut); border-radius: 4px; }

/* ── Blue photo-attach button ── */
.photo-attach-btn button {
    background: var(--blue) !important;
    color: #fff !important;
    border: 3px solid var(--ink) !important;
    box-shadow: 3px 4px 0 var(--ink) !important;
    border-radius: 14px !important;
    min-height: 52px !important;
    width: 100% !important;
    font-family: 'Fredoka', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: transform 0.13s ease, box-shadow 0.13s ease !important;
}
.photo-attach-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 3px 6px 0 var(--ink) !important;
}
.photo-attach-btn button:active {
    transform: translateY(1px) !important;
    box-shadow: 1px 2px 0 var(--ink) !important;
}

/* ── Falling hearts — behind all content ── */
@keyframes fall {
    0%   { transform: translateY(-60px) rotate(0deg);   opacity: 0; }
    10%  { opacity: .5; }
    90%  { opacity: .5; }
    100% { transform: translateY(110vh) rotate(220deg); opacity: 0; }
}
.heart-fall {
    position: fixed; pointer-events: none; z-index: -1; user-select: none;
    animation: fall linear infinite;
}

/* ── Hidden crop-coords box ── */
#crop-coords-box { display: none !important; }

/* ── Mobile ── */
@media (max-width: 640px) {
    .source-actions { flex-direction: column !important; }
    .source-actions .clear-photo-button { flex: 1 1 auto !important; max-width: none !important; }
    div.tab-container { flex-wrap: wrap !important; height: auto !important; }
    .tab-wrapper.svelte-11gaq1 { height: auto !important; overflow: visible !important; }
    button.svelte-11gaq1 {
        flex: 1 1 calc(50% - 8px) !important;
        min-width: 0 !important;
        padding: 9px 6px !important;
        font-size: 0.66rem !important;
    }
}
"""

HEADER_HTML = """
<div class="app-header">
    <h1 class="app-title">Patient <span style="color:#2F6FE0">Scribe</span></h1>
    <p class="app-sub">voice-first &nbsp;·&nbsp; camera &nbsp;·&nbsp; appointment-ready</p>
</div>
"""

CUSTOM_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
// ── Falling hearts ───────────────────────────────────────────────────────────
(function () {
    function spawnHearts() {
        var glyphs = ['❤️','🧡','🩷','💛','💙'];
        for (var i = 0; i < 14; i++) {
            var h = document.createElement('span');
            h.className = 'heart-fall';
            h.textContent = glyphs[i % glyphs.length];
            h.style.left = (Math.random() * 94) + '%';
            h.style.top = '-40px';
            h.style.fontSize = (30 + Math.random() * 26) + 'px';
            h.style.animationDuration = (7 + Math.random() * 7) + 's';
            h.style.animationDelay = (-Math.random() * 10) + 's';
            document.body.appendChild(h);
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', spawnHearts);
    } else {
        spawnHearts();
    }
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
                ctx.strokeStyle = '#2F6FE0';
                ctx.lineWidth   = 2;
                ctx.setLineDash([6, 3]);
                ctx.strokeRect(rx, ry, rw, rh);
                ctx.fillStyle = 'rgba(47,111,224,0.08)';
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
    primary_hue="red",
    secondary_hue="blue",
    neutral_hue="yellow",
    font=gr.themes.GoogleFont("Fredoka"),
    font_mono=gr.themes.GoogleFont("Fredoka"),
)

with gr.Blocks(title="Patient Scribe") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── 🎙️ Check-in ───────────────────────────────────────────────────────
        with gr.Tab("🎙️  Check-in"):
            gr.HTML(
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Press <strong style="color:#2F6FE0;">Record</strong> and speak your check-in, '
                'then <strong style="color:#2F6FE0;">Stop</strong>.</p>'
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'Optionally attach a photo — it will appear as a link in your doctor brief.</p>'
            )
            audio_in = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️  Your voice",
            )
            with gr.Row(elem_classes=["source-actions"]):
                checkin_photo_btn = gr.UploadButton(
                    "📷  Attach photo (optional)",
                    file_types=["image"],
                    type="filepath",
                    variant="secondary",
                    elem_classes=["source-button", "photo-attach-btn"],
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
        with gr.Tab("📷  Camera & Read"):
            gr.HTML(
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Point the camera at a <strong style="color:#2F6FE0;">medicine box</strong>, '
                'device screen, or letter.</p>'
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
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
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 0.6rem 0;">'
                'Generates a <strong style="color:#2F6FE0;">change-focused brief</strong> '
                'with six sections.</p>'
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">'
                'New &nbsp;·&nbsp; Changed &nbsp;·&nbsp; Resolved &nbsp;·&nbsp; '
                'Ongoing &nbsp;·&nbsp; Readings &nbsp;·&nbsp; Questions to raise.</p>'
            )
            days_slider = gr.Slider(
                7, 90, value=30, step=1,
                label="📅  Days to include",
            )
            brief_btn = gr.Button("📋  Generate Brief", variant="primary")
            brief_out = gr.Textbox(
                label="📄  Doctor brief",
                lines=8,
                interactive=False,
                elem_classes=["brief-box"],
            )
            read_brief_btn = gr.Button("🔊  Read Aloud", variant="secondary")
            brief_audio_out = gr.HTML(value=_EMPTY_AUDIO_HTML)
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
                '<p style="text-align:center;color:#8a8064;font-size:0.93rem;'
                'line-height:1.65;margin:0 0 1rem 0;">Last 7 days of log entries.</p>'
            )
            refresh_btn = gr.Button("🔄  Refresh", variant="secondary")
            history_out = gr.Markdown(value="_Press Refresh to load entries._")
            refresh_btn.click(handle_history, outputs=history_out)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, head=CUSTOM_HEAD)
