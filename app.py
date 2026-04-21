import streamlit as st
import streamlit.components.v1 as components
import time
import math
import numpy as np
import warnings
import logging
import os
import html

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, HF_EMBED_MODEL, HF_LLM_MODEL
from src.document_store import DocumentStore
from src.llm import decompose_query, answer_sub_query, aggregate_answers
from src.pipeline import SubQueryResult, PipelineResult

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ──────────────────────────────────────────────────────────────────
#  PDF helper
# ──────────────────────────────────────────────────────────────────
def extract_text_from_file(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            import fitz
            raw = uploaded_file.read()
            doc = fitz.open(stream=raw, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            try:
                import pdfplumber, io
                raw = uploaded_file.read()
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                st.error("PDF support requires PyMuPDF or pdfplumber.")
                return ""
    else:
        return uploaded_file.read().decode("utf-8")


# ──────────────────────────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralHop · Multi-Hop RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HOP_COLORS = ["#00E5FF", "#00E676", "#FFB300", "#FF2D78", "#9B59F5"]
HOP_DARKS  = ["#001F26", "#00200F", "#1F1600", "#200010", "#100820"]
HOP_NAMES  = ["cyan", "emerald", "amber", "pink", "violet"]

MAX_CHUNK_PREVIEW = 400


# ──────────────────────────────────────────────────────────────────
#  SVG / HTML helpers
# ──────────────────────────────────────────────────────────────────

def confidence_gauge(score: float) -> str:
    r = 52; cx = cy = 68
    circ   = 2 * math.pi * r
    arc    = circ * 0.75
    filled = arc * min(max(score, 0), 1)
    color  = "#00E676" if score > 0.7 else "#FFB300" if score > 0.4 else "#FF2D78"
    label  = "HIGH" if score > 0.7 else "MED" if score > 0.4 else "LOW"
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;">
  <svg width="136" height="136" viewBox="0 0 136 136" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <filter id="cglow">
        <feGaussianBlur stdDeviation="3.5" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <circle cx="{cx}" cy="{cy}" r="64" fill="none" stroke="{color}" stroke-width="0.5" opacity="0.12"/>
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1C2E50" stroke-width="9"
      stroke-linecap="round" stroke-dasharray="{arc:.2f} {circ:.2f}"
      transform="rotate(135 {cx} {cy})"/>
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="9"
      stroke-linecap="round" stroke-dasharray="{filled:.2f} {circ:.2f}"
      transform="rotate(135 {cx} {cy})" filter="url(#cglow)"/>
    <text x="{cx}" y="{cy-5}" text-anchor="middle"
      fill="{color}" style="font-family:'DM Mono',monospace;font-size:22px;font-weight:800;">{score:.3f}</text>
    <text x="{cx}" y="{cy+12}" text-anchor="middle" fill="#4A5568"
      style="font-family:'DM Mono',monospace;font-size:8px;letter-spacing:0.14em;">{label} CONF</text>
    <text x="16" y="118" fill="#1E2D45" style="font-family:'DM Mono',monospace;font-size:7px;">0</text>
    <text x="108" y="118" fill="#1E2D45" style="font-family:'DM Mono',monospace;font-size:7px;">1</text>
  </svg>
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{color};
              letter-spacing:0.14em;text-transform:uppercase;opacity:0.7;margin-top:-6px;">Confidence</div>
</div>"""


def hop_flow_svg(sub_queries: list, hop_colors: list) -> str:
    n = len(sub_queries); total = n + 2
    W, H = 820, 120; padx = 55; nr = 23; ys = H // 2 - 8
    xs = [padx + i * (W - 2 * padx) // (total - 1) for i in range(total)]
    p = ['<svg width="100%" viewBox="0 0 820 120" xmlns="http://www.w3.org/2000/svg">',
         '<defs>',
         '<filter id="qglow"><feGaussianBlur stdDeviation="4" result="b"/>'
         '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>']
    for i, c in enumerate(hop_colors[:n]):
        p.append(f'<filter id="hg{i}"><feGaussianBlur stdDeviation="4" result="b"/>'
                 f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    p.append('</defs>')

    for i in range(total - 1):
        x1, x2 = xs[i] + nr + 2, xs[i + 1] - nr - 2
        c = hop_colors[i % len(hop_colors)] if i < n else "#9B59F5"
        p.append(f'<line x1="{x1}" y1="{ys}" x2="{x2}" y2="{ys}" '
                 f'stroke="{c}" stroke-width="1.5" stroke-dasharray="5 3" opacity="0.4"/>'
                 f'<polygon points="{x2-5},{ys-4} {x2+1},{ys} {x2-5},{ys+4}" '
                 f'fill="{c}" opacity="0.55"/>')

    p.append(f'<circle cx="{xs[0]}" cy="{ys}" r="{nr}" fill="#050E1F" '
             f'stroke="#00E5FF" stroke-width="1.8" filter="url(#qglow)"/>'
             f'<text x="{xs[0]}" y="{ys+1}" text-anchor="middle" dominant-baseline="middle" '
             f'fill="#00E5FF" style="font-family:\'DM Mono\',monospace;'
             f'font-size:7px;letter-spacing:0.08em;font-weight:700;">QUERY</text>'
             f'<text x="{xs[0]}" y="{ys+nr+14}" text-anchor="middle" fill="#2D3E5A" '
             f'style="font-family:\'DM Mono\',monospace;font-size:6.5px;">input</text>')

    for i in range(n):
        xi = xs[i + 1]; c = hop_colors[i % len(hop_colors)]
        short = (sub_queries[i][:15] + "…") if len(sub_queries[i]) > 15 else sub_queries[i]
        p.append(f'<circle cx="{xi}" cy="{ys}" r="{nr}" fill="{c}18" '
                 f'stroke="{c}" stroke-width="1.8" filter="url(#hg{i})"/>'
                 f'<text x="{xi}" y="{ys-3}" text-anchor="middle" dominant-baseline="middle" '
                 f'fill="{c}" style="font-family:\'Syne\',sans-serif;font-size:12px;font-weight:800;">H{i+1}</text>'
                 f'<text x="{xi}" y="{ys+9}" text-anchor="middle" dominant-baseline="middle" '
                 f'fill="{c}" style="font-family:\'DM Mono\',monospace;font-size:6px;opacity:0.6;">hop</text>'
                 f'<text x="{xi}" y="{ys+nr+14}" text-anchor="middle" fill="#2D3E5A" '
                 f'style="font-family:\'DM Mono\',monospace;font-size:6px;">{short}</text>')

    ax = xs[-1]
    p.append(f'<circle cx="{ax}" cy="{ys}" r="{nr}" fill="#9B59F518" '
             f'stroke="#9B59F5" stroke-width="1.8" filter="url(#qglow)"/>'
             f'<text x="{ax}" y="{ys+1}" text-anchor="middle" dominant-baseline="middle" '
             f'fill="#9B59F5" style="font-family:\'DM Mono\',monospace;'
             f'font-size:6.5px;letter-spacing:0.07em;font-weight:700;">ANSWER</text>'
             f'<text x="{ax}" y="{ys+nr+14}" text-anchor="middle" fill="#2D3E5A" '
             f'style="font-family:\'DM Mono\',monospace;font-size:6.5px;">output</text>')

    p.append('</svg>')
    return "".join(p)


def chunk_breakdown_html(chunk_map: dict) -> str:
    rows = ""
    max_count = max(chunk_map.values()) if chunk_map else 1
    for doc_name, count in chunk_map.items():
        short = doc_name if len(doc_name) <= 28 else doc_name[:25] + "…"
        bar_w = min(100, max(4, int(count / max(max_count, 1) * 100)))
        rows += f"""
        <div class="row">
          <span class="doc" title="{html.escape(doc_name)}">📄 {html.escape(short)}</span>
          <div class="bar-wrap">
            <div class="bar" style="width:{bar_w}%"></div>
          </div>
          <span class="count">{count} <span class="unit">chunks</span></span>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #050D1A;
    font-family: 'DM Mono', monospace;
    padding: .85rem 1.1rem;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    margin: 0;
  }}
  .label {{ font-size:.55rem;letter-spacing:.14em;text-transform:uppercase;color:#3D5270;margin-bottom:.6rem; }}
  .row {{ display:flex;align-items:center;gap:10px;padding:.35rem 0;border-bottom:1px solid rgba(255,255,255,0.04); }}
  .row:last-child {{ border-bottom:none; }}
  .doc {{ font-size:.65rem;color:#6B8AB0;min-width:160px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:0; }}
  .bar-wrap {{ flex:1;background:rgba(255,255,255,0.05);border-radius:99px;height:5px;overflow:hidden; }}
  .bar {{ height:100%;background:linear-gradient(90deg,#00E5FF,#9B59F5);border-radius:99px; }}
  .count {{ font-size:.7rem;color:#00E5FF;font-weight:600;min-width:38px;text-align:right;flex-shrink:0; }}
  .unit {{ opacity:.45;font-size:.6rem; }}
</style>
</head>
<body>
  <div class="label">◆ Chunk Breakdown</div>
  {rows}
</body>
</html>"""


def chunk_card_html(doc_name: str, chunk_idx: int, score: float, text: str,
                    color: str, max_preview: int = 400) -> str:
    """
    Returns a fully self-contained HTML page for a single chunk card.
    All user-supplied strings are html.escape()'d so raw HTML in chunk
    text can NEVER break the surrounding layout.
    """
    is_truncated = len(text) > max_preview
    preview      = text[:max_preview] + ("…" if is_truncated else "")
    extra_chars  = len(text) - max_preview if is_truncated else 0

    doc_esc     = html.escape(str(doc_name))
    preview_esc = html.escape(preview)          # ← critical: escapes ALL html in chunk text
    score_str   = f"{score:.3f}"

    trunc_badge = (
        f'<span class="trunc">+{extra_chars} chars</span>'
        if is_truncated else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background: #050D1A;
    font-family: 'DM Mono', monospace;
    padding: .85rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    margin: 0;
  }}
  .header {{
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:.55rem; padding-bottom:.45rem;
    border-bottom:1px solid rgba(255,255,255,0.06);
    gap: 8px;
  }}
  .source {{
    font-size:.68rem; color:#6B8AB0; letter-spacing:.05em;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    max-width:180px; flex-shrink:1;
  }}
  .meta {{ display:flex; gap:8px; align-items:center; flex-shrink:0; }}
  .idx   {{ font-size:.58rem; color:#3D5270; }}
  .score {{ font-size:.73rem; font-weight:600; color:{color}; }}
  .trunc {{ font-size:.53rem; color:#3D5270; letter-spacing:.06em; }}
  .body  {{
    font-size:.86rem; line-height:1.7; color:#C0D0E4;
    white-space:pre-wrap; word-break:break-word;
    font-family:'DM Mono',monospace;
  }}
</style>
</head>
<body>
  <div class="header">
    <span class="source" title="{doc_esc}">📄 {doc_esc}</span>
    <div class="meta">
      <span class="idx">#{chunk_idx}</span>
      <span class="score">score: {score_str}</span>
      {trunc_badge}
    </div>
  </div>
  <div class="body">{preview_esc}</div>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&display=swap');

:root {
  --bg:#03060F; --surface:#07101F; --card:#0A1628; --card-hi:#0D1C34;
  --border:rgba(255,255,255,0.07); --border-hi:rgba(0,229,255,0.28);
  --cyan:#00E5FF; --emerald:#00E676; --amber:#FFB300; --pink:#FF2D78; --violet:#9B59F5;
  --text:#E8F2FF; --muted:#6B8AB0; --dim:#172136; --radius:16px; --radius-sm:10px;
}

html, body, [class*="css"] { font-family:'Syne',sans-serif !important; background:var(--bg) !important; color:var(--text) !important; }
.stApp { background:var(--bg) !important; }
.stApp::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(ellipse 80% 60% at 5%  0%,  rgba(0,229,255,0.07) 0%,transparent 55%),
    radial-gradient(ellipse 50% 45% at 95% 8%,  rgba(0,230,118,0.05) 0%,transparent 50%),
    radial-gradient(ellipse 70% 55% at 50% 105%,rgba(155,89,245,0.07) 0%,transparent 55%),
    radial-gradient(ellipse 40% 35% at 80% 60%, rgba(255,45,120,0.04) 0%,transparent 45%);
  animation:aurora 16s ease-in-out infinite alternate;
}
@keyframes aurora { 0%{opacity:.7;transform:scale(1)} 100%{opacity:1;transform:scale(1.04)} }
.stApp::after {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  background-size:256px 256px; opacity:.4;
}

.main .block-container { position:relative;z-index:1;padding:2.5rem 3rem 6rem !important;max-width:1380px !important; }

[data-testid="stSidebar"] { background:linear-gradient(175deg,#04080F 0%,#070E1C 100%) !important;border-right:1px solid rgba(0,229,255,0.06) !important; }
[data-testid="stSidebar"] .block-container { padding:1.5rem 1.1rem !important; }

[data-testid="stTextInput"] input { background:var(--card) !important;border:1px solid var(--border-hi) !important;border-radius:var(--radius-sm) !important;color:var(--text) !important;font-family:'DM Mono',monospace !important;font-size:0.88rem !important;padding:.9rem 1.2rem !important;transition:border-color .2s,box-shadow .2s !important; }
[data-testid="stTextInput"] input:focus { border-color:var(--cyan) !important;box-shadow:0 0 0 3px rgba(0,229,255,.08) !important; }
[data-testid="stTextInput"] label { font-family:'DM Mono',monospace !important;font-size:.6rem !important;text-transform:uppercase;letter-spacing:.14em;color:var(--muted) !important; }

.stButton > button { background:linear-gradient(135deg,rgba(0,229,255,.1),rgba(155,89,245,.1)) !important;border:1px solid rgba(0,229,255,.25) !important;border-radius:var(--radius-sm) !important;color:var(--cyan) !important;font-family:'Syne',sans-serif !important;font-weight:700 !important;font-size:.82rem !important;letter-spacing:.08em;text-transform:uppercase;width:100% !important;padding:.7rem 1rem !important;transition:all .2s !important; }
.stButton > button:hover { background:linear-gradient(135deg,rgba(0,229,255,.2),rgba(155,89,245,.2)) !important;border-color:var(--cyan) !important;box-shadow:0 0 28px rgba(0,229,255,.18) !important;transform:translateY(-1px) !important; }
.delete-btn > button { background:linear-gradient(135deg,rgba(255,45,120,.1),rgba(255,100,50,.1)) !important;border:1px solid rgba(255,45,120,.35) !important;color:#FF2D78 !important; }
.delete-btn > button:hover { background:linear-gradient(135deg,rgba(255,45,120,.22),rgba(255,100,50,.18)) !important;border-color:#FF2D78 !important;box-shadow:0 0 28px rgba(255,45,120,.18) !important; }

[data-testid="stFileUploader"] { background:transparent !important;border:none !important;padding:0 !important; }
[data-testid="stFileUploaderDropzone"] { background:rgba(0,229,255,0.03) !important;border:1.5px dashed rgba(0,229,255,.18) !important;border-radius:var(--radius) !important;transition:border-color .2s,background .2s !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:rgba(0,229,255,.35) !important;background:rgba(0,229,255,.05) !important; }
[data-testid="stFileUploader"] label { font-family:'DM Mono',monospace !important;font-size:.6rem !important;text-transform:uppercase;letter-spacing:.14em;color:var(--muted) !important; }

[data-testid="stExpander"] { background:var(--card) !important;border:1px solid var(--border) !important;border-radius:var(--radius) !important;overflow:hidden !important;margin-top:.75rem !important; }
[data-testid="stExpander"] summary { color:var(--text) !important;font-family:'Syne',sans-serif !important;font-weight:600 !important;padding:1rem 1.25rem !important; }
[data-testid="stStatus"] { background:var(--card) !important;border:1px solid var(--border) !important;border-radius:var(--radius) !important; }
[data-testid="stTextArea"] textarea { background:#050D1A !important;border:1px solid rgba(255,255,255,0.1) !important;border-radius:8px !important;color:#C5D5E8 !important;font-family:'DM Mono',monospace !important;font-size:.85rem !important;line-height:1.7 !important; }

::-webkit-scrollbar { width:4px;height:4px }
::-webkit-scrollbar-track { background:var(--bg) }
::-webkit-scrollbar-thumb { background:rgba(0,229,255,.15);border-radius:99px }
hr { border-color:var(--border) !important; }

.nh-topbar { display:flex;align-items:center;justify-content:space-between;padding:0 0 2rem 0;border-bottom:1px solid var(--border);margin-bottom:2.5rem; }
.nh-logo { display:flex;align-items:center;gap:10px; }
.nh-logo-icon { width:34px;height:34px;background:linear-gradient(135deg,var(--cyan),var(--violet));border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;box-shadow:0 0 20px rgba(0,229,255,.25); }
.nh-logo-text { font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;letter-spacing:-.03em;background:linear-gradient(120deg,var(--cyan),var(--violet));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.nh-logo-version { font-family:'DM Mono',monospace;font-size:.55rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-top:1px; }
.nh-status-bar { display:flex;align-items:center;gap:12px; }
.nh-status-pill { display:inline-flex;align-items:center;gap:6px;padding:.3rem .85rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.1em;text-transform:uppercase; }
.nh-status-pill.ready { background:rgba(0,230,118,.07);border:1px solid rgba(0,230,118,.22);color:#00E676; }
.nh-status-pill.no-index { background:rgba(255,45,120,.07);border:1px solid rgba(255,45,120,.22);color:#FF2D78; }
.nh-dot { width:6px;height:6px;border-radius:50%;flex-shrink:0; }
.nh-dot.ready { background:#00E676;box-shadow:0 0 7px #00E676;animation:pdot 2.4s ease-in-out infinite; }
.nh-dot.noindex { background:#FF2D78;box-shadow:0 0 7px #FF2D78; }
@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(.7)} }

.nh-model-tags { display:flex;gap:8px;flex-wrap:wrap; }
.nh-model-tag { display:flex;align-items:center;gap:5px;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:.3rem .7rem; }
.nh-model-tag-k { font-family:'DM Mono',monospace;font-size:.55rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase; }
.nh-model-tag-v { font-family:'DM Mono',monospace;font-size:.6rem;color:var(--cyan);max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }

.nh-hero { margin-bottom:2.5rem; }
.nh-pill { display:inline-flex;align-items:center;gap:6px;background:rgba(0,229,255,.06);border:1px solid rgba(0,229,255,.18);border-radius:99px;padding:.28rem .9rem;font-family:'DM Mono',monospace;font-size:.6rem;color:var(--cyan);letter-spacing:.12em;margin-bottom:.85rem; }
.nh-title { font-size:3.4rem;font-weight:800;letter-spacing:-.05em;line-height:.95;background:linear-gradient(135deg,var(--cyan) 0%,var(--emerald) 30%,var(--violet) 65%,var(--pink) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;background-size:250% 250%;animation:shimmer 7s ease-in-out infinite; }
@keyframes shimmer { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.nh-sub { color:var(--muted);font-family:'DM Mono',monospace;font-size:.8rem;line-height:1.75;max-width:580px;margin:.65rem 0 0; }

.nh-ingest-panel { background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;position:sticky;top:1.5rem; }
.nh-panel-header { display:flex;align-items:center;gap:8px;margin-bottom:1.25rem; }
.nh-panel-icon { width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.85rem;flex-shrink:0; }
.nh-panel-title { font-family:'Syne',sans-serif;font-size:.88rem;font-weight:700;color:var(--text); }
.nh-panel-sub { font-family:'DM Mono',monospace;font-size:.58rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase; }

.nh-file-count { display:inline-flex;align-items:center;gap:5px;background:rgba(0,229,255,.07);border:1px solid rgba(0,229,255,.18);border-radius:99px;padding:.2rem .7rem;font-family:'DM Mono',monospace;font-size:.58rem;color:var(--cyan);letter-spacing:.08em;margin-top:.75rem; }
.nh-query-panel { background:var(--card);border:1px solid var(--border-hi);border-radius:var(--radius);padding:1.5rem; }

.nh-div { display:flex;align-items:center;gap:.75rem;margin:2rem 0 1.25rem; }
.nh-divline { flex:1;height:1px;background:linear-gradient(90deg,var(--dim),transparent) }
.nh-divtxt { font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);white-space:nowrap; }

.nh-answer-wrap { position:relative;border-radius:18px;padding:2px;background:linear-gradient(135deg,var(--cyan),var(--emerald),var(--violet),var(--pink));background-size:300% 300%;animation:gborder 5s ease infinite; }
@keyframes gborder { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.nh-answer-inner { background:#080E1E;border-radius:16px;padding:1.75rem 2rem;font-size:1.15rem;line-height:1.85;color:var(--text); }
.nh-answer-lbl { font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);opacity:.7;margin-bottom:.9rem; }

.nh-metric { background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem .75rem;text-align:center;margin-bottom:.65rem;transition:border-color .2s,box-shadow .2s; }
.nh-metric:hover { border-color:rgba(0,229,255,.15);box-shadow:0 0 16px rgba(0,229,255,.05) }
.nh-ml { font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:.35rem; }
.nh-m { font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;line-height:1; }
.nh-gauge-card { background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem .75rem .75rem;display:flex;flex-direction:column;align-items:center;margin-bottom:.65rem; }

.nh-flow-wrap { background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.25rem 1.5rem 0.85rem;margin:1.5rem 0 0; }
.nh-flow-lbl { font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:.85rem; }

.nh-hop { border-radius:14px;padding:1.4rem 1.5rem 1.2rem;margin-bottom:.5rem;position:relative;overflow:hidden;border-width:1px;border-style:solid; }
.nh-hop-num { font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.4rem;opacity:.7; }
.nh-hop-query { font-size:1.05rem;font-weight:700;color:var(--text);margin-bottom:.85rem; }
.nh-hop-answer { font-size:.95rem;line-height:1.7;color:#A0B4CC;padding:.85rem 1rem;border-radius:10px;background:rgba(0,0,0,.28);margin-bottom:.85rem;font-family:'DM Mono',monospace; }

.nh-chunk-label { font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;opacity:.7;margin-bottom:.5rem; }

.nh-hist { background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem;margin-bottom:.6rem; }
.nh-hist-q { font-weight:700;font-size:.95rem;margin-bottom:.35rem; }
.nh-hist-a { font-size:.82rem;color:var(--muted);line-height:1.55;font-family:'DM Mono',monospace; }
.nh-footer { font-family:'DM Mono',monospace;font-size:.56rem;letter-spacing:.1em;color:var(--dim);text-align:center;margin-top:4rem;text-transform:uppercase; }

.nh-index-info { background:rgba(0,230,118,.04);border:1px solid rgba(0,230,118,.15);border-radius:10px;padding:.7rem 1rem;margin-top:.85rem;display:flex;align-items:center;gap:8px; }
.nh-index-info-text { font-family:'DM Mono',monospace;font-size:.6rem;color:#00E676;letter-spacing:.06em; }

.nh-chunk-progress-row { display:flex;align-items:center;justify-content:space-between;font-family:'DM Mono',monospace;font-size:.65rem;color:#6B8AB0;padding:.22rem 0; }
.nh-chunk-progress-row .doc { color:#A0B4CC;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px; }
.nh-chunk-progress-row .cnt { color:#00E5FF;font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  Session state
# ──────────────────────────────────────────────────────────────────
for k, v in [
    ("document_store", None),
    ("history", []),
    ("chunk_map", {}),
    ("total_chunks", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                letter-spacing:-.03em;background:linear-gradient(120deg,#00E5FF,#9B59F5);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:.3rem;">⚡ NeuralHop</div>
    <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#3D5270;
                letter-spacing:.12em;text-transform:uppercase;margin-bottom:1.5rem;">
      Multi-Hop RAG Engine
    </div>
    <div style="background:#0A1628;border:1px solid rgba(255,255,255,0.05);border-radius:12px;
                padding:1rem;margin-bottom:1rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:.22rem 0;">
        <span style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.1em;
               text-transform:uppercase;color:#3D5270;">Model</span>
        <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#00E5FF;
               max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(HF_LLM_MODEL)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:.22rem 0;">
        <span style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.1em;
               text-transform:uppercase;color:#3D5270;">Embedder</span>
        <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#00E5FF;
               max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(HF_EMBED_MODEL)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:.6rem 0 .22rem;
                  border-top:1px solid rgba(255,255,255,0.05);margin-top:.4rem;">
        <span style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.1em;
               text-transform:uppercase;color:#3D5270;">Total Chunks</span>
        <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:#FFB300;font-weight:700;">
          {st.session_state.total_chunks if st.session_state.total_chunks else "—"}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.button("↺ Reset Session"):
        st.session_state.history = []
        st.session_state.document_store = None
        st.session_state.chunk_map = {}
        st.session_state.total_chunks = 0
        st.rerun()


# ──────────────────────────────────────────────────────────────────
#  TOP BAR
# ──────────────────────────────────────────────────────────────────
is_ready = st.session_state.document_store is not None
dot_cls  = "ready" if is_ready else "noindex"
pill_cls = "ready" if is_ready else "no-index"

if is_ready and st.session_state.total_chunks:
    n_docs   = len(st.session_state.chunk_map)
    n_chunks = st.session_state.total_chunks
    pill_txt = (
        f'Index Ready'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:.55rem;'
        f'color:#00E676;opacity:.65;margin-left:7px;">'
        f'· {n_docs} doc{"s" if n_docs!=1 else ""} · {n_chunks} chunks</span>'
    )
else:
    pill_txt = "No Index" if not is_ready else "Index Ready"

st.markdown(f"""
<div class="nh-topbar">
  <div class="nh-logo">
    <div class="nh-logo-icon">⚡</div>
    <div>
      <div class="nh-logo-text">NeuralHop</div>
      <div class="nh-logo-version">Multi-Hop RAG ·</div>
    </div>
  </div>
  <div class="nh-status-bar">
    <div class="nh-model-tags">
      <div class="nh-model-tag">
        <span class="nh-model-tag-k">LLM</span>
        <span class="nh-model-tag-v">{html.escape(HF_LLM_MODEL)}</span>
      </div>
      <div class="nh-model-tag">
        <span class="nh-model-tag-k">Embed</span>
        <span class="nh-model-tag-v">{html.escape(HF_EMBED_MODEL)}</span>
      </div>
    </div>
    <div class="nh-status-pill {pill_cls}">
      <span class="nh-dot {dot_cls}"></span>
      {pill_txt}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  HERO
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nh-hero">
  <div class="nh-pill"><span class="nh-dot ready"></span>Multi-Hop Reasoning Engine</div>
  <div class="nh-title">Ask across<br>your documents.</div>
  <p class="nh-sub">Upload your files, build the index, then ask questions that<br>
  span multiple sources. NeuralHop hops, retrieves, and synthesises.</p>
</div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  WORKSPACE
# ──────────────────────────────────────────────────────────────────
ingest_col, query_col = st.columns([5, 7], gap="large")


# ── INGEST PANEL ─────────────────────────────────────────────────
with ingest_col:
    st.markdown("""
    <div class="nh-ingest-panel">
      <div class="nh-panel-header">
        <div class="nh-panel-icon" style="background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.2);">📂</div>
        <div>
          <div class="nh-panel-title">Document Ingestion</div>
          <div class="nh-panel-sub">Upload · Chunk · Embed</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "UPLOAD .TXT OR .PDF FILES",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload one or more .txt or .pdf documents to index",
    )

    if uploaded_files:
        txt_count = sum(1 for f in uploaded_files if f.name.lower().endswith(".txt"))
        pdf_count = sum(1 for f in uploaded_files if f.name.lower().endswith(".pdf"))
        parts = []
        if txt_count: parts.append(f"{txt_count} txt")
        if pdf_count: parts.append(f"{pdf_count} pdf")
        badge_txt = " · ".join(parts) + f"  ({len(uploaded_files)} total)"
        st.markdown(
            f'<div class="nh-file-count">'
            f'<span style="width:5px;height:5px;background:var(--cyan);border-radius:50%;'
            f'box-shadow:0 0 5px var(--cyan);"></span>'
            f'{html.escape(badge_txt)}'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    if st.button("⚡ Build Index", disabled=not uploaded_files):
        store = DocumentStore()
        failed = []
        chunk_map = {}
        total = len(uploaded_files)

        progress_bar = st.progress(0, text="Starting ingestion…")
        status_text  = st.empty()
        chunk_log    = st.empty()
        running_total = 0

        for i, f in enumerate(uploaded_files):
            pct = int(i / total * 100)
            progress_bar.progress(pct, text=f"Processing file {i+1} / {total}…")
            status_text.markdown(
                f'<div class="nh-chunk-progress-row">'
                f'<span class="doc">✦ {html.escape(f.name)}</span>'
                f'<span style="color:#FFB300;font-family:\'DM Mono\',monospace;font-size:.65rem;">chunking…</span>'
                f'</div>', unsafe_allow_html=True)

            text = extract_text_from_file(f)
            if text.strip():
                before = len(store._chunks)
                store.add_document(doc_id=f.name, text=text)
                after = len(store._chunks)
                n_new = after - before
                chunk_map[f.name] = n_new
                running_total += n_new
                status_text.markdown(
                    f'<div class="nh-chunk-progress-row">'
                    f'<span class="doc">✓ {html.escape(f.name)}</span>'
                    f'<span class="cnt">{n_new} chunks</span>'
                    f'</div>', unsafe_allow_html=True)
            else:
                failed.append(f.name)
                chunk_map[f.name] = 0
                status_text.markdown(
                    f'<div class="nh-chunk-progress-row">'
                    f'<span class="doc" style="color:#FF2D78;">✗ {html.escape(f.name)}</span>'
                    f'<span style="color:#FF2D78;font-family:\'DM Mono\',monospace;font-size:.65rem;">no text</span>'
                    f'</div>', unsafe_allow_html=True)

            chunk_log.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.62rem;color:#6B8AB0;margin-top:.25rem;">'
                f'Running total: <span style="color:#FFB300;font-weight:700;">{running_total} chunks</span>'
                f' across <span style="color:#00E5FF;">{i+1} file{"s" if i+1>1 else ""}</span>'
                f'</div>', unsafe_allow_html=True)

        progress_bar.progress(90, text="Uploading to Qdrant…")
        status_text.markdown(
            '<div class="nh-chunk-progress-row">'
            '<span class="doc">⚡ Building vector index…</span>'
            '<span style="color:#9B59F5;font-family:\'DM Mono\',monospace;font-size:.65rem;">embedding</span>'
            '</div>', unsafe_allow_html=True)

        store.build_index()
        st.session_state.document_store = store
        st.session_state.chunk_map = chunk_map
        st.session_state.total_chunks = running_total

        progress_bar.progress(100, text="Index complete!")
        status_text.empty()
        chunk_log.empty()

        if failed:
            st.warning(f"⚠ Could not extract text from: {', '.join(failed)}")

        st.success(
            f"✓ Indexed **{len(uploaded_files) - len(failed)}** file(s) → "
            f"**{running_total}** chunks → Qdrant"
        )
        st.rerun()

    if st.session_state.document_store is not None:
        chunk_count = st.session_state.total_chunks
        doc_count   = len(st.session_state.chunk_map)

        st.markdown(f"""
        <div class="nh-index-info">
          <span style="width:7px;height:7px;background:#00E676;border-radius:50%;
                box-shadow:0 0 7px #00E676;flex-shrink:0;"></span>
          <span class="nh-index-info-text">
            Index live &nbsp;·&nbsp; {doc_count} doc{"s" if doc_count!=1 else ""}
            &nbsp;·&nbsp;
            <span style="color:#FFB300;">{chunk_count} chunks</span>
            &nbsp;in Qdrant
          </span>
        </div>""", unsafe_allow_html=True)

        if st.session_state.chunk_map:
            n_rows = len(st.session_state.chunk_map)
            components.html(
                chunk_breakdown_html(st.session_state.chunk_map),
                height=60 + 44 * n_rows,
                scrolling=False,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
        if st.button("🗑 Delete All Documents"):
            st.session_state.document_store = None
            st.session_state.history = []
            st.session_state.chunk_map = {}
            st.session_state.total_chunks = 0
            st.success("All documents cleared from the index.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1.25rem;padding:.85rem 1rem;background:rgba(0,0,0,.2);
                border-radius:10px;border:1px solid var(--border);">
      <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:var(--muted);
                  letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;">
        Chunking Config <span style="opacity:.4;">(semantic)</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">
        <div style="background:var(--card);border-radius:8px;padding:.5rem .7rem;border:1px solid var(--border);">
          <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;">Strategy</div>
          <div style="font-family:'Syne',sans-serif;font-size:.9rem;font-weight:800;color:var(--amber);margin-top:2px;">Semantic</div>
        </div>
        <div style="background:var(--card);border-radius:8px;padding:.5rem .7rem;border:1px solid var(--border);">
          <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;">Threshold</div>
          <div style="font-family:'Syne',sans-serif;font-size:.9rem;font-weight:800;color:var(--violet);margin-top:2px;">Percentile</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── QUERY PANEL ───────────────────────────────────────────────────
with query_col:
    st.markdown("""
    <div class="nh-query-panel">
      <div class="nh-panel-header">
        <div class="nh-panel-icon" style="background:rgba(155,89,245,.1);border:1px solid rgba(155,89,245,.2);">🔍</div>
        <div>
          <div class="nh-panel-title">Multi-Hop Query</div>
          <div class="nh-panel-sub">Decompose · Retrieve · Synthesise</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    query = st.text_input(
        "YOUR QUERY",
        placeholder="e.g. Compare the revenue growth of Company A with the R&D spend of Company B…",
        disabled=not st.session_state.document_store,
    )

    if st.session_state.document_store and st.session_state.total_chunks:
        st.markdown(
            f'<div style="display:flex;gap:1.25rem;margin:.5rem 0 .25rem;">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:.6rem;color:#3D5270;">'
            f'Searching across '
            f'<span style="color:#00E5FF;">{st.session_state.total_chunks} chunks</span>'
            f' · '
            f'<span style="color:#00E676;">{len(st.session_state.chunk_map)} doc{"s" if len(st.session_state.chunk_map)!=1 else ""}</span>'
            f'</span></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    run = st.button(
        "⚡ Analyze Query",
        disabled=not st.session_state.document_store or not query,
    )

    if not st.session_state.document_store:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;background:rgba(255,179,0,.05);
                    border:1px solid rgba(255,179,0,.15);border-radius:10px;padding:.75rem 1rem;margin-top:.5rem;">
          <span style="font-size:1rem;">💡</span>
          <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#FFB300;letter-spacing:.06em;">
            Upload documents on the left and build the index to start querying.
          </span>
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  PIPELINE
# ──────────────────────────────────────────────────────────────────
if run and query and st.session_state.document_store:
    t0 = time.perf_counter()

    with st.status("🚀 Initializing multi-hop pipeline…", expanded=True) as status:
        st.write("🔍 Decomposing query into atomic reasoning steps…")
        sub_queries = decompose_query(query)
        st.write(f"✦ Generated **{len(sub_queries)}** sub-queries")

        sub_results = []
        for i, sq in enumerate(sub_queries):
            status.update(label=f"🔄 Hop {i+1}/{len(sub_queries)} — retrieving evidence…")
            st.write(f"**[Hop {i+1}]** {sq}")
            chunks, scores = st.session_state.document_store.retrieve(sq)
            llm_answer     = answer_sub_query(sq, chunks)
            avg_sim        = float(np.mean(scores)) if scores else 0.0
            sub_results.append(SubQueryResult(
                sub_query=sq, chunks=chunks, chunk_scores=scores,
                llm_answer=llm_answer, avg_similarity=avg_sim,
            ))

        status.update(label="✍️ Synthesizing final answer…", state="running")
        final_answer = aggregate_answers(
            query, sub_queries, [r.llm_answer for r in sub_results])

        elapsed    = time.perf_counter() - t0
        all_scores = [s for r in sub_results for s in r.chunk_scores]
        agg_score  = float(np.mean(all_scores)) if all_scores else 0.0

        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    # ── RESULTS ──────────────────────────────────────────────────
    st.markdown(
        '<div class="nh-div"><span class="nh-divtxt">Results</span>'
        '<div class="nh-divline"></div></div>', unsafe_allow_html=True)

    ans_col, panel_col = st.columns([3, 1], gap="medium")

    with ans_col:
        st.markdown(f"""
        <div class="nh-answer-wrap">
          <div class="nh-answer-inner">
            <div class="nh-answer-lbl">◆ Synthesised Response</div>
            {html.escape(final_answer)}
          </div>
        </div>""", unsafe_allow_html=True)

    with panel_col:
        st.markdown(
            f'<div class="nh-gauge-card">{confidence_gauge(agg_score)}</div>',
            unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nh-metric">
          <div class="nh-ml">Processing Time</div>
          <div class="nh-m" style="color:#00E5FF;">{elapsed:.2f}
            <span style="font-size:.9rem;opacity:.5;">s</span></div>
        </div>
        <div class="nh-metric">
          <div class="nh-ml">Reasoning Hops</div>
          <div class="nh-m" style="color:#9B59F5;">{len(sub_queries)}</div>
        </div>
        <div class="nh-metric">
          <div class="nh-ml">Chunks Retrieved</div>
          <div class="nh-m" style="color:#FFB300;">{len(all_scores)}</div>
        </div>
        <div class="nh-metric">
          <div class="nh-ml">Index Size</div>
          <div class="nh-m" style="color:#00E676;font-size:1.4rem;">{st.session_state.total_chunks}
            <span style="font-size:.75rem;opacity:.5;">total</span></div>
        </div>""", unsafe_allow_html=True)

    # ── HOP FLOW ─────────────────────────────────────────────────
    colors_used = [HOP_COLORS[i % len(HOP_COLORS)] for i in range(len(sub_queries))]
    st.markdown(
        f'<div class="nh-flow-wrap">'
        f'<div class="nh-flow-lbl">◆ Reasoning Flow</div>'
        f'{hop_flow_svg(sub_queries, colors_used)}'
        f'</div>', unsafe_allow_html=True)

    # ── REASONING CHAIN ──────────────────────────────────────────
    st.markdown(
        '<div class="nh-div"><span class="nh-divtxt">Reasoning Chain</span>'
        '<div class="nh-divline"></div></div>', unsafe_allow_html=True)

    for i, res in enumerate(sub_results):
        c    = HOP_COLORS[i % len(HOP_COLORS)]
        dark = HOP_DARKS[i % len(HOP_DARKS)]
        name = HOP_NAMES[i % len(HOP_NAMES)]

        left_c, right_c = st.columns([3, 2], gap="medium")

        with left_c:
            # hop_answer is also escaped — it comes from the LLM but better safe
            st.markdown(f"""
            <div class="nh-hop" style="border-color:{c}22;background:{dark}88;">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,{c},transparent);"></div>
              <div class="nh-hop-num" style="color:{c};">
                ◆ HOP {i+1} &nbsp;/&nbsp; {len(sub_results)} &nbsp;—&nbsp; {name}
              </div>
              <div class="nh-hop-query">{html.escape(res.sub_query)}</div>
              <div class="nh-hop-answer">{html.escape(res.llm_answer)}</div>
            </div>""", unsafe_allow_html=True)

        with right_c:
            # ── Chunk label (safe — no user content) ─────────────
            st.markdown(
                f'<div class="nh-chunk-label" style="color:{c};">◆ Retrieved Chunks</div>',
                unsafe_allow_html=True)

            # ── Each chunk card rendered in its own iframe ────────
            # This completely isolates chunk text from the parent HTML,
            # so no amount of HTML tags in a chunk can break the layout.
            for idx, (chunk, sc) in enumerate(zip(res.chunks[:3], res.chunk_scores[:3])):
                doc_name  = getattr(chunk, "doc_id", f"doc_{idx}")
                chunk_idx = getattr(chunk, "index", idx)
                raw_text  = chunk.text

                card_html = chunk_card_html(
                    doc_name=doc_name,
                    chunk_idx=chunk_idx,
                    score=sc,
                    text=raw_text,
                    color=c,
                    max_preview=MAX_CHUNK_PREVIEW,
                )
                # Estimate height: header ~50px + ~16px per line of text
                preview_lines = min(len(raw_text), MAX_CHUNK_PREVIEW) // 55 + 1
                card_height   = 70 + preview_lines * 22
                components.html(card_html, height=card_height, scrolling=False)

    st.session_state.history.append(
        {"query": query, "answer": final_answer, "score": agg_score})


# ──────────────────────────────────────────────────────────────────
#  HISTORY
# ──────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown(
        '<div class="nh-div"><span class="nh-divtxt">History</span>'
        '<div class="nh-divline"></div></div>', unsafe_allow_html=True)

    with st.expander(
            f"📜  {len(st.session_state.history)} previous "
            f"quer{'ies' if len(st.session_state.history) > 1 else 'y'}",
            expanded=False):
        for h in reversed(st.session_state.history):
            sc_c = "#00E676" if h["score"] > 0.7 else "#FFB300" if h["score"] > 0.4 else "#FF2D78"
            st.markdown(f"""
            <div class="nh-hist">
              <div class="nh-hist-q">↳ {html.escape(h["query"])}</div>
              <div class="nh-hist-a">{html.escape(h["answer"][:200])}…</div>
              <div style="margin-top:.5rem;">
                <span style="font-family:'DM Mono',monospace;font-size:.65rem;
                             color:{sc_c};letter-spacing:.1em;">SCORE {h["score"]:.3f}</span>
              </div>
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nh-footer">
  NeuralHop &nbsp;·&nbsp; Powered by LangChain · HuggingFace · Qdrant
</div>""", unsafe_allow_html=True)