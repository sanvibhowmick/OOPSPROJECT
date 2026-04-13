import streamlit as st
import time
import math
import numpy as np

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, OLLAMA_EMBED_MODEL, OLLAMA_MODEL
from src.document_store import DocumentStore
from src.llm import decompose_query, answer_sub_query, aggregate_answers
from src.pipeline import SubQueryResult, PipelineResult

# ──────────────────────────────────────────────────────────────────
#  PDF helper
# ──────────────────────────────────────────────────────────────────
def extract_text_from_file(uploaded_file) -> str:
    """Return plain text from a .txt or .pdf upload."""
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            raw = uploaded_file.read()
            doc = fitz.open(stream=raw, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            try:
                import pdfplumber
                import io
                raw = uploaded_file.read()
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    return "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
            except ImportError:
                st.error(
                    "PDF support requires PyMuPDF or pdfplumber. "
                    "Run: pip install pymupdf   or   pip install pdfplumber"
                )
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


# ──────────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&display=swap');

:root {
  --bg:#03060F;
  --surface:#07101F;
  --card:#0A1628;
  --card-hi:#0D1C34;
  --border:rgba(255,255,255,0.07);
  --border-hi:rgba(0,229,255,0.28);
  --cyan:#00E5FF;
  --emerald:#00E676;
  --amber:#FFB300;
  --pink:#FF2D78;
  --violet:#9B59F5;
  --text:#E8F2FF;
  --muted:#6B8AB0;
  --dim:#172136;
  --radius:16px;
  --radius-sm:10px;
}

html, body, [class*="css"] {
  font-family: 'Syne', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Aurora background ── */
.stApp { background: var(--bg) !important; }
.stApp::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(ellipse 80% 60% at 5%  0%,   rgba(0,229,255,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 50% 45% at 95% 8%,   rgba(0,230,118,0.05) 0%, transparent 50%),
    radial-gradient(ellipse 70% 55% at 50% 105%,  rgba(155,89,245,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 40% 35% at 80% 60%,  rgba(255,45,120,0.04) 0%, transparent 45%);
  animation: aurora 16s ease-in-out infinite alternate;
}
@keyframes aurora { 0% { opacity:.7; transform:scale(1) } 100% { opacity:1; transform:scale(1.04) } }

/* ── Noise texture overlay ── */
.stApp::after {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  background-size: 256px 256px;
  opacity: 0.4;
}

.main .block-container {
  position: relative; z-index: 1;
  padding: 2.5rem 3rem 6rem !important;
  max-width: 1380px !important;
}

/* ── Sidebar (minimal) ── */
[data-testid="stSidebar"] {
  background: linear-gradient(175deg, #04080F 0%, #070E1C 100%) !important;
  border-right: 1px solid rgba(0,229,255,0.06) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.1rem !important; }

/* ── Inputs ── */
[data-testid="stTextInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.88rem !important;
  padding: .9rem 1.2rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 3px rgba(0,229,255,.08) !important;
}
[data-testid="stTextInput"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: .6rem !important;
  text-transform: uppercase; letter-spacing: .14em; color: var(--muted) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,.1), rgba(155,89,245,.1)) !important;
  border: 1px solid rgba(0,229,255,.25) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--cyan) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .82rem !important;
  letter-spacing: .08em;
  text-transform: uppercase;
  width: 100% !important;
  padding: .7rem 1rem !important;
  transition: all .2s !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,229,255,.2), rgba(155,89,245,.2)) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 28px rgba(0,229,255,.18) !important;
  transform: translateY(-1px) !important;
}

/* ── Delete button variant ── */
.delete-btn > button {
  background: linear-gradient(135deg, rgba(255,45,120,.1), rgba(255,100,50,.1)) !important;
  border: 1px solid rgba(255,45,120,.35) !important;
  color: #FF2D78 !important;
}
.delete-btn > button:hover {
  background: linear-gradient(135deg, rgba(255,45,120,.22), rgba(255,100,50,.18)) !important;
  border-color: #FF2D78 !important;
  box-shadow: 0 0 28px rgba(255,45,120,.18) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(0,229,255,0.03) !important;
  border: 1.5px dashed rgba(0,229,255,.18) !important;
  border-radius: var(--radius) !important;
  transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(0,229,255,.35) !important;
  background: rgba(0,229,255,.05) !important;
}
[data-testid="stFileUploader"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: .6rem !important;
  text-transform: uppercase; letter-spacing: .14em; color: var(--muted) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  margin-top: .75rem !important;
}
[data-testid="stExpander"] summary {
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  padding: 1rem 1.25rem !important;
}

/* ── Status ── */
[data-testid="stStatus"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
  background: #050D1A !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 8px !important;
  color: #C5D5E8 !important;
  font-family: 'DM Mono', monospace !important;
  font-size: .85rem !important;
  line-height: 1.7 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px }
::-webkit-scrollbar-track { background: var(--bg) }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,.15); border-radius: 99px }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ════════════════════════════
   CUSTOM COMPONENTS
   ════════════════════════════ */

/* Topbar */
.nh-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 0 2rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;
}
.nh-logo {
  display: flex; align-items: center; gap: 10px;
}
.nh-logo-icon {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex-shrink: 0;
  box-shadow: 0 0 20px rgba(0,229,255,.25);
}
.nh-logo-text {
  font-family: 'Syne', sans-serif;
  font-size: 1.15rem; font-weight: 800;
  letter-spacing: -.03em;
  background: linear-gradient(120deg, var(--cyan), var(--violet));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.nh-logo-version {
  font-family: 'DM Mono', monospace;
  font-size: .55rem; color: var(--muted);
  letter-spacing: .1em; text-transform: uppercase;
  margin-top: 1px;
}
.nh-status-bar {
  display: flex; align-items: center; gap: 12px;
}
.nh-status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: .3rem .85rem;
  border-radius: 99px;
  font-family: 'DM Mono', monospace;
  font-size: .58rem; letter-spacing: .1em; text-transform: uppercase;
}
.nh-status-pill.ready {
  background: rgba(0,230,118,.07);
  border: 1px solid rgba(0,230,118,.22);
  color: #00E676;
}
.nh-status-pill.no-index {
  background: rgba(255,45,120,.07);
  border: 1px solid rgba(255,45,120,.22);
  color: #FF2D78;
}
.nh-dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.nh-dot.ready { background: #00E676; box-shadow: 0 0 7px #00E676; animation: pdot 2.4s ease-in-out infinite; }
.nh-dot.noindex { background: #FF2D78; box-shadow: 0 0 7px #FF2D78; }
@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(.7)} }

/* Model tags in topbar */
.nh-model-tags {
  display: flex; gap: 8px; flex-wrap: wrap;
}
.nh-model-tag {
  display: flex; align-items: center; gap: 5px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .3rem .7rem;
}
.nh-model-tag-k {
  font-family: 'DM Mono', monospace;
  font-size: .55rem; color: var(--muted);
  letter-spacing: .1em; text-transform: uppercase;
}
.nh-model-tag-v {
  font-family: 'DM Mono', monospace;
  font-size: .6rem; color: var(--cyan);
  max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* Hero */
.nh-hero {
  margin-bottom: 2.5rem;
}
.nh-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,229,255,.06);
  border: 1px solid rgba(0,229,255,.18);
  border-radius: 99px;
  padding: .28rem .9rem;
  font-family: 'DM Mono', monospace;
  font-size: .6rem; color: var(--cyan);
  letter-spacing: .12em;
  margin-bottom: .85rem;
}
.nh-title {
  font-size: 3.4rem; font-weight: 800; letter-spacing: -.05em; line-height: .95;
  background: linear-gradient(135deg, var(--cyan) 0%, var(--emerald) 30%, var(--violet) 65%, var(--pink) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; background-size: 250% 250%;
  animation: shimmer 7s ease-in-out infinite;
}
@keyframes shimmer { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.nh-sub {
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: .8rem; line-height: 1.75;
  max-width: 580px; margin: .65rem 0 0;
}

/* ── Main workspace: two-pane ── */
.nh-workspace {
  display: grid;
  grid-template-columns: 1fr 1.8fr;
  gap: 1.5rem;
  align-items: start;
}

/* Ingest panel */
.nh-ingest-panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  position: sticky;
  top: 1.5rem;
}
.nh-panel-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 1.25rem;
}
.nh-panel-icon {
  width: 28px; height: 28px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: .85rem; flex-shrink: 0;
}
.nh-panel-title {
  font-family: 'Syne', sans-serif;
  font-size: .88rem; font-weight: 700;
  color: var(--text);
}
.nh-panel-sub {
  font-family: 'DM Mono', monospace;
  font-size: .58rem; color: var(--muted);
  letter-spacing: .08em; text-transform: uppercase;
}

/* File count badge */
.nh-file-count {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(0,229,255,.07);
  border: 1px solid rgba(0,229,255,.18);
  border-radius: 99px;
  padding: .2rem .7rem;
  font-family: 'DM Mono', monospace;
  font-size: .58rem; color: var(--cyan);
  letter-spacing: .08em; margin-top: .75rem;
}

/* Query panel */
.nh-query-panel {
  background: var(--card);
  border: 1px solid var(--border-hi);
  border-radius: var(--radius);
  padding: 1.5rem;
}

/* Section divider */
.nh-div {
  display: flex; align-items: center; gap: .75rem;
  margin: 2rem 0 1.25rem;
}
.nh-divline { flex: 1; height: 1px; background: linear-gradient(90deg, var(--dim), transparent) }
.nh-divtxt {
  font-family: 'DM Mono', monospace;
  font-size: .6rem; letter-spacing: .18em;
  text-transform: uppercase; color: var(--muted); white-space: nowrap;
}

/* Answer card */
.nh-answer-wrap {
  position: relative; border-radius: 18px; padding: 2px;
  background: linear-gradient(135deg, var(--cyan), var(--emerald), var(--violet), var(--pink));
  background-size: 300% 300%; animation: gborder 5s ease infinite;
}
@keyframes gborder { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.nh-answer-inner {
  background: #080E1E; border-radius: 16px;
  padding: 1.75rem 2rem;
  font-size: 1.15rem; line-height: 1.85; color: var(--text);
}
.nh-answer-lbl {
  font-family: 'DM Mono', monospace;
  font-size: .65rem; letter-spacing: .18em;
  text-transform: uppercase; color: var(--cyan); opacity: .7; margin-bottom: .9rem;
}

/* Metrics */
.nh-metric {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem .75rem;
  text-align: center;
  margin-bottom: .65rem;
  transition: border-color .2s, box-shadow .2s;
}
.nh-metric:hover { border-color: rgba(0,229,255,.15); box-shadow: 0 0 16px rgba(0,229,255,.05) }
.nh-ml {
  font-family: 'DM Mono', monospace;
  font-size: .58rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: .35rem;
}
.nh-m {
  font-family: 'Syne', sans-serif;
  font-size: 1.9rem; font-weight: 800; line-height: 1;
}
.nh-gauge-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.25rem .75rem .75rem;
  display: flex; flex-direction: column; align-items: center;
  margin-bottom: .65rem;
}

/* Flow */
.nh-flow-wrap {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.25rem 1.5rem 0.85rem;
  margin: 1.5rem 0 0;
}
.nh-flow-lbl {
  font-family: 'DM Mono', monospace;
  font-size: .58rem; letter-spacing: .18em;
  text-transform: uppercase; color: var(--muted); margin-bottom: .85rem;
}

/* Hop cards */
.nh-hop {
  border-radius: 14px; padding: 1.4rem 1.5rem 1.2rem;
  margin-bottom: .5rem; position: relative; overflow: hidden;
  border-width: 1px; border-style: solid;
}
.nh-hop-num {
  font-family: 'DM Mono', monospace;
  font-size: .65rem; letter-spacing: .18em;
  text-transform: uppercase; margin-bottom: .4rem; opacity: .7;
}
.nh-hop-query { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: .85rem; }
.nh-hop-answer {
  font-size: .95rem; line-height: 1.7; color: #A0B4CC;
  padding: .85rem 1rem; border-radius: 10px;
  background: rgba(0,0,0,.28); margin-bottom: .85rem;
  font-family: 'DM Mono', monospace;
}

/* Chunk display — plain & readable */
.nh-chunk-block {
  background: #050D1A;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin-bottom: .75rem;
}
.nh-chunk-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: .6rem;
  padding-bottom: .5rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.nh-chunk-source {
  font-family: 'DM Mono', monospace;
  font-size: .7rem; color: var(--muted);
  letter-spacing: .05em;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px;
}
.nh-chunk-score {
  font-family: 'DM Mono', monospace;
  font-size: .75rem; font-weight: 600;
  flex-shrink: 0;
}
.nh-chunk-text {
  font-family: 'DM Mono', monospace;
  font-size: .88rem; line-height: 1.72; color: #C0D0E4;
  white-space: pre-wrap; word-break: break-word;
}

/* History */
.nh-hist {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  margin-bottom: .6rem;
}
.nh-hist-q { font-weight: 700; font-size: .95rem; margin-bottom: .35rem; }
.nh-hist-a { font-size: .82rem; color: var(--muted); line-height: 1.55; font-family: 'DM Mono', monospace; }

/* Footer */
.nh-footer {
  font-family: 'DM Mono', monospace;
  font-size: .56rem; letter-spacing: .1em;
  color: var(--dim); text-align: center; margin-top: 4rem;
  text-transform: uppercase;
}

/* Index info card */
.nh-index-info {
  background: rgba(0,230,118,.04);
  border: 1px solid rgba(0,230,118,.15);
  border-radius: 10px;
  padding: .7rem 1rem;
  margin-top: .85rem;
  display: flex; align-items: center; gap: 8px;
}
.nh-index-info-text {
  font-family: 'DM Mono', monospace;
  font-size: .6rem; color: #00E676;
  letter-spacing: .06em;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
#  Session state
# ──────────────────────────────────────────────────────────────────
for k, v in [("document_store", None), ("history", [])]:
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────
#  SIDEBAR  (model info + reset only)
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                letter-spacing:-.03em;
                background:linear-gradient(120deg,#00E5FF,#9B59F5);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:.3rem;">⚡ NeuralHop</div>
    <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#3D5270;
                letter-spacing:.12em;text-transform:uppercase;margin-bottom:1.5rem;">
      Multi-Hop RAG Engine 
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0A1628;border:1px solid rgba(255,255,255,0.05);border-radius:12px;
                padding:1rem;margin-bottom:1rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:.22rem 0;">
        <span style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.1em;
               text-transform:uppercase;color:#3D5270;">Model</span>
        <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#00E5FF;
               max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{OLLAMA_MODEL}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:.22rem 0;">
        <span style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.1em;
               text-transform:uppercase;color:#3D5270;">Embedder</span>
        <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#00E5FF;
               max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{OLLAMA_EMBED_MODEL}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:.22rem 0;">
        <span style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.1em;
               text-transform:uppercase;color:#3D5270;">Chunk / Overlap</span>
        <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#00E5FF;">{CHUNK_SIZE} / {CHUNK_OVERLAP}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.button("↺ Reset Session"):
        st.session_state.history = []
        st.session_state.document_store = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────
#  TOP BAR
# ──────────────────────────────────────────────────────────────────
is_ready = st.session_state.document_store is not None
dot_cls   = "ready" if is_ready else "noindex"
pill_cls  = "ready" if is_ready else "no-index"
pill_txt  = "Index Ready" if is_ready else "No Index"

doc_count_html = ""
if is_ready:
    try:
        n = len(st.session_state.document_store.documents)
        doc_count_html = f'<span style="font-family:\'DM Mono\',monospace;font-size:.58rem;color:#00E676;opacity:.7;margin-left:6px;">· {n} doc(s)</span>'
    except Exception:
        pass

st.markdown(f"""
<div class="nh-topbar">
  <div class="nh-logo">
    <div class="nh-logo-icon">⚡</div>
    <div>
      <div class="nh-logo-text">NeuralHop</div>
      <div class="nh-logo-version">Multi-Hop RAG · </div>
    </div>
  </div>
  <div class="nh-status-bar">
    <div class="nh-model-tags">
      <div class="nh-model-tag">
        <span class="nh-model-tag-k">LLM</span>
        <span class="nh-model-tag-v">{OLLAMA_MODEL}</span>
      </div>
      <div class="nh-model-tag">
        <span class="nh-model-tag-k">Embed</span>
        <span class="nh-model-tag-v">{OLLAMA_EMBED_MODEL}</span>
      </div>
    </div>
    <div class="nh-status-pill {pill_cls}">
      <span class="nh-dot {dot_cls}"></span>
      {pill_txt}{doc_count_html}
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
#  WORKSPACE  (two-column: ingest left | query right)
# ──────────────────────────────────────────────────────────────────
ingest_col, query_col = st.columns([5, 7], gap="large")

# ── INGEST PANEL ─────────────────────────────────────────────────
with ingest_col:
    st.markdown("""
    <div class="nh-ingest-panel">
      <div class="nh-panel-header">
        <div class="nh-panel-icon" style="background:rgba(0,229,255,.1);
             border:1px solid rgba(0,229,255,.2);">📂</div>
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
        if txt_count:
            parts.append(f"{txt_count} txt")
        if pdf_count:
            parts.append(f"{pdf_count} pdf")
        badge_txt = " · ".join(parts) + f"  ({len(uploaded_files)} total)"
        st.markdown(
            f'<div class="nh-file-count">'
            f'<span style="width:5px;height:5px;background:var(--cyan);border-radius:50%;'
            f'box-shadow:0 0 5px var(--cyan);"></span>'
            f'{badge_txt}'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    build_disabled = not uploaded_files
    if st.button("⚡ Build Index", disabled=build_disabled):
        with st.spinner("Chunking & embedding…"):
            store = DocumentStore(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            failed = []
            for f in uploaded_files:
                text = extract_text_from_file(f)
                if text.strip():
                    store.add_document(doc_id=f.name, text=text)
                else:
                    failed.append(f.name)
            store.build_index()
            st.session_state.document_store = store
        if failed:
            st.warning(f"⚠ Could not extract text from: {', '.join(failed)}")
        st.success(f"✓ Indexed {len(uploaded_files) - len(failed)} document(s)")
        st.rerun()

    # ── DELETE DOCS BUTTON ────────────────────────────────────────
    if st.session_state.document_store is not None:
        st.markdown("""
        <div class="nh-index-info">
          <span style="width:7px;height:7px;background:#00E676;border-radius:50%;
                box-shadow:0 0 7px #00E676;flex-shrink:0;"></span>
          <span class="nh-index-info-text">Index is live — ready to query</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
        if st.button("🗑 Delete All Documents"):
            st.session_state.document_store = None
            st.session_state.history = []
            st.success("All documents cleared from the index.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chunk config display
    st.markdown(f"""
    <div style="margin-top:1.25rem;padding:.85rem 1rem;
                background:rgba(0,0,0,.2);border-radius:10px;
                border:1px solid var(--border);">
      <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:var(--muted);
                  letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;">
        Chunking Config
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">
        <div style="background:var(--card);border-radius:8px;padding:.5rem .7rem;
                    border:1px solid var(--border);">
          <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:var(--muted);
                      letter-spacing:.1em;text-transform:uppercase;">Chunk Size</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                      color:var(--amber);margin-top:2px;">{CHUNK_SIZE}</div>
        </div>
        <div style="background:var(--card);border-radius:8px;padding:.5rem .7rem;
                    border:1px solid var(--border);">
          <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:var(--muted);
                      letter-spacing:.1em;text-transform:uppercase;">Overlap</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                      color:var(--violet);margin-top:2px;">{CHUNK_OVERLAP}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── QUERY PANEL ───────────────────────────────────────────────────
with query_col:
    st.markdown("""
    <div class="nh-query-panel">
      <div class="nh-panel-header">
        <div class="nh-panel-icon" style="background:rgba(155,89,245,.1);
             border:1px solid rgba(155,89,245,.2);">🔍</div>
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

    st.markdown("<br/>", unsafe_allow_html=True)

    run = st.button(
        "⚡ Analyze Query",
        disabled=not st.session_state.document_store or not query,
    )

    if not st.session_state.document_store:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;
                    background:rgba(255,179,0,.05);
                    border:1px solid rgba(255,179,0,.15);
                    border-radius:10px;padding:.75rem 1rem;margin-top:.5rem;">
          <span style="font-size:1rem;">💡</span>
          <span style="font-family:'DM Mono',monospace;font-size:.62rem;
                       color:#FFB300;letter-spacing:.06em;">
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
            {final_answer}
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
            st.markdown(f"""
            <div class="nh-hop" style="border-color:{c}22;background:{dark}88;">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,{c},transparent);"></div>
              <div class="nh-hop-num" style="color:{c};">
                ◆ HOP {i+1} &nbsp;/&nbsp; {len(sub_results)} &nbsp;—&nbsp; {name}
              </div>
              <div class="nh-hop-query">{res.sub_query}</div>
              <div class="nh-hop-answer">{res.llm_answer}</div>
            </div>""", unsafe_allow_html=True)

        with right_c:
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:.65rem;'
                f'letter-spacing:.14em;text-transform:uppercase;color:{c};'
                f'opacity:.7;margin-bottom:.5rem;">◆ Retrieved Chunks</div>',
                unsafe_allow_html=True)

            for idx, (chunk, sc) in enumerate(zip(res.chunks[:3], res.chunk_scores[:3])):
                doc = getattr(chunk, "doc_id", f"doc_{idx}")
                st.markdown(f"""
                <div class="nh-chunk-block">
                  <div class="nh-chunk-header">
                    <span class="nh-chunk-source">📄 {doc}</span>
                    <span class="nh-chunk-score" style="color:{c};">score: {sc:.3f}</span>
                  </div>
                  <div class="nh-chunk-text">{chunk.text}</div>
                </div>""", unsafe_allow_html=True)

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
              <div class="nh-hist-q">↳ {h["query"]}</div>
              <div class="nh-hist-a">{h["answer"][:200]}…</div>
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
  NeuralHop &nbsp;·&nbsp; Powered by LangChain · Ollama · Qdrant
</div>""", unsafe_allow_html=True)