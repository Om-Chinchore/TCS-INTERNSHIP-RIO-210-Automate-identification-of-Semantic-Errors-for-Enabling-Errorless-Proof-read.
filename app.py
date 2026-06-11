"""
app.py
======
Interactive Streamlit Dashboard — Automated Grammar & Semantic Error Correction.

Pages
-----
  Page 1 — Real-Time Proofreader
      Paste text → instant side-by-side error detection + transformer correction.

  Page 2 — Architecture & Metrics
      Visual breakdown of the dual-model pipeline + live performance metrics.

Run
---
    streamlit run app.py
"""

import time
import textwrap
from typing import Dict, List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import CONFIG
from preprocessing import TextPreprocessor, ErrorPatternDetector
from models import GrammarClassifier, SemanticCorrector, ErrorCorrectionPipeline

# ============================================================
# Page / global config
# ============================================================
st.set_page_config(
    page_title=CONFIG.app.app_title,
    page_icon=CONFIG.app.page_icon,
    layout=CONFIG.app.layout,
    initial_sidebar_state=CONFIG.app.sidebar_state,
)

# ============================================================
# CSS Theme
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Background ── */
:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #1c2128;
    --bg-card-hover:#21262d;
    --accent-blue:  #58a6ff;
    --accent-green: #3fb950;
    --accent-red:   #f85149;
    --accent-orange:#ffa657;
    --accent-purple:#bc8cff;
    --text-primary: #e6edf3;
    --text-secondary:#8b949e;
    --border:       #30363d;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid var(--border);
}

/* ── Header ── */
.hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #1c2128 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(88,166,255,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(188,140,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
}

/* ── Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s, background 0.2s;
}
.metric-card:hover {
    border-color: var(--accent-blue);
    background: var(--bg-card-hover);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-blue);
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

/* ── Status Badges ── */
.badge {
    display: inline-block;
    padding: 0.25em 0.75em;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-error   { background: rgba(248,81,73,0.15);  color: #f85149; border:1px solid rgba(248,81,73,0.3); }
.badge-warning { background: rgba(255,166,87,0.15); color: #ffa657; border:1px solid rgba(255,166,87,0.3); }
.badge-success { background: rgba(63,185,80,0.15);  color: #3fb950; border:1px solid rgba(63,185,80,0.3); }
.badge-info    { background: rgba(88,166,255,0.15); color: #58a6ff; border:1px solid rgba(88,166,255,0.3); }

/* ── Text panels ── */
.text-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    min-height: 120px;
    white-space: pre-wrap;
}
.text-panel.original { border-top: 3px solid var(--accent-red); }
.text-panel.corrected { border-top: 3px solid var(--accent-green); }

/* ── Finding items ── */
.finding-item {
    background: var(--bg-card);
    border-left: 4px solid var(--accent-orange);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.finding-item.ml-finding {
    border-left-color: var(--accent-purple);
}

/* ── Confidence bar ── */
.conf-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    transition: width 0.4s ease;
}

/* ── Pipeline diagram labels ── */
.pipeline-node {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stTextArea"] textarea {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
}
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Cached model loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_classifier() -> GrammarClassifier:
    clf = GrammarClassifier()
    if not clf.load():
        clf.train()
        clf.save()
    return clf


@st.cache_resource(show_spinner=False)
def load_corrector() -> SemanticCorrector:
    corrector = SemanticCorrector()
    corrector.load_model()
    return corrector


@st.cache_resource(show_spinner=False)
def load_preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


@st.cache_resource(show_spinner=False)
def load_rule_detector() -> ErrorPatternDetector:
    return ErrorPatternDetector()


# ============================================================
# Sidebar Navigation
# ============================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <div style="font-size:2.5rem;">✍️</div>
            <div style="font-size:1.05rem; font-weight:700; 
                        background:linear-gradient(135deg,#58a6ff,#bc8cff);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; margin-top:0.5rem;">
                AI Proofreader
            </div>
            <div style="font-size:0.72rem; color:#8b949e; margin-top:0.25rem;">
                v1.0 · Dual-Model NLP
            </div>
        </div>
        <hr style="border-color:#30363d; margin:0.5rem 0 1rem;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🔍 Real-Time Proofreader", "📊 Architecture & Metrics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.78rem; color:#8b949e;">
            <b style="color:#e6edf3;">Models Used</b><br><br>
            🤖 <b>ML:</b> Random Forest<br>
            🧠 <b>DL:</b> T5 Grammar Correction<br>
            📚 <b>NLP:</b> NLTK + WordNet<br>
            🌐 <b>Multilingual-ready:</b> mBERT / mT5
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("⚙️ Configuration", expanded=False):
            st.markdown(f"""
            <div style="font-size:0.78rem; color:#8b949e; font-family:'JetBrains Mono',monospace;">
            Model: {CONFIG.transformer.model_name}<br>
            Beams: {CONFIG.transformer.num_beams}<br>
            Candidates: {CONFIG.transformer.num_return_sequences}<br>
            Max Length: {CONFIG.transformer.max_input_length}<br>
            Language: {CONFIG.transformer.language}
            </div>
            """, unsafe_allow_html=True)

    return page


# ============================================================
# Helper renderers
# ============================================================

def _badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def _conf_bar(value: float, label: str = "") -> str:
    pct = int(value * 100)
    return f"""
    <div style="margin-bottom:0.4rem;">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.78rem;color:#8b949e;margin-bottom:3px;">
            <span>{label}</span><span>{pct}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%;"></div>
        </div>
    </div>
    """


def _diff_highlight(original: str, corrected: str) -> str:
    """
    Simple token-level diff — highlights changed tokens in the corrected text.
    """
    import difflib
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    parts = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.append(" ".join(corr_words[j1:j2]))
        elif op in ("replace", "insert"):
            span = " ".join(corr_words[j1:j2])
            parts.append(
                f'<mark style="background:rgba(63,185,80,0.25);'
                f'border-radius:3px;padding:0 2px;color:#3fb950;">{span}</mark>'
            )
    return " ".join(parts)


# ============================================================
# PAGE 1 — Real-Time Proofreader
# ============================================================

def page_proofreader() -> None:
    # ── Hero header ──
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🔍 Real-Time AI Proofreader</h1>
        <p class="hero-subtitle">
            Paste any text below — the dual-model pipeline detects grammatical &amp;
            semantic errors and suggests precise corrections in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model loading status ──
    with st.spinner("Loading AI models (first run may take a moment)…"):
        clf = load_classifier()
        preprocessor = load_preprocessor()
        rule_detector = load_rule_detector()

    try:
        corrector = load_corrector()
        transformer_available = True
    except Exception as e:
        corrector = None
        transformer_available = False
        st.warning(
            f"⚠️ Transformer model could not be loaded ({e}). "
            "Grammar classification and rule-based detection will still work."
        )

    # Callback to sync selectbox selection to text_area session state and trigger analysis
    def update_demo_selection():
        choice = st.session_state["demo_selector"]
        if choice != "(select a demo…)":
            st.session_state["user_input"] = choice
            st.session_state["auto_analyze"] = True

    # ── Quick-demo picker ──
    st.markdown("**💡 Try a demo sentence:**")
    demo_col, _ = st.columns([3, 1])
    with demo_col:
        demo_choice = st.selectbox(
            "demo_selector",
            ["(select a demo…)"] + CONFIG.app.demo_texts,
            label_visibility="collapsed",
            key="demo_selector",
            on_change=update_demo_selection,
        )

    # ── Text input ──
    user_text = st.text_area(
        "Enter your text here:",
        height=160,
        max_chars=CONFIG.app.max_text_length,
        placeholder="Paste or type text to proofread…",
        key="user_input",
    )

    char_count = len(user_text)
    st.caption(f"{char_count} / {CONFIG.app.max_text_length} characters")

    analyze_btn = st.button("🚀 Analyze & Correct", use_container_width=False)
    auto_analyze = st.session_state.get("auto_analyze", False)

    if not (analyze_btn or auto_analyze) or not user_text.strip():
        st.info("👆 Enter some text and click **Analyze & Correct** to begin.")
        return

    # Consume the auto_analyze flag
    if auto_analyze:
        st.session_state["auto_analyze"] = False

    # ── Run pipeline ──
    with st.spinner("Analysing…"):
        t0 = time.time()
        preprocessed = preprocessor.preprocess(user_text)
        rule_findings = rule_detector.detect_all(user_text)
        ml_result = clf.predict(user_text)

        correction_result: Dict = {}
        top_correction = user_text
        if transformer_available and corrector:
            correction_result = corrector.correct(user_text)
            top_correction = correction_result.get("top_correction", user_text)
        elapsed = time.time() - t0

    has_error = ml_result["has_error"] or bool(rule_findings)

    # ── Quick-stat bar ──
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{preprocessed['features']['num_words']}</div>
            <div class="metric-label">Words</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(rule_findings)}</div>
            <div class="metric-label">Rule Findings</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        error_color = "var(--accent-red)" if has_error else "var(--accent-green)"
        error_icon = "⚠" if has_error else "✓"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{error_color};">{error_icon}</div>
            <div class="metric-label">ML Verdict</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;">{elapsed:.2f}s</div>
            <div class="metric-label">Analysis Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Two-column: Original vs Corrected ──
    col_orig, col_corr = st.columns(2)

    with col_orig:
        st.markdown("#### 📝 Original Text")
        st.markdown(
            f'<div class="text-panel original">{user_text}</div>',
            unsafe_allow_html=True,
        )

    with col_corr:
        st.markdown("#### ✅ Corrected Text")
        if transformer_available and top_correction.strip() != user_text.strip():
            diff_html = _diff_highlight(user_text, top_correction)
            st.markdown(
                f'<div class="text-panel corrected">{diff_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="text-panel corrected">{top_correction}</div>',
                unsafe_allow_html=True,
            )

    # ── ML Classification ──
    st.markdown("---")
    st.markdown("### 🤖 ML Grammatical Classification")
    cl_col1, cl_col2 = st.columns([1, 2])

    with cl_col1:
        label_badge_kind = "error" if ml_result["has_error"] else "success"
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                {_badge('GRAMMAR ERROR DETECTED' if ml_result['has_error'] else 'NO ERROR DETECTED', label_badge_kind)}
            </div>
            <div style="font-size:1.4rem;font-weight:700;color:#bc8cff;margin-bottom:0.5rem;">
                {ml_result['label']}
            </div>
            <div style="color:#8b949e;font-size:0.85rem;">Predicted Error Category</div>
            """,
            unsafe_allow_html=True,
        )

    with cl_col2:
        st.markdown("**Probability Distribution**")
        probs = ml_result["all_probs"]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs[:5]:
            is_pred = cls == ml_result["label"]
            label_str = f"{'▶ ' if is_pred else ''}{cls}"
            st.markdown(_conf_bar(prob, label_str), unsafe_allow_html=True)

    # ── Rule-based findings ──
    if rule_findings:
        st.markdown("---")
        st.markdown(f"### 📋 Rule-Based Findings  ({len(rule_findings)} issues)")
        for i, finding in enumerate(rule_findings):
            icon = "🔴" if "Error" in finding["error_type"] else "🟡"
            sugg = (
                f"  →  **Suggestion:** `{finding['suggestion']}`"
                if finding.get("suggestion")
                else ""
            )
            st.markdown(
                f"""
                <div class="finding-item">
                    {icon} <b>{finding['error_type']}</b><br>
                    <span style="color:#8b949e;font-size:0.85rem;">
                        {finding['description']}{(' → Suggestion: ' + finding['suggestion']) if finding.get('suggestion') else ''}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Transformer candidates ──
    if transformer_available and correction_result.get("corrections"):
        st.markdown("---")
        st.markdown("### 🧠 Transformer Correction Candidates")
        st.caption(f"Model: `{correction_result.get('model_name', 'N/A')}`")
        cands = correction_result["corrections"]
        for idx, cand in enumerate(cands, 1):
            icon = "🥇" if idx == 1 else f"#{idx}"
            tag = " ← Best" if idx == 1 else ""
            st.markdown(
                f"""
                <div style="background:{'rgba(63,185,80,0.08)' if idx==1 else 'var(--bg-card)'};
                            border:1px solid {'var(--accent-green)' if idx==1 else 'var(--border)'};
                            border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.5rem;
                            font-family:'JetBrains Mono',monospace; font-size:0.88rem;">
                    <span style="color:#8b949e;">{icon}{tag}</span><br>
                    {cand}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── NLP Token analysis ──
    with st.expander("🔬 Deep NLP Token Analysis", expanded=False):
        tokens = preprocessed["tokens"]
        pos_tags = preprocessed["pos_tags"]
        lemmas = preprocessed["lemmas"]

        df_tokens = pd.DataFrame({
            "Token": tokens,
            "POS Tag": [tag for _, tag in pos_tags],
            "Lemma": lemmas,
        })
        st.dataframe(
            df_tokens,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Feature Vector (ML Input)**")
        feats = preprocessed["features"]
        feat_df = pd.DataFrame(
            list(feats.items()), columns=["Feature", "Value"]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2 — Architecture & Metrics
# ============================================================

def page_architecture() -> None:
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">📊 Architecture & Metrics</h1>
        <p class="hero-subtitle">
            Visual breakdown of the dual-model NLP pipeline — from raw text to
            corrected output.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline flow diagram ──
    st.markdown("### 🔄 Pipeline Architecture")

    fig_pipeline = go.Figure()

    nodes = [
        ("Raw Text\nInput", 0.05, 0.5, "#58a6ff", "rgba(88, 166, 255, 0.13)"),
        ("NLP\nPreprocessor\n(NLTK)", 0.22, 0.5, "#bc8cff", "rgba(188, 140, 255, 0.13)"),
        ("Rule-Based\nDetector", 0.40, 0.75, "#ffa657", "rgba(255, 166, 87, 0.13)"),
        ("ML Classifier\n(Random Forest)", 0.40, 0.25, "#f85149", "rgba(248, 81, 73, 0.13)"),
        ("Transformer\n(T5 / mT5)", 0.60, 0.5, "#3fb950", "rgba(63, 185, 80, 0.13)"),
        ("Result\nAggregator", 0.78, 0.5, "#58a6ff", "rgba(88, 166, 255, 0.13)"),
        ("Corrected\nOutput", 0.95, 0.5, "#3fb950", "rgba(63, 185, 80, 0.13)"),
    ]

    edges = [
        (0, 1), (1, 2), (1, 3), (2, 5), (3, 5),
        (1, 4), (4, 5), (5, 6),
    ]

    for (label, x, y, color, rgba_color) in nodes:
        fig_pipeline.add_shape(
            type="rect",
            x0=x - 0.08, y0=y - 0.15, x1=x + 0.08, y1=y + 0.15,
            fillcolor=rgba_color,
            line=dict(color=color, width=2),
        )
        fig_pipeline.add_annotation(
            x=x, y=y, text=label,
            showarrow=False,
            font=dict(color=color, size=11, family="Inter"),
            align="center",
        )

    for (a, b) in edges:
        x0, y0 = nodes[a][1], nodes[a][2]
        x1, y1 = nodes[b][1], nodes[b][2]
        fig_pipeline.add_annotation(
            ax=x0 + 0.08, ay=y0, x=x1 - 0.08, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1,
            arrowcolor="#8b949e", arrowwidth=1.5,
        )

    fig_pipeline.update_layout(
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(range=[0, 1], visible=False),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pipeline, use_container_width=True)

    # ── Architecture description cards ──
    st.markdown("---")
    st.markdown("### 🧱 Component Breakdown")

    arch_cols = st.columns(3)
    components = [
        {
            "icon": "🔤",
            "name": "NLP Preprocessing Engine",
            "color": "#bc8cff",
            "items": [
                "Word tokenisation (NLTK punkt)",
                "Part-of-Speech tagging (Penn Treebank)",
                "WordNet lemmatisation",
                "14-dimensional feature extraction",
                "Rule-based error detection",
            ],
        },
        {
            "icon": "🌲",
            "name": "ML Grammatical Classifier",
            "color": "#f85149",
            "items": [
                "Random Forest (200 estimators)",
                "8 error categories",
                "Trained on curated synthetic corpus",
                "Confidence probability distribution",
                "Fallback SVM mode (config.py)",
            ],
        },
        {
            "icon": "🧠",
            "name": "Transformer Semantic Corrector",
            "color": "#3fb950",
            "items": [
                "T5 fine-tuned on grammar correction",
                "Beam search (4 beams, 3 candidates)",
                "Contextual semantic understanding",
                "Token-level diff highlighting",
                "Multilingual-ready (swap in config.py)",
            ],
        },
    ]

    for col, comp in zip(arch_cols, components):
        with col:
            items_html = "".join(
                f'<li style="margin-bottom:0.3rem;">{item}</li>'
                for item in comp["items"]
            )
            st.markdown(
                f"""
                <div style="background:var(--bg-card);border:1px solid {comp['color']}44;
                            border-top:3px solid {comp['color']};border-radius:10px;
                            padding:1.25rem;min-height:220px;">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem;">{comp['icon']}</div>
                    <div style="font-size:0.95rem;font-weight:700;color:{comp['color']};
                                margin-bottom:0.75rem;">{comp['name']}</div>
                    <ul style="color:#8b949e;font-size:0.82rem;padding-left:1.1rem;margin:0;">
                        {items_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Classifier performance (synthetic) ──
    st.markdown("---")
    st.markdown("### 📈 Classifier Performance (Synthetic Dataset)")

    with st.spinner("Computing metrics…"):
        clf = load_classifier()
        train_info = clf.train()

    report = train_info["report"]
    label_classes = [
        c for c in train_info["classes"] if c not in ("accuracy", "macro avg", "weighted avg")
    ]

    precision_vals = [report[c]["precision"] for c in label_classes]
    recall_vals = [report[c]["recall"] for c in label_classes]
    f1_vals = [report[c]["f1-score"] for c in label_classes]

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        name="Precision", x=label_classes, y=precision_vals,
        marker_color="#58a6ff", marker_line_width=0,
    ))
    fig_perf.add_trace(go.Bar(
        name="Recall", x=label_classes, y=recall_vals,
        marker_color="#bc8cff", marker_line_width=0,
    ))
    fig_perf.add_trace(go.Bar(
        name="F1-Score", x=label_classes, y=f1_vals,
        marker_color="#3fb950", marker_line_width=0,
    ))
    fig_perf.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        font=dict(color="#8b949e", family="Inter"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="#21262d",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#21262d",
            range=[0, 1.05],
            title="Score",
        ),
        margin=dict(l=40, r=10, t=10, b=60),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── Accuracy summary row ──
    acc = report.get("accuracy", 0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
    w_f1 = report.get("weighted avg", {}).get("f1-score", 0)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (f"{acc:.1%}", "Overall Accuracy"),
        (f"{macro_f1:.1%}", "Macro F1"),
        (f"{w_f1:.1%}", "Weighted F1"),
        (str(len(label_classes)), "Error Categories"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Feature importance ──
    st.markdown("---")
    st.markdown("### 🌲 Random Forest Feature Importance")

    if hasattr(clf._model, "feature_importances_"):
        feature_names = [
            "num_tokens", "num_words", "type_token_ratio",
            "avg_word_length", "avg_sent_length",
            "3rd_sg_verbs", "base_verbs", "modals",
            "past_tense", "aux_verbs",
            "punct_count", "oov_count",
            "repeated_bigrams", "sentence_count",
        ]
        importances = clf._model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#1c2128", "#58a6ff", "#bc8cff"],
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            font=dict(color="#8b949e", family="Inter"),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Multilingual roadmap ──
    st.markdown("---")
    st.markdown("### 🌍 Multilingual Scalability Roadmap")

    ml_cols = st.columns(2)
    with ml_cols[0]:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:1.25rem;">
            <div style="font-size:1rem;font-weight:700;color:#58a6ff;margin-bottom:1rem;">
                Current → Multilingual Swap
            </div>
            <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
                <thead>
                    <tr style="color:#58a6ff;border-bottom:1px solid #30363d;">
                        <th style="text-align:left;padding:4px 8px;">Language</th>
                        <th style="text-align:left;padding:4px 8px;">Model</th>
                        <th style="text-align:left;padding:4px 8px;">Config Key</th>
                    </tr>
                </thead>
                <tbody style="color:#8b949e;">
                    <tr><td style="padding:4px 8px;">English ✅</td><td>vennify/t5-base-grammar-correction</td><td>model_name</td></tr>
                    <tr><td style="padding:4px 8px;">Multilingual</td><td>google/mt5-base</td><td>model_name</td></tr>
                    <tr><td style="padding:4px 8px;">mBERT</td><td>bert-base-multilingual-cased</td><td>model_name</td></tr>
                    <tr><td style="padding:4px 8px;">French</td><td>moussaKam/barthez-orangesum-abstract</td><td>model_name</td></tr>
                    <tr><td style="padding:4px 8px;">German</td><td>Helsinki-NLP/opus-mt-de-en</td><td>model_name</td></tr>
                    <tr><td style="padding:4px 8px;">Spanish</td><td>Helsinki-NLP/opus-mt-es-en</td><td>model_name</td></tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with ml_cols[1]:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid #30363d;border-radius:10px;padding:1.25rem;height:100%;">
            <div style="font-size:1rem;font-weight:700;color:#3fb950;margin-bottom:0.75rem;">
                How to Swap Models
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                        background:#0d1117;border-radius:8px;padding:1rem;
                        border:1px solid #30363d;color:#8b949e;white-space:pre;">
<span style="color:#58a6ff;"># config.py</span>
<span style="color:#3fb950;">@dataclass</span>
<span style="color:#bc8cff;">class</span> TransformerConfig:
    model_name: str = <span style="color:#ffa657;">"google/mt5-base"</span>
    language:   str = <span style="color:#ffa657;">"fr"</span>  <span style="color:#8b949e;"># French</span>

<span style="color:#8b949e;"># No other code changes needed.</span>
<span style="color:#8b949e;"># The SemanticCorrector loads</span>
<span style="color:#8b949e;"># the model transparently.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# Main entrypoint
# ============================================================

def main() -> None:
    page = render_sidebar()

    if page == "🔍 Real-Time Proofreader":
        page_proofreader()
    elif page == "📊 Architecture & Metrics":
        page_architecture()


if __name__ == "__main__":
    main()
