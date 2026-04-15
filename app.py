import streamlit as st

st.set_page_config(
    page_title="Mithra | Clinical PGx Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global dark theme CSS - (Generated with AI)
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp {
    background: #0f172a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Widgets ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #6366f1 !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: white !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px;
}

/* ── Info / Warning / Success ── */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr {
    border-color: #334155 !important;
}

/* ── Custom card ── */
.mithra-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.mithra-card-accent {
    background: linear-gradient(135deg, #1e293b, #1a1f35);
    border: 1px solid #6366f1;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.mithra-badge-green {
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.mithra-badge-yellow {
    background: #1c1408;
    color: #fbbf24;
    border: 1px solid #92400e;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.mithra-badge-red {
    background: #1c0a0a;
    color: #f87171;
    border: 1px solid #991b1b;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.mithra-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
}
.mithra-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
.gene-chip {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #1e40af;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: monospace;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# Session state defaults
if "disease"         not in st.session_state: st.session_state.disease         = None
if "disease_config"  not in st.session_state: st.session_state.disease_config  = None
if "agent"           not in st.session_state: st.session_state.agent           = None
if "env"             not in st.session_state: st.session_state.env             = None
if "trained"         not in st.session_state: st.session_state.trained         = False
if "patient_info"    not in st.session_state: st.session_state.patient_info    = {}
if "llm_result"      not in st.session_state: st.session_state.llm_result      = None
if "rl_result"       not in st.session_state: st.session_state.rl_result       = None
if "all_patients"    not in st.session_state: st.session_state.all_patients    = []
if "baseline_result" not in st.session_state: st.session_state.baseline_result = None
if "train_metrics"   not in st.session_state: st.session_state.train_metrics   = None

# Sidebar navigation logo
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0; border-bottom: 1px solid #334155; margin-bottom:1rem;">
        <div style="font-size:1.5rem; font-weight:800; color:#6366f1; letter-spacing:-0.5px;">🧬 MITHRA</div>
        <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">
            Clinical PGx Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">
    Navigation
    </div>
    """, unsafe_allow_html=True)

# Home page
st.markdown("""
<div style="text-align:center; padding: 3rem 0 2rem 0;">
    <div style="font-size:3rem; margin-bottom:0.5rem;">🧬</div>
    <div style="font-size:2.5rem; font-weight:800; color:#e2e8f0; letter-spacing:-1px;">
        MITHRA
    </div>
    <div style="font-size:1.1rem; color:#6366f1; font-weight:600; margin-bottom:0.5rem;">
        Clinical Pharmacogenomics Intelligence Platform
    </div>
    <div style="font-size:0.95rem; color:#64748b; max-width:600px; margin:0 auto 2rem auto;">
        Combining Bootstrapped Deep Reinforcement Learning with Large Language Models
        to deliver precision medicine treatment decisions grounded in PharmaGKB evidence.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="mithra-card">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">⚡</div>
        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.3rem;">Bootstrapped RL</div>
        <div style="font-size:0.82rem; color:#94a3b8;">
            Multi-head Deep Q-Network trained on pharmacogenomic patient profiles.
            Bootstrap exploration ensures confident, uncertainty-aware arm assignment.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="mithra-card">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">🤖</div>
        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.3rem;">LLM-Powered Analysis</div>
        <div style="font-size:0.82rem; color:#94a3b8;">
            GPT-4o-mini with RAG over PharmaGKB delivers structured ADME reasoning
            for each drug-gene interaction — explainable, evidence-grounded.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="mithra-card">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">📊</div>
        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.3rem;">Production-Ready</div>
        <div style="font-size:0.82rem; color:#94a3b8;">
            Quantized models, head pruning, token budgeting — optimised for
            deployment on standard clinical workstations without GPU.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.82rem;">
    Navigate using the <strong style="color:#6366f1;">sidebar pages</strong> to set up a trial,
    enter a patient, train the agent, and generate a clinical report.
</div>
""", unsafe_allow_html=True)

# Status bar
if st.session_state.disease:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Disease", st.session_state.disease.split("/")[0].strip()[:25])
    drugs = list(st.session_state.disease_config.get("drugs",{}).keys()) if st.session_state.disease_config else []
    c2.metric("Treatment Arms", len(drugs))
    c3.metric("Agent Trained", "✅ Yes" if st.session_state.trained else "⏳ No")
    c4.metric("Patients Analysed", len(st.session_state.all_patients))
