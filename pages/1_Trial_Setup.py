import streamlit as st
from config import DISEASE_CONFIG, RL_CONFIG, GENE_PHENOTYPES
from core.data_loader import loadPharmGKB, diseaseSummary

st.markdown("""
<div class="mithra-header">⚙️ Trial Configuration</div>
<div class="mithra-sub">Select a disease area — drugs, genes, and parameters auto-populate from PharmaGKB evidence.</div>
""", unsafe_allow_html=True)

#  Disease selector 
st.markdown("### 🏥 Disease Area")

disease_options = list(DISEASE_CONFIG.keys())
current_idx = disease_options.index(st.session_state.disease) if st.session_state.disease in disease_options else 0

selected_disease = st.selectbox(
    "Select disease area",
    options=disease_options,
    index=current_idx,
    help="The system will automatically load relevant drugs and genetic markers from PharmaGKB.",
)

cfg = DISEASE_CONFIG[selected_disease]

# Disease info card
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown(f"""
    <div class="mithra-card">
        <div style="font-weight:600; color:#6366f1; margin-bottom:0.4rem;">{selected_disease}</div>
        <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.5rem;">{cfg['description']}</div>
        <div style="font-size:0.75rem; color:#475569;">ICD-10: {cfg['icd10']}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.metric("Available Drugs", len(cfg["drugs"]))
with col3:
    st.metric("Key Genes", len(cfg["primary_genes"]))

st.markdown("---")

#  Drug arm selection 
st.markdown("### 💊 Treatment Arms")
st.caption("All drugs below have CPIC/PharmaGKB dosing guidelines. Select which to include in the trial.")

drug_cols = st.columns(min(len(cfg["drugs"]), 3))
selected_drugs = []

for i, (drug_name, drug_info) in enumerate(cfg["drugs"].items()):
    col = drug_cols[i % len(drug_cols)]
    with col:
        genes_html = "".join([f'<span class="gene-chip">{g}</span>' for g in drug_info["genes"]])
        checked = st.checkbox(
            drug_name,
            value=True,
            key=f"drug_{drug_name}",
            help=f"Class: {drug_info['class']} | Genes: {', '.join(drug_info['genes'])}",
        )
        st.markdown(f'<div style="margin-top:-8px; margin-bottom:8px;">{genes_html}</div>', unsafe_allow_html=True)
        if checked:
            selected_drugs.append(drug_name)

if len(selected_drugs) < 2:
    st.error("⚠️ Please select at least 2 treatment arms for a valid trial.")

st.markdown("---")

#  Gene panel 
st.markdown("### 🧬 Genetic Markers")
st.caption("Primary genes are pre-selected based on CPIC guidelines for the chosen disease and drugs.")

all_genes_for_disease = set()
for d in selected_drugs:
    if d in cfg["drugs"]:
        all_genes_for_disease.update(cfg["drugs"][d]["genes"])
all_genes_for_disease.update(cfg.get("secondary_genes", []))

selected_genes = []
gene_col1, gene_col2, gene_col3 = st.columns(3)
all_genes_list = sorted(all_genes_for_disease)

for i, gene in enumerate(all_genes_list):
    col = [gene_col1, gene_col2, gene_col3][i % 3]
    with col:
        is_primary = gene in cfg["primary_genes"]
        checked = st.checkbox(
            gene,
            value=True,
            key=f"gene_{gene}",
            help=f"{'★ Primary gene' if is_primary else 'Secondary gene'} for {selected_disease}",
        )
        if checked:
            selected_genes.append(gene)
        phenotypes = GENE_PHENOTYPES.get(gene, [])
        st.caption(f"{len(phenotypes)} phenotypes")

st.markdown("---")

#  RL Hyperparameters 
st.markdown("### 🤖 RL Agent Configuration")
st.caption("Pre-tuned for CPU execution on standard workstations. Adjust only if needed.")

# Default values (used if expander is collapsed / not interacted with)
n_episodes       = 400
n_heads          = 4
hidden_size      = 128
lr               = 5e-4
batch_size       = 64
gamma            = 0.95

with st.expander("⚙️ Advanced Hyperparameters", expanded=False):
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        n_episodes  = st.slider("Training Episodes", 100, 1000, 400, 50,
                                help="More episodes = better convergence but longer training.")
        n_heads     = st.slider("Bootstrap Heads (K)", 2, 8, 4,
                                help="More heads = richer exploration but more compute.")
    with hcol2:
        hidden_size = st.select_slider("Hidden Layer Size", [64, 128, 256, 512], 128,
                                       help="Larger = more capacity but slower on CPU.")
        lr          = st.select_slider("Learning Rate",
                                       [1e-4, 5e-4, 1e-3, 5e-3], 5e-4,
                                       format_func=lambda x: f"{x:.0e}")
    with hcol3:
        batch_size  = st.select_slider("Batch Size", [32, 64, 128], 64)
        gamma       = st.slider("Discount Factor (γ)", 0.8, 0.99, 0.95, 0.01)

custom_rl_config = {
    **RL_CONFIG,
    "num_heads"    : n_heads,
    "hidden_size"  : hidden_size,
    "learning_rate": lr,
    "batch_size"   : batch_size,
    "gamma"        : gamma,
}

#  Summary & Confirm 
st.markdown("---")
st.markdown("### ✅ Trial Summary")

if selected_drugs and selected_genes:
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Disease",        selected_disease.split("/")[0].strip()[:20])
    sc2.metric("Treatment Arms", len(selected_drugs))
    sc3.metric("Genetic Markers",len(selected_genes))
    sc4.metric("Bootstrap Heads",custom_rl_config["num_heads"])

    st.markdown(f"""
    <div class="mithra-card-accent">
        <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.5rem;">Selected Drugs</div>
        <div style="font-weight:600; color:#e2e8f0;">{' · '.join(selected_drugs)}</div>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.5rem; margin-bottom:0.3rem;">Active Genes</div>
        <div>{''.join([f'<span class="gene-chip">{g}</span>' for g in selected_genes])}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Initialise Trial", type="primary", use_container_width=True):
        if len(selected_drugs) < 2:
            st.error("Select at least 2 drugs.")
        elif len(selected_genes) < 1:
            st.error("Select at least 1 gene.")
        else:
            # Build a filtered disease config with only selected drugs/genes
            filtered_cfg = {
                **cfg,
                "drugs": {d: cfg["drugs"][d] for d in selected_drugs if d in cfg["drugs"]},
                "primary_genes": [g for g in cfg["primary_genes"] if g in selected_genes],
            }

            st.session_state.disease        = selected_disease
            st.session_state.disease_config = filtered_cfg
            st.session_state.selected_genes = selected_genes
            st.session_state.selected_drugs = selected_drugs
            st.session_state.rl_config      = custom_rl_config
            st.session_state.n_episodes     = n_episodes
            st.session_state.trained        = False
            st.session_state.agent          = None
            st.session_state.env            = None
            st.session_state.llm_result     = None
            st.session_state.rl_result      = None
            st.session_state.all_patients   = []

            st.success(f"✅ Trial initialised! Navigate to **Patient Input** to enter a patient.")
            st.balloons()
else:
    st.warning("Select at least 2 drugs and 1 gene to continue.")
