import streamlit as st
import json
from config import GENE_PHENOTYPES, GENE_DEFAULT_PHENOTYPE, SYNTHETIC_PATIENTS, DISEASE_CONFIG

#  Guard 
if not st.session_state.get("disease"):
    st.warning("⚠️ Please complete **Trial Setup** first.")
    st.stop()

disease = st.session_state.disease
cfg     = st.session_state.disease_config
genes   = st.session_state.get("selected_genes", cfg["primary_genes"])

st.markdown(f"""
<div class="mithra-header">👤 Patient Profile</div>
<div class="mithra-sub">Enter patient genetic and clinical data. Use a synthetic template to explore system capabilities.</div>
""", unsafe_allow_html=True)

#  Synthetic patient loader 
st.markdown("### ⚡ Quick Load — Synthetic Patient Templates")
st.caption("Pre-built clinically realistic patients. Select one to auto-fill all fields, then edit as needed.")

templates = SYNTHETIC_PATIENTS.get(disease, [])
if not templates:
    # Generate generic templates for diseases without specific ones
    templates = [
        {"name": "Patient A — Typical Profile",
         "age": 45, "weight": 75, "sex": "Male",
         "diagnosis_notes": f"Standard presentation of {disease}. No significant comorbidities.",
         "genes": {g: GENE_DEFAULT_PHENOTYPE.get(g, "") for g in genes}},
    ]

template_names = ["— Manual Entry —"] + [t["name"] for t in templates]
selected_template_name = st.selectbox("Select a patient template", template_names,
                                       help="Templates represent realistic pharmacogenomic profiles.")

if selected_template_name != "— Manual Entry —":
    template = next(t for t in templates if t["name"] == selected_template_name)
else:
    template = None

st.markdown("---")

#  Patient Demographics 
st.markdown("### 📋 Demographics & Clinical Notes")

dcol1, dcol2, dcol3, dcol4 = st.columns([2,1,1,1])

with dcol1:
    patient_name = st.text_input(
        "Patient ID / Name",
        value=template["name"] if template else "New Patient",
        placeholder="e.g. PAT-2024-001",
    )
with dcol2:
    age = st.number_input("Age", min_value=18, max_value=100,
                           value=template["age"] if template else 45)
with dcol3:
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200,
                              value=template["weight"] if template else 70)
with dcol4:
    sex = st.selectbox("Sex", ["Male","Female","Other"],
                        index=["Male","Female","Other"].index(
                            template["sex"] if template else "Male"))

diagnosis_notes = st.text_area(
    "Clinical Notes",
    value=template["diagnosis_notes"] if template else f"Patient with {disease}. Refer to genetic panel below.",
    height=100,
    help="Describe clinical presentation, prior treatment history, relevant comorbidities.",
)

st.markdown("---")

#  Gene Profile Entry 
st.markdown("### 🧬 Pharmacogenomic Panel")
st.caption("Metabolizer phenotypes detected from genetic panel. Select the patient's status for each gene.")

gene_values = {}
drug_gene_relevance = {}
# Build drug→gene map for relevance indicators
for d_name, d_info in cfg["drugs"].items():
    for g in d_info["genes"]:
        drug_gene_relevance.setdefault(g, []).append(d_name)

gcols = st.columns(2)
for i, gene in enumerate(genes):
    col = gcols[i % 2]
    with col:
        phenotypes = GENE_PHENOTYPES.get(gene, ["Normal"])

        # Get default: from template if available, else population default
        if template and gene in template.get("genes", {}):
            default_status = template["genes"][gene]
        else:
            default_status = GENE_DEFAULT_PHENOTYPE.get(gene, phenotypes[0])

        default_idx = phenotypes.index(default_status) if default_status in phenotypes else 0

        relevant_drugs = drug_gene_relevance.get(gene, [])
        drugs_str = ", ".join(relevant_drugs[:3])
        if len(relevant_drugs) > 3:
            drugs_str += f" +{len(relevant_drugs)-3}"

        status = st.selectbox(
            f"**{gene}**",
            options=phenotypes,
            index=default_idx,
            key=f"gene_input_{gene}",
            help=f"Affects: {drugs_str}",
        )
        gene_values[gene] = status

        # Show clinical implication badge
        risk_map = {
            "Poor Metabolizer"               : ("⚠️ High Risk", "mithra-badge-red"),
            "Poor Metabolizer (No Function)" : ("⚠️ High Risk", "mithra-badge-red"),
            "Ultrarapid Metabolizer"         : ("⚡ Ultra Rapid", "mithra-badge-yellow"),
            "Intermediate Metabolizer"       : ("⚡ Intermediate", "mithra-badge-yellow"),
            "*57:01 Present (Positive)"      : ("🚨 Contraindicated", "mithra-badge-red"),
            "*58:01 Present (Positive)"      : ("🚨 Contraindicated", "mithra-badge-red"),
            "*31:01 Present (Positive)"      : ("⚠️ Risk", "mithra-badge-red"),
            "Deficient Activity"             : ("⚠️ Deficient", "mithra-badge-red"),
            "Low Dose Required"              : ("💊 Low Dose", "mithra-badge-yellow"),
            "High Dose Required"             : ("💊 High Dose", "mithra-badge-yellow"),
        }
        if status in risk_map:
            label, badge_class = risk_map[status]
            st.markdown(f'<span class="{badge_class}">{label}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="mithra-badge-green">✓ Standard</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

#  Profile preview 
st.markdown("### 👁️ Profile Preview")
with st.expander("View encoded patient state", expanded=False):
    from core.environment import patientStateEncoder, geneStateDim
    state_vec = patientStateEncoder(gene_values, genes)
    col1, col2 = st.columns(2)
    with col1:
        st.json({g: gene_values[g] for g in genes})
    with col2:
        st.caption("Encoded state vector (input to RL agent)")
        st.code(str(state_vec.round(2).tolist()), language=None)
        st.caption(f"State dimension: {len(state_vec)}")

#  Submit 
st.markdown("---")

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    submit = st.button("🧬 Analyse Patient", type="primary", use_container_width=True,
                        help="Run LLM pharmacogenomic analysis and RL arm recommendation.")
with col_btn2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state.llm_result = None
    st.session_state.rl_result  = None
    st.rerun()

if submit:
    patient_info = {
        "name"            : patient_name,
        "age"             : age,
        "weight"          : weight,
        "sex"             : sex,
        "diagnosis_notes" : diagnosis_notes,
        "genes"           : gene_values,
        "disease"         : disease,
    }
    st.session_state.patient_info = patient_info
    st.success("✅ Patient profile saved. Navigate to **Train Agent** or **Patient Report** to see results.")
    st.info("💡 Tip: Train the RL agent first for the most accurate arm recommendation.")
