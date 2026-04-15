import streamlit as st
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from core.data_loader import loadPharmGKB

#  Guard 
if not st.session_state.get("disease"):
    st.warning("⚠️ Please complete **Trial Setup** first.")
    st.stop()
if not st.session_state.get("patient_info"):
    st.warning("⚠️ Please enter a patient in **Patient Input** first.")
    st.stop()

disease      = st.session_state.disease
cfg          = st.session_state.disease_config
genes        = st.session_state.get("selected_genes", cfg["primary_genes"])
drugs        = list(cfg["drugs"].keys())
patient_info = st.session_state.patient_info
agent        = st.session_state.get("agent")
trained      = st.session_state.get("trained", False)

st.markdown(f"""
<div class="mithra-header">📋 Clinical Report</div>
<div class="mithra-sub">AI-powered pharmacogenomic analysis and treatment recommendation for {patient_info.get('name','Patient')}.</div>
""", unsafe_allow_html=True)

#  Patient header card 
genes_html = "".join([f'<span class="gene-chip">{g}: {patient_info["genes"].get(g,"-")}</span>'
                       for g in genes])
st.markdown(f"""
<div class="mithra-card-accent">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.3rem;">
                {patient_info.get('name','Anonymous')}
            </div>
            <div style="font-size:0.82rem; color:#64748b;">
                Age {patient_info.get('age','-')} · {patient_info.get('sex','-')} · {disease}
            </div>
            <div style="margin-top:0.5rem;">{genes_html}</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.72rem; color:#64748b;">Clinical Notes</div>
            <div style="font-size:0.82rem; color:#94a3b8; max-width:300px;">
                {patient_info.get('diagnosis_notes','')[:120]}{'...' if len(patient_info.get('diagnosis_notes',''))>120 else ''}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

#  Run Analysis 
col_llm, col_rl = st.columns([3,1])

with col_rl:
    st.markdown("### 🤖 RL Decision")
    if trained and agent:
        from core.environment import patientStateEncoder
        state      = patientStateEncoder(patient_info["genes"], genes)
        confidence = agent.get_action_confidence(state)
        best_idx   = confidence["best_action"]
        best_drug  = drugs[best_idx]
        best_prob  = confidence["probabilities"][best_idx] * 100
        head_votes = confidence["head_votes"]
        certainty  = confidence["certainty"]

        rl_result = {
            "recommended_drug": best_drug,
            "confidence_pct"  : best_prob,
            "head_votes"      : head_votes,
            "certainty"       : certainty,
            "probabilities"   : confidence["probabilities"],
        }
        st.session_state.rl_result = rl_result

        # Recommendation badge
        color = "#4ade80" if certainty > 0.6 else "#fbbf24" if certainty > 0.3 else "#f87171"
        st.markdown(f"""
        <div class="mithra-card" style="text-align:center; border-color:{color};">
            <div style="font-size:0.72rem; color:#64748b; margin-bottom:0.3rem;">RECOMMENDED</div>
            <div style="font-size:1.4rem; font-weight:800; color:{color}; margin-bottom:0.3rem;">
                {best_drug}
            </div>
            <div style="font-size:0.82rem; color:#94a3b8;">Confidence</div>
            <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">{best_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Head votes
        st.markdown("**Bootstrap Head Votes**")
        for drug_n, votes in head_votes.items():
            bar_w = int(votes / agent.num_heads * 100)
            col = "#6366f1" if drug_n == best_drug else "#334155"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                <div style="font-size:0.75rem; color:#94a3b8; width:80px; text-align:right;">{drug_n[:10]}</div>
                <div style="flex:1; background:#1e293b; border-radius:4px; height:16px; overflow:hidden;">
                    <div style="width:{bar_w}%; height:100%; background:{col}; border-radius:4px;"></div>
                </div>
                <div style="font-size:0.75rem; color:#e2e8f0; width:20px;">{votes}</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability chart
        fig_prob = go.Figure(go.Bar(
            x=drugs, y=[p*100 for p in confidence["probabilities"]],
            marker_color=["#6366f1" if i==best_idx else "#334155" for i in range(len(drugs))],
            text=[f"{p*100:.1f}%" for p in confidence["probabilities"]],
            textposition="outside",
        ))
        fig_prob.update_layout(
            height=200, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#94a3b8", size=9), showlegend=False,
            margin=dict(t=10, b=30, l=10, r=10),
            yaxis=dict(range=[0, 105], gridcolor="#334155"),
            xaxis=dict(linecolor="#334155"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    else:
        st.info("Train the RL agent on the **Train Agent** page for a data-driven recommendation.")
        rl_result = {"recommended_drug": drugs[0], "confidence_pct": 0,
                     "head_votes": {}, "certainty": 0, "probabilities": [1/len(drugs)]*len(drugs)}
        st.session_state.rl_result = rl_result

with col_llm:
    st.markdown("### 🧬 LLM Pharmacogenomic Analysis")

    run_analysis = st.button("🔍 Run LLM Analysis", type="primary", use_container_width=True,
                              help="GPT-4o-mini analyses gene-drug interactions via RAG over PharmaGKB")

    if run_analysis or st.session_state.get("llm_result"):
        if run_analysis:
            st.session_state.llm_result = None

        if not st.session_state.get("llm_result"):
            from core.llm_pipeline import analysePatient_LLM
            _, _, _, rel_df = loadPharmGKB()

            with st.spinner("🤖 LLM reasoning over PharmaGKB evidence..."):
                # Show streaming token budget
                token_info = st.empty()
                response_box = st.empty()
                full_text = [""]

                def stream_cb(delta):
                    full_text[0] += delta
                    response_box.markdown(f"""
                    <div style="font-family:monospace; font-size:0.75rem; color:#94a3b8;
                                background:#1e293b; border-radius:8px; padding:0.8rem;
                                max-height:150px; overflow-y:auto; border:1px solid #334155;">
                        {full_text[0][-300:]}
                    </div>
                    """, unsafe_allow_html=True)

                result = analysePatient_LLM(
                    gene_profile   = patient_info["genes"],
                    drug_names     = drugs,
                    disease        = disease,
                    patient_notes  = patient_info.get("diagnosis_notes",""),
                    rel_df         = rel_df,
                    stream_callback= stream_cb,
                )
                response_box.empty()
                token_info.empty()
                st.session_state.llm_result = result

        result = st.session_state.llm_result

        # Patient summary
        st.markdown(f"""
        <div class="mithra-card">
            <div style="font-size:0.72rem; color:#64748b; margin-bottom:0.3rem; text-transform:uppercase; letter-spacing:0.05em;">
                Patient Summary
            </div>
            <div style="font-size:0.88rem; color:#e2e8f0; line-height:1.6;">
                {result.get('patient_summary', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Reasoning
        st.markdown(f"""
        <div class="mithra-card-accent">
            <div style="font-size:0.72rem; color:#6366f1; margin-bottom:0.3rem; text-transform:uppercase; letter-spacing:0.05em;">
                Clinical Reasoning
            </div>
            <div style="font-size:0.85rem; color:#e2e8f0; line-height:1.6;">
                {result.get('reasoning', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Warnings
        warnings = result.get("warnings", [])
        if warnings:
            for w in warnings:
                st.error(f"⚠️ {w}")

        # Token budget
        tokens = result.get("_token_estimate", 0)
        model  = result.get("_model","")
        st.caption(f"Model: {model} · Estimated tokens: ~{tokens:,} · RAG: PharmaGKB relationships")

#  Per-drug ADME breakdown 
if st.session_state.get("llm_result"):
    st.markdown("---")
    st.markdown("### 💊 Per-Drug Analysis")

    result       = st.session_state.llm_result
    drugs_data   = result.get("drugs", {})
    rl_result    = st.session_state.get("rl_result", {})
    top_drug     = rl_result.get("recommended_drug","")

    if drugs_data:
        # Sort by rank
        sorted_drugs = sorted(drugs_data.items(), key=lambda x: x[1].get("rank", 99))

        for drug_name, info in sorted_drugs:
            rec     = info.get("recommendation","")
            conf    = info.get("confidence","")
            is_top  = drug_name == top_drug

            col_map = {"Recommended": ("#052e16","#4ade80","#166534"),
                       "Use with Caution": ("#1c1408","#fbbf24","#92400e"),
                       "Avoid": ("#1c0a0a","#f87171","#991b1b")}
            bg, txt, border = col_map.get(rec, ("#1e293b","#94a3b8","#334155"))

            rl_badge = f'<span style="background:#1e3a5f;color:#60a5fa;border:1px solid #1e40af;border-radius:4px;padding:1px 6px;font-size:0.7rem;margin-left:6px;">🤖 RL Top Pick</span>' if is_top else ""

            with st.expander(f"{'★ ' if is_top else ''}{drug_name}  -  {rec}  ({conf} confidence){' ← RL Recommended' if is_top else ''}", expanded=is_top):
                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {border}; border-radius:8px; padding:0.8rem; margin-bottom:0.5rem;">
                    <span style="color:{txt}; font-weight:700; font-size:0.9rem;">{rec}</span>
                    <span style="color:#94a3b8; font-size:0.8rem; margin-left:1rem;">Confidence: {conf}</span>
                    {rl_badge}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="font-size:0.78rem; color:#64748b; margin-bottom:0.3rem; text-transform:uppercase; letter-spacing:0.05em;">
                    Key Gene: <span style="color:#60a5fa; font-family:monospace;">{info.get('key_gene','')}</span>
                </div>
                <div style="background:#1e293b; border-radius:6px; padding:0.6rem; margin-bottom:0.5rem; font-size:0.82rem; color:#fbbf24; border-left:3px solid #f59e0b;">
                    📋 {info.get('clinical_action','')}
                </div>
                """, unsafe_allow_html=True)

                adme_col1, adme_col2 = st.columns(2)
                adme_items = [
                    ("🔬 Absorption",    info.get("absorption","N/A")),
                    ("🌐 Distribution",  info.get("distribution","N/A")),
                    ("⚗️ Metabolism",    info.get("metabolism","N/A")),
                    ("💧 Excretion",     info.get("excretion","N/A")),
                ]
                for i, (label, text) in enumerate(adme_items):
                    col = adme_col1 if i % 2 == 0 else adme_col2
                    with col:
                        st.markdown(f"""
                        <div class="mithra-card" style="margin-bottom:0.5rem;">
                            <div style="font-size:0.72rem; color:#6366f1; font-weight:600; margin-bottom:0.3rem;">{label}</div>
                            <div style="font-size:0.8rem; color:#94a3b8; line-height:1.5;">{text}</div>
                        </div>
                        """, unsafe_allow_html=True)

#  Export 
if st.session_state.get("llm_result") and st.session_state.get("rl_result"):
    st.markdown("---")
    st.markdown("### 📤 Export Report")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            from core.llm_pipeline import patientReport_LLM
            from core.report import getPdfReport
            import os

            with st.spinner("Generating clinical narrative and PDF..."):
                narrative = patientReport_LLM(
                    patient_info = patient_info,
                    llm_result   = st.session_state.llm_result,
                    rl_result    = st.session_state.rl_result,
                    disease      = disease,
                )
                os.makedirs("outputs", exist_ok=True)
                pdf_path = getPdfReport(
                    patient_info = patient_info,
                    llm_result   = st.session_state.llm_result,
                    rl_result    = st.session_state.rl_result,
                    disease      = disease,
                    narrative    = narrative,
                    output_path  = f"outputs/{patient_info['name'].replace(' ','_')}_report.pdf",
                )

            if pdf_path.endswith(".pdf"):
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name=f"mithra_report.pdf", mime="application/pdf")
            else:
                st.error(pdf_path)

    with exp2:
        json_data = json.dumps({
            "patient"    : patient_info,
            "llm_result" : st.session_state.llm_result,
            "rl_result"  : st.session_state.rl_result,
            "disease"    : disease,
        }, indent=2, default=str)
        st.download_button("📥 Export JSON", json_data,
                            file_name="mithra_report.json", mime="application/json",
                            use_container_width=True)

    with exp3:
        # Add to trial patients list
        if st.button("➕ Add to Trial Analytics", use_container_width=True):
            record = {
                "name"         : patient_info.get("name",""),
                "age"          : patient_info.get("age",""),
                "sex"          : patient_info.get("sex",""),
                "genes"        : patient_info.get("genes",{}),
                "llm_rec"      : st.session_state.llm_result.get("overall_recommendation",""),
                "rl_rec"       : st.session_state.rl_result.get("recommended_drug",""),
                "confidence"   : st.session_state.rl_result.get("confidence_pct",0),
            }
            st.session_state.all_patients.append(record)
            st.success(f"✅ Added to trial. Total: {len(st.session_state.all_patients)} patients.")
