import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#  Guard 
if not st.session_state.get("disease"):
    st.warning("⚠️ Please complete **Trial Setup** first.")
    st.stop()

disease = st.session_state.disease
cfg     = st.session_state.disease_config
drugs   = list(cfg["drugs"].keys())
agent   = st.session_state.get("agent")
trained = st.session_state.get("trained", False)

PLOTLY_BASE = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
    font=dict(color="#94a3b8", size=10),
    margin=dict(t=40, b=30, l=30, r=20),
)
GRID = dict(gridcolor="#334155", linecolor="#334155", zerolinecolor="#334155")

st.markdown(f"""
<div class="mithra-header">📊 Trial Analytics</div>
<div class="mithra-sub">Live trial dashboard — agent performance, arm distribution, and patient cohort analysis.</div>
""", unsafe_allow_html=True)

#  Top KPI strip 
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Disease",         disease.split("/")[0].strip()[:20])
k2.metric("Treatment Arms",  len(drugs))
k3.metric("Agent Trained",   "✅ Yes" if trained else "⏳ No")
k4.metric("Patients in Trial", len(st.session_state.get("all_patients", [])))
if trained and agent:
    k5.metric("Best Episode Reward", f"{max(agent.episode_rewards, default=0):.3f}")
else:
    k5.metric("Best Episode Reward", "—")

st.markdown("---")

# ═══════════════════════
# RL Training Performance
# ═══════════════════════
if trained and agent:
    st.markdown("### 🤖 Agent Training Performance")

    rewards   = agent.episode_rewards
    losses    = [l for l in agent.episode_losses if l and l > 0]
    epsilons  = agent.epsilons
    certainty = agent.head_certainty
    baseline  = st.session_state.get("baseline_result")

    def smooth(data, w=25):
        if len(data) < w:
            return data
        return list(np.convolve(data, np.ones(w) / w, mode="valid"))

    #  training chart 
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cumulative Episode Reward",
            "Training Loss (Huber)",
            "Exploration Schedule (ε-greedy)",
            "Bootstrap Head Certainty",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # Reward
    fig.add_trace(go.Scatter(
        y=rewards, mode="lines",
        line=dict(color="#6366f1", width=1), opacity=0.3,
        name="Episode Reward", showlegend=True,
    ), row=1, col=1)
    s_r = smooth(rewards)
    off = len(rewards) - len(s_r)
    fig.add_trace(go.Scatter(
        x=list(range(off, off + len(s_r))), y=s_r, mode="lines",
        line=dict(color="#8b5cf6", width=2.5),
        name="Smoothed (25ep)", showlegend=True,
    ), row=1, col=1)
    if baseline:
        fig.add_hline(y=baseline["mean_reward"], line_dash="dash",
                      line_color="#ef4444", annotation_text="Random Baseline",
                      annotation_font_color="#ef4444", row=1, col=1)

    # Loss
    fig.add_trace(go.Scatter(
        y=losses, mode="lines",
        line=dict(color="#22c55e", width=1), opacity=0.35,
        name="Loss", showlegend=True,
    ), row=1, col=2)
    s_l = smooth(losses)
    off_l = len(losses) - len(s_l)
    fig.add_trace(go.Scatter(
        x=list(range(off_l, off_l + len(s_l))), y=s_l, mode="lines",
        line=dict(color="#4ade80", width=2.5),
        name="Smoothed Loss", showlegend=True,
    ), row=1, col=2)

    # Epsilon
    fig.add_trace(go.Scatter(
        y=epsilons, mode="lines", fill="tozeroy",
        line=dict(color="#a78bfa", width=2),
        fillcolor="rgba(167,139,250,0.12)",
        name="Epsilon", showlegend=True,
    ), row=2, col=1)

    # Certainty
    if certainty:
        fig.add_trace(go.Scatter(
            y=certainty, mode="lines", fill="tozeroy",
            line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56,189,248,0.1)",
            name="Head Certainty", showlegend=True,
        ), row=2, col=2)

    fig.update_layout(
        **PLOTLY_BASE, height=520,
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", font_color="#94a3b8"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#e2e8f0"
        ann.font.size  = 11
    for ax in ["xaxis","yaxis","xaxis2","yaxis2","xaxis3","yaxis3","xaxis4","yaxis4"]:
        getattr(fig.layout, ax).update(**GRID)

    st.plotly_chart(fig, use_container_width=True)

    #  BDQL++ vs Random comparison 
    if baseline:
        st.markdown("### ⚖️ BDQL++ vs Random Assignment")

        rl_mean  = float(np.mean(rewards[-50:]))
        rl_std   = float(np.std(rewards[-50:]))
        rnd_mean = baseline["mean_reward"]
        rnd_std  = baseline["std_reward"]
        delta    = rl_mean - rnd_mean
        delta_pct= (delta / (abs(rnd_mean) + 1e-6)) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BDQL++ Mean Reward",  f"{rl_mean:.4f}",  f"±{rl_std:.4f}")
        c2.metric("Random Mean Reward",  f"{rnd_mean:.4f}", f"±{rnd_std:.4f}")
        c3.metric("Absolute Improvement",f"{delta:+.4f}")
        c4.metric("Relative Improvement",f"{delta_pct:+.1f}%",
                  delta_color="normal" if delta_pct > 0 else "inverse")

        # Side-by-side reward distribution
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Violin(
            y=rewards[-200:], name="BDQL++",
            line_color="#8b5cf6", fillcolor="rgba(139,92,246,0.2)",
            box_visible=True, meanline_visible=True,
        ))
        fig_cmp.add_trace(go.Violin(
            y=baseline["rewards"][-200:], name="Random",
            line_color="#ef4444", fillcolor="rgba(239,68,68,0.15)",
            box_visible=True, meanline_visible=True,
        ))
        fig_cmp.update_layout(
            **PLOTLY_BASE, height=300,
            yaxis=dict(**GRID, title="Episode Reward"),
            title=dict(text="Reward Distribution (last 200 episodes)", font_color="#e2e8f0"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    #  Arm assignment distribution 
    st.markdown("### 💊 Treatment Arm Assignment Distribution")

    arm_col1, arm_col2 = st.columns(2)
    arm_counts = agent.arm_counts

    with arm_col1:
        fig_bar = go.Figure(go.Bar(
            x=drugs, y=arm_counts,
            marker=dict(
                color=arm_counts,
                colorscale=[[0,"#1e3a5f"],[1,"#6366f1"]],
                showscale=False,
            ),
            text=arm_counts,
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_bar.update_layout(
            **PLOTLY_BASE, height=300,
            yaxis=dict(**GRID, title="Times Assigned"),
            xaxis=dict(**GRID),
            title=dict(text="Arm Selection Count (Training)", font_color="#e2e8f0"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with arm_col2:
        total_arms = sum(arm_counts) or 1
        fig_pie = go.Figure(go.Pie(
            labels=drugs,
            values=arm_counts,
            hole=0.55,
            marker=dict(colors=["#6366f1","#8b5cf6","#a78bfa","#c4b5fd","#7c3aed","#4f46e5"][:len(drugs)]),
            textfont=dict(color="#e2e8f0"),
        ))
        fig_pie.update_layout(
            **PLOTLY_BASE, height=300,
            title=dict(text="Arm Distribution (%)", font_color="#e2e8f0"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    #  Rolling statistics table 
    st.markdown("### 📈 Rolling Performance Statistics")

    window_sizes = [10, 25, 50, 100]
    stats_rows = []
    for w in window_sizes:
        if len(rewards) >= w:
            chunk = rewards[-w:]
            stats_rows.append({
                "Window"        : f"Last {w} episodes",
                "Mean Reward"   : f"{np.mean(chunk):.4f}",
                "Std Dev"       : f"{np.std(chunk):.4f}",
                "Min"           : f"{np.min(chunk):.4f}",
                "Max"           : f"{np.max(chunk):.4f}",
                "vs Random"     : f"{(np.mean(chunk) - rnd_mean if baseline else 0):+.4f}",
            })
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class="mithra-card" style="text-align:center; padding:3rem;">
        <div style="font-size:2rem; margin-bottom:0.5rem;">🤖</div>
        <div style="font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:0.3rem;">No Training Data Yet</div>
        <div style="font-size:0.85rem; color:#64748b;">
            Complete Trial Setup, then go to <strong style="color:#6366f1;">Train Agent</strong> to run the BDQL++ agent.
            Training analytics will appear here live.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════
# Section B — Patient Cohort Analytics
# ════════════════════════════════════
st.markdown("---")
st.markdown("### 👥 Patient Cohort")

patients = st.session_state.get("all_patients", [])

if patients:
    df_patients = pd.DataFrame(patients)

    # Metrics
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Patients", len(patients))

    # Agreement between LLM and RL
    if "llm_rec" in df_patients.columns and "rl_rec" in df_patients.columns:
        agreement = (df_patients["llm_rec"] == df_patients["rl_rec"]).mean() * 100
        p2.metric("LLM ↔ RL Agreement", f"{agreement:.0f}%")
    else:
        p2.metric("LLM ↔ RL Agreement", "—")

    if "confidence" in df_patients.columns:
        avg_conf = df_patients["confidence"].mean()
        p3.metric("Avg RL Confidence", f"{avg_conf:.1f}%")
    p4.metric("Drugs Recommended", df_patients["rl_rec"].nunique() if "rl_rec" in df_patients.columns else "—")

    # Cohort table
    st.dataframe(
        df_patients[[c for c in ["name","age","sex","rl_rec","llm_rec","confidence"]
                     if c in df_patients.columns]].rename(columns={
            "name"      : "Patient",
            "age"       : "Age",
            "sex"       : "Sex",
            "rl_rec"    : "RL Recommendation",
            "llm_rec"   : "LLM Recommendation",
            "confidence": "Confidence (%)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # RL recommendation pie
    if "rl_rec" in df_patients.columns and len(df_patients) > 1:
        fig_coh = go.Figure(go.Pie(
            labels=df_patients["rl_rec"].value_counts().index.tolist(),
            values=df_patients["rl_rec"].value_counts().values.tolist(),
            hole=0.5,
            marker=dict(colors=["#6366f1","#22c55e","#f59e0b","#ef4444","#8b5cf6","#06b6d4"][:len(drugs)]),
        ))
        fig_coh.update_layout(
            **PLOTLY_BASE, height=300,
            title=dict(text="Cohort — RL Arm Distribution", font_color="#e2e8f0"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        st.plotly_chart(fig_coh, use_container_width=True)

    # Export cohort
    csv = df_patients.to_csv(index=False)
    st.download_button("📥 Export Cohort CSV", csv,
                        file_name="mithra_cohort.csv", mime="text/csv")

else:
    st.markdown("""
    <div class="mithra-card" style="text-align:center; padding:2rem;">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">👥</div>
        <div style="font-size:0.9rem; color:#64748b;">
            No patients added yet. Run a patient analysis in
            <strong style="color:#6366f1;">Patient Report</strong>
            and click <em>Add to Trial Analytics</em>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════
# Section C — Model Architecture Card
# ═══════════════════════════════════
st.markdown("---")
st.markdown("### 🏗️ System Architecture")

arch_col1, arch_col2, arch_col3 = st.columns(3)

with arch_col1:
    st.markdown("""
    <div class="mithra-card">
        <div style="font-size:0.72rem; color:#6366f1; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:0.6rem;">LLM Pipeline</div>
        <div style="font-family:monospace; font-size:0.78rem; color:#94a3b8; line-height:2;">
            Patient Lab Report<br>
            ↓ RAG Retrieval<br>
            &nbsp;&nbsp;PharmaGKB FAISS<br>
            ↓ GPT-4o-mini<br>
            &nbsp;&nbsp;React Reasoning<br>
            ↓ Pydantic Parser<br>
            &nbsp;&nbsp;Structured ADME<br>
            ↓ Clinical Report
        </div>
    </div>
    """, unsafe_allow_html=True)

with arch_col2:
    rl_cfg = st.session_state.get("rl_config", {})
    heads  = rl_cfg.get("num_heads", 4)
    hidden = rl_cfg.get("hidden_size", 128)
    st.markdown(f"""
    <div class="mithra-card">
        <div style="font-size:0.72rem; color:#6366f1; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:0.6rem;">BDQL++ Agent</div>
        <div style="font-family:monospace; font-size:0.78rem; color:#94a3b8; line-height:2;">
            Gene Profile State<br>
            ↓ Shared Encoder<br>
            &nbsp;&nbsp;Linear→ReLU [{hidden}]<br>
            &nbsp;&nbsp;Linear→ReLU [{hidden}]<br>
            ↓ {heads} Bootstrap Heads<br>
            &nbsp;&nbsp;Bernoulli Masks<br>
            ↓ Ensemble Vote<br>
            &nbsp;&nbsp;Arm Assignment
        </div>
    </div>
    """, unsafe_allow_html=True)

with arch_col3:
    st.markdown("""
    <div class="mithra-card">
        <div style="font-size:0.72rem; color:#6366f1; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.05em; margin-bottom:0.6rem;">Optimisation Stack</div>
        <div style="font-family:monospace; font-size:0.78rem; color:#94a3b8; line-height:2;">
            Full Model (float32)<br>
            ↓ Weight Quantization<br>
            &nbsp;&nbsp;float32 → int8<br>
            &nbsp;&nbsp;4× compression<br>
            ↓ Head Pruning<br>
            &nbsp;&nbsp;K heads → 1 head<br>
            ↓ Production Model<br>
            &nbsp;&nbsp;&lt;1ms inference
        </div>
    </div>
    """, unsafe_allow_html=True)

#  PharmaGKB data stats 
st.markdown("---")
st.markdown("### 🗄️ Knowledge Base Statistics")

try:
    from core.data_loader import loadPharmGKB
    drugs_df, genes_df, pheno_df, rel_df = loadPharmGKB()

    ds1, ds2, ds3, ds4, ds5 = st.columns(5)
    ds1.metric("Drugs in DB",         f"{len(drugs_df):,}")
    ds2.metric("Genes in DB",         f"{len(genes_df):,}")
    ds3.metric("Phenotypes in DB",    f"{len(pheno_df):,}")
    ds4.metric("Total Relationships", f"{len(rel_df):,}")
    associated = rel_df[rel_df["Association"] == "associated"]
    ds5.metric("PGx Associations",    f"{len(associated):,}")
except Exception as e:
    st.caption(f"Data stats unavailable: {e}")
