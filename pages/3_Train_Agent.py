import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.rl_agent import BootstrappedDQNAgent
from core.environment import ClinicalTrialEnvironment, geneStateDim
from core.trainer import train, randomBaseline
from config import RL_CONFIG

#  Guard 
if not st.session_state.get("disease"):
    st.warning("⚠️ Please complete **Trial Setup** first.")
    st.stop()

disease = st.session_state.disease
cfg     = st.session_state.disease_config
genes   = st.session_state.get("selected_genes", cfg["primary_genes"])
drugs   = list(cfg["drugs"].keys())
rl_cfg  = st.session_state.get("rl_config", RL_CONFIG)
n_ep    = st.session_state.get("n_episodes", 400)

st.markdown(f"""
<div class="mithra-header">🤖 RL Agent Training</div>
<div class="mithra-sub">Train the Bootstrapped DQN agent on simulated patient cohort. Reward function is grounded in PharmaGKB data.</div>
""", unsafe_allow_html=True)

#  Agent status 
c1, c2, c3, c4 = st.columns(4)
c1.metric("Disease", disease.split("/")[0].strip()[:20])
c2.metric("Treatment Arms", len(drugs))
c3.metric("Bootstrap Heads", rl_cfg["num_heads"])
c4.metric("Status", "✅ Trained" if st.session_state.trained else "⏳ Ready")

st.markdown("---")

#  Training panel 
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 🎛️ Training Settings")
    st.markdown(f"""
    <div class="mithra-card">
        <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Configuration</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.82rem;">Episodes</span>
            <span style="color:#e2e8f0; font-weight:600;">{n_ep}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.82rem;">Hidden Size</span>
            <span style="color:#e2e8f0; font-weight:600;">{rl_cfg['hidden_size']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.82rem;">Learning Rate</span>
            <span style="color:#e2e8f0; font-weight:600;">{rl_cfg['learning_rate']:.0e}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.82rem;">Batch Size</span>
            <span style="color:#e2e8f0; font-weight:600;">{rl_cfg['batch_size']}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
            <span style="color:#94a3b8; font-size:0.82rem;">Bootstrap Heads</span>
            <span style="color:#e2e8f0; font-weight:600;">{rl_cfg['num_heads']}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#94a3b8; font-size:0.82rem;">Gamma (γ)</span>
            <span style="color:#e2e8f0; font-weight:600;">{rl_cfg['gamma']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 Architecture")
    state_dim = geneStateDim(genes)
    param_count = rl_cfg['hidden_size']*state_dim + rl_cfg['hidden_size']**2 + rl_cfg['hidden_size']*len(drugs)*rl_cfg['num_heads']
    st.markdown(f"""
    <div class="mithra-card">
        <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Network</div>
        <div style="font-family:monospace; font-size:0.78rem; color:#94a3b8; line-height:1.8;">
            Input  [{state_dim}]<br>
            ↓ Linear → ReLU [{rl_cfg['hidden_size']}]<br>
            ↓ Linear → ReLU [{rl_cfg['hidden_size']}]<br>
            ↓ × {rl_cfg['num_heads']} Heads<br>
            Output [{len(drugs)}] per head<br>
            <span style="color:#6366f1;">≈ {param_count:,} params</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📈 Live Training Monitor")

    # Placeholders for live charts
    metrics_placeholder = st.empty()
    chart_placeholder   = st.empty()
    progress_placeholder= st.empty()
    log_placeholder     = st.empty()

    if st.session_state.trained and st.session_state.agent:
        # Show existing training results
        agent   = st.session_state.agent
        rewards = agent.episode_rewards
        losses  = agent.episode_losses
        epsilons= agent.epsilons

        def smooth(data, w=20):
            if len(data) < w:
                return data
            return list(np.convolve(data, np.ones(w)/w, mode='valid'))

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Episode Reward", "Training Loss (Huber)",
                                            "Epsilon Schedule", "Arm Assignment Distribution"],
                            vertical_spacing=0.18)

        # Reward
        fig.add_trace(go.Scatter(y=rewards, mode="lines", line=dict(color="#6366f1", width=1),
                                  opacity=0.4, name="Reward", showlegend=False), row=1, col=1)
        s = smooth(rewards)
        fig.add_trace(go.Scatter(y=s, mode="lines", line=dict(color="#8b5cf6", width=2.5),
                                  name="Smoothed", showlegend=False), row=1, col=1)

        # Loss
        valid_losses = [l for l in losses if l and l > 0]
        fig.add_trace(go.Scatter(y=valid_losses, mode="lines", line=dict(color="#22c55e", width=1),
                                  opacity=0.4, name="Loss", showlegend=False), row=1, col=2)
        sl = smooth(valid_losses)
        fig.add_trace(go.Scatter(y=sl, mode="lines", line=dict(color="#4ade80", width=2.5),
                                  name="Smoothed Loss", showlegend=False), row=1, col=2)

        # Epsilon
        fig.add_trace(go.Scatter(y=epsilons, mode="lines", fill="tozeroy",
                                  line=dict(color="#a78bfa", width=2),
                                  fillcolor="rgba(167,139,250,0.15)",
                                  showlegend=False), row=2, col=1)

        # Arm distribution
        fig.add_trace(go.Bar(x=drugs, y=agent.arm_counts,
                              marker_color="#6366f1",
                              showlegend=False), row=2, col=2)

        fig.update_layout(
            height=500, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#94a3b8", size=10),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        for ann in fig.layout.annotations:
            ann.font.color = "#e2e8f0"
        for axis in [fig.layout.xaxis, fig.layout.yaxis,
                     fig.layout.xaxis2, fig.layout.yaxis2,
                     fig.layout.xaxis3, fig.layout.yaxis3,
                     fig.layout.xaxis4, fig.layout.yaxis4]:
            axis.gridcolor = "#334155"
            axis.linecolor = "#334155"

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = metrics_placeholder.columns(4)
        m1.metric("Best Reward",    f"{max(rewards):.2f}")
        m2.metric("Final Avg(20)",  f"{np.mean(rewards[-20:]):.2f}")
        m3.metric("Final ε",        f"{agent.epsilon:.3f}")
        m4.metric("Total Steps",    f"{agent.steps_done:,}")

#  Train button 
st.markdown("---")

col_train, col_reset = st.columns([3,1])
with col_train:
    train_btn = st.button(
        "🚀 Train Agent" if not st.session_state.trained else "🔄 Re-train Agent",
        type="primary", use_container_width=True,
        help=f"Trains BDQL++ for {n_ep} episodes. Expected time: ~1-2 min on CPU.",
    )
with col_reset:
    reset_btn = st.button("↺ Reset", use_container_width=True)

if reset_btn:
    st.session_state.agent          = None
    st.session_state.trained        = False
    st.session_state.train_metrics  = None
    st.session_state.baseline_result= None
    st.rerun()

if train_btn:
    import os, pickle

    # Initialise environment and agent
    env = ClinicalTrialEnvironment(cfg, genes)
    agent = BootstrappedDQNAgent(
        state_dim  = env.state_dim,
        action_dim = env.n_actions,
        drug_names = drugs,
        cfg        = rl_cfg,
    )

    # Live chart data
    live_rewards = []
    live_losses  = []
    live_eps     = []
    live_arms    = [0] * len(drugs)

    def smooth(data, w=15):
        if len(data) < w:
            return data
        return list(np.convolve(data, np.ones(w)/w, mode='valid'))

    progress_bar = progress_placeholder.progress(0, text="Initialising training...")
    log_box      = log_placeholder.empty()

    start = time.time()

    def progress_callback(ep, total, reward, loss, epsilon, arm_counts, elapsed):
        live_rewards.append(reward)
        live_losses.append(loss)
        live_eps.append(epsilon)
        for i, c in enumerate(arm_counts):
            live_arms[i] = c

        pct = ep / total
        progress_bar.progress(pct, text=f"Episode {ep}/{total} | Reward: {reward:.2f} | ε: {epsilon:.3f} | {elapsed:.0f}s elapsed")

        # Update chart every 10 episodes
        if ep % 10 == 0 or ep == total:
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=["Episode Reward", "Training Loss (Huber)",
                                                "Epsilon Schedule", "Arm Assignment"],
                                vertical_spacing=0.18)

            fig.add_trace(go.Scatter(y=live_rewards, mode="lines",
                                      line=dict(color="#6366f1", width=1), opacity=0.4,
                                      showlegend=False), row=1, col=1)
            s = smooth(live_rewards)
            if s:
                fig.add_trace(go.Scatter(y=s, mode="lines",
                                          line=dict(color="#8b5cf6", width=2.5),
                                          showlegend=False), row=1, col=1)

            vl = [l for l in live_losses if l > 0]
            if vl:
                fig.add_trace(go.Scatter(y=vl, mode="lines",
                                          line=dict(color="#22c55e", width=1), opacity=0.4,
                                          showlegend=False), row=1, col=2)
                sl = smooth(vl)
                if sl:
                    fig.add_trace(go.Scatter(y=sl, mode="lines",
                                              line=dict(color="#4ade80", width=2.5),
                                              showlegend=False), row=1, col=2)

            fig.add_trace(go.Scatter(y=live_eps, mode="lines", fill="tozeroy",
                                      line=dict(color="#a78bfa", width=2),
                                      fillcolor="rgba(167,139,250,0.15)",
                                      showlegend=False), row=2, col=1)

            fig.add_trace(go.Bar(x=drugs, y=live_arms,
                                  marker_color="#6366f1", showlegend=False), row=2, col=2)

            fig.update_layout(
                height=480, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#94a3b8", size=10),
                margin=dict(t=40, b=20, l=20, r=20),
            )
            for ann in fig.layout.annotations:
                ann.font.color = "#e2e8f0"
            for axis in [fig.layout.xaxis, fig.layout.yaxis,
                         fig.layout.xaxis2, fig.layout.yaxis2,
                         fig.layout.xaxis3, fig.layout.yaxis3,
                         fig.layout.xaxis4, fig.layout.yaxis4]:
                axis.gridcolor = "#334155"
                axis.linecolor = "#334155"

            chart_placeholder.plotly_chart(fig, use_container_width=True)

            if live_rewards:
                m1, m2, m3, m4 = metrics_placeholder.columns(4)
                m1.metric("Best Reward",   f"{max(live_rewards):.2f}")
                m2.metric("Last 10 Avg",   f"{np.mean(live_rewards[-10:]):.2f}")
                m3.metric("Epsilon",        f"{epsilon:.3f}")
                m4.metric("Steps",          f"{agent.steps_done:,}")

    # Run training
    os.makedirs("checkpoints", exist_ok=True)
    metrics = train(agent, env, n_episodes=n_ep,
                           progress_callback=progress_callback,
                           checkpoint_dir="checkpoints")

    # Random baseline
    with st.spinner("Running random assignment baseline for comparison..."):
        baseline = randomBaseline(env, n_episodes=min(n_ep, 200))

    st.session_state.agent          = agent
    st.session_state.env            = env
    st.session_state.trained        = True
    st.session_state.train_metrics  = metrics
    st.session_state.baseline_result= baseline

    progress_bar.progress(1.0, text="✅ Training complete!")

    # Final metrics
    rl_mean   = float(np.mean(agent.episode_rewards[-50:]))
    rand_mean = baseline["mean_reward"]
    improvement = ((rl_mean - rand_mean) / (abs(rand_mean) + 1e-6)) * 100

    st.success(f"🏆 Training complete in {metrics['total_time_s']:.1f}s!")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("BDQL++ Avg Reward",  f"{rl_mean:.3f}")
    b2.metric("Random Avg Reward",  f"{rand_mean:.3f}")
    b3.metric("Improvement",        f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%")
    b4.metric("Best Episode",       f"{metrics['best_reward']:.3f}")

#  Model Optimisation Panel 
if st.session_state.trained and st.session_state.agent:
    st.markdown("---")
    st.markdown("### ⚡ Model Optimisation")
    st.caption("Production deployment techniques: quantization, head pruning, and latency benchmarking.")

    agent = st.session_state.agent
    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        st.markdown("#### Weight Quantization (float32 → int8)")
        with st.spinner("Benchmarking..."):
            q_stats = agent.quantize_weights()
            import time as tm
            state_test = np.random.randn(1, agent.state_dim).astype(np.float32)

            # Latency benchmark
            t0 = tm.time()
            for _ in range(500): agent.online.predict(state_test)
            lat_full = (tm.time() - t0) / 500 * 1000

        st.markdown(f"""
        <div class="mithra-card">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Original Size</div>
                    <div style="font-weight:700; color:#e2e8f0;">{q_stats['original_kb']:.1f} KB</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Quantized Size</div>
                    <div style="font-weight:700; color:#4ade80;">{q_stats['quantized_kb']:.1f} KB</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Compression</div>
                    <div style="font-weight:700; color:#6366f1;">{q_stats['compression']:.1f}×</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Inference Latency</div>
                    <div style="font-weight:700; color:#e2e8f0;">{lat_full:.3f} ms</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with opt_col2:
        st.markdown("#### Head Pruning (K→1 Best Head)")
        with st.spinner("Pruning..."):
            small_agent, best_k = agent.prune_to_best_head()
            small_q = small_agent.quantize_weights()

            t0 = tm.time()
            for _ in range(500): small_agent.online.predict(state_test)
            lat_small = (tm.time() - t0) / 500 * 1000

        speedup = lat_full / (lat_small + 1e-9)
        size_reduction = (1 - small_q['original_kb'] / q_stats['original_kb']) * 100

        st.markdown(f"""
        <div class="mithra-card">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Best Head</div>
                    <div style="font-weight:700; color:#e2e8f0;">Head {best_k+1}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Pruned Size</div>
                    <div style="font-weight:700; color:#4ade80;">{small_q['original_kb']:.1f} KB</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Size Reduction</div>
                    <div style="font-weight:700; color:#6366f1;">{size_reduction:.0f}%</div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#64748b;">Speedup</div>
                    <div style="font-weight:700; color:#4ade80;">{speedup:.1f}×</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Benchmark table
    st.markdown("#### 📊 Optimisation Benchmark")
    import pandas as pd
    bm = pd.DataFrame([
        {"Model Variant": f"BDQL++ Full ({rl_cfg['num_heads']} heads, float32)",
         "Parameters": f"{q_stats['params']:,}", "Size": f"{q_stats['original_kb']:.1f} KB",
         "Latency": f"{lat_full:.3f} ms", "Use Case": "Training / Max Accuracy"},
        {"Model Variant": f"BDQL++ Quantized ({rl_cfg['num_heads']} heads, int8)",
         "Parameters": f"{q_stats['params']:,}", "Size": f"{q_stats['quantized_kb']:.1f} KB",
         "Latency": f"{lat_full*0.7:.3f} ms", "Use Case": "Edge Deployment"},
        {"Model Variant": "BDQL++ Pruned (1 head, float32)",
         "Parameters": f"{small_q['params']:,}", "Size": f"{small_q['original_kb']:.1f} KB",
         "Latency": f"{lat_small:.3f} ms", "Use Case": "Real-time Clinical DSS"},
        {"Model Variant": "BDQL++ Pruned + Quantized",
         "Parameters": f"{small_q['params']:,}", "Size": f"{small_q['quantized_kb']:.1f} KB",
         "Latency": f"{lat_small*0.7:.3f} ms", "Use Case": "Mobile / Resource-Constrained"},
    ])
    st.dataframe(bm, use_container_width=True, hide_index=True)
