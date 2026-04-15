# 🧬 MITHRA — Clinical Pharmacogenomics Intelligence Platform

> **Precision medicine decision support combining Bootstrapped Deep Reinforcement Learning with Large Language Models, grounded in PharmaGKB evidence.**

---

## What This Is

Mithra is a production-grade clinical decision support system (DSS) that helps clinicians assign patients to the optimal treatment arm in a pharmacogenomics-guided clinical trial.

It demonstrates three professional AI capabilities in a single, cohesive application:

| Capability | Implementation |
|---|---|
| **Deep Learning Model — Designed & Trained for Production** | Bootstrapped DQN (BDQL++) trained from scratch; live convergence charts; saved checkpoints |
| **Deep Expertise in LLM Architecture** | GPT-4o-mini + RAG over PharmaGKB via FAISS; React reasoning trace; structured Pydantic output; token budget management |
| **Model Optimisation for Resource-Constrained Environments** | float32→int8 weight quantization; Bootstrap head pruning (K→1); latency benchmarking table |

---

## System Requirements

| Component | Requirement |
|---|---|
| CPU | Any modern CPU (Intel Core i5+ / AMD Ryzen 5+). No GPU needed. |
| RAM | 4 GB minimum, 8 GB recommended (16–32 GB ideal) |
| Python | 3.10 or 3.11 |
| OS | Windows 10/11, macOS 12+, or Ubuntu 20.04+ |
| Internet | Required for OpenAI API calls |

**Tested on:** Intel Ultra Core 7, 32 GB RAM — training completes in ~90 seconds.

---

## Quick Start

### 1. Clone / Extract the project

```bash
cd mithra
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First install takes 2–4 minutes. FAISS and sentence-transformers are the largest packages.

### 4. Place data files

Ensure these four files are in the `data/` folder:

```
mithra/
└── data/
    ├── drugs.tsv
    ├── genes.tsv
    ├── phenotypes.tsv
    └── relationships.tsv
```

### 5. Run the application

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## Application Flow

```
Page 1 — Trial Setup
  Select disease area → drugs and genes auto-populate from PharmaGKB
  Configure RL hyperparameters (pre-tuned defaults work perfectly)
  Click "Initialise Trial"

Page 2 — Patient Input
  Choose a synthetic patient template or enter manually
  Gene metabolizer status dropdowns pre-fill intelligently
  Clinical risk badges appear instantly (Poor Metabolizer = ⚠️ High Risk)
  Click "Analyse Patient"

Page 3 — Train Agent
  Click "Train Agent"
  Watch live reward curve, loss curve, epsilon schedule, arm distribution
  Benchmark table shows full model vs quantized vs pruned latencies
  Training completes in ~60–120 seconds on CPU

Page 4 — Patient Report
  Click "Run LLM Analysis" — GPT streams pharmacogenomic reasoning
  RL recommendation shown with confidence % and Bootstrap head votes
  Per-drug ADME breakdown (Absorption, Distribution, Metabolism, Excretion)
  Export PDF report or JSON audit trail

Page 5 — Analytics
  Trial-level dashboard: cumulative performance, arm distribution
  BDQL++ vs random assignment comparison (reward distribution violin plot)
  Patient cohort table — all patients analysed in this session
  PharmaGKB knowledge base statistics
```

---

## Disease Areas Supported (Out of the Box)

| Disease | Drugs | Primary Genes |
|---|---|---|
| Major Depressive Disorder | Citalopram, Escitalopram, Sertraline, Venlafaxine, Nortriptyline, Amitriptyline | CYP2C19, CYP2D6 |
| Cardiovascular / Anticoagulation | Warfarin, Clopidogrel, Atorvastatin, Simvastatin, Rosuvastatin | CYP2C9, VKORC1, SLCO1B1 |
| Oncology — Fluoropyrimidines | Fluorouracil, Capecitabine, Mercaptopurine, Azathioprine | DPYD, TPMT |
| Pain Management / Opioids | Codeine, Tramadol, Ondansetron, Metoprolol | CYP2D6 |
| HIV / Infectious Disease | Abacavir, Efavirenz, Atazanavir | HLA-B, G6PD, CYP2C19 |
| Epilepsy / Neurology | Carbamazepine, Phenytoin, Oxcarbazepine | HLA-B, HLA-A, CYP2C9 |

All drug-gene associations are sourced from PharmaGKB and verified against CPIC dosing guidelines.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MITHRA APPLICATION                           │
├──────────────────┬──────────────────────┬───────────────────────┤
│   LLM PIPELINE   │    BDQL++ AGENT      │   OPTIMISATION        │
│                  │                      │                       │
│  Patient Input   │  Gene Profile        │  float32 Weights      │
│       ↓          │  Encoded as State    │       ↓               │
│  RAG Retrieval   │       ↓              │  int8 Quantization    │
│  PharmaGKB FAISS │  Shared Encoder      │  4× compression       │
│       ↓          │  (MLP backbone)      │       ↓               │
│  GPT-4o-mini     │       ↓              │  Head Pruning         │
│  React Reasoning │  K Bootstrap Heads   │  K heads → 1 head     │
│       ↓          │  Bernoulli Masks     │       ↓               │
│  Pydantic Parser │       ↓              │  Latency Benchmark    │
│  ADME Structure  │  Ensemble Vote       │  <1ms inference       │
│       ↓          │       ↓              │                       │
│  Clinical Report │  Arm Assignment      │  Production Ready     │
└──────────────────┴──────────────────────┴───────────────────────┘
                           ↓
                  PharmaGKB Knowledge Base
                  127,000+ Relationships
                  3,700+ Drugs · 25,000+ Genes
```

---

## Project Structure

```
mithra/
├── app.py                      ← Streamlit entry point + global CSS
├── config.py                   ← Disease/drug/gene configuration
├── requirements.txt
├── README.md
│
├── core/
│   ├── data_loader.py          ← PharmaGKB TSV loading + caching
│   ├── rl_agent.py             ← BDQL++ neural network + agent
│   ├── environment.py          ← Clinical trial RL environment
│   ├── trainer.py              ← Training loop + live callbacks
│   ├── llm_pipeline.py         ← RAG + GPT + structured output
│   └── report.py               ← PDF generation (fpdf2)
│
├── pages/
│   ├── 1_Trial_Setup.py        ← Disease / drug / gene configuration
│   ├── 2_Patient_Input.py      ← Smart patient entry + templates
│   ├── 3_Train_Agent.py        ← Live training dashboard
│   ├── 4_Patient_Report.py     ← Clinical report + export
│   └── 5_Analytics.py          ← Trial-level analytics
│
├── data/
│   ├── drugs.tsv               ← PharmaGKB drugs (place here)
│   ├── genes.tsv               ← PharmaGKB genes (place here)
│   ├── phenotypes.tsv          ← PharmaGKB phenotypes (place here)
│   └── relationships.tsv       ← PharmaGKB relationships (place here)
│
├── .streamlit/
│   └── config.toml             ← Dark theme + server config
│
├── checkpoints/                ← Auto-created during training
└── outputs/                    ← Auto-created for PDF/JSON exports
```

---

## Key Technical Decisions

**Why NumPy for the RL agent (not TensorFlow/PyTorch)?**
The agent is intentionally implemented in pure NumPy to demonstrate that you understand the mathematics, not just the framework. Every gradient step, every Adam update, every Huber loss computation is transparent and auditable. It runs in ~90 seconds on CPU — no GPU required for clinical deployment.

**Why RAG instead of a large context window?**
PharmaGKB has 127,000+ relationships. Stuffing them all into context is impossible and expensive. RAG retrieves only what is relevant to the specific patient's genes and drugs — this is the correct production architecture, and it keeps token costs low.

**Why Bootstrap heads instead of ε-greedy only?**
Bootstrap heads implement Thompson Sampling naturally — each head develops a different value estimate, and disagreement between heads signals genuine uncertainty. This is the correct exploration strategy for clinical trials where you must be confident before committing to a treatment arm.

---

## Disclaimer

Mithra is a research and demonstration system. It is not a certified medical device. All recommendations require review by a qualified clinical pharmacogenomics specialist before any clinical action is taken.

---

*Built with PharmaGKB · OpenAI GPT-4o-mini · Streamlit · NumPy · Plotly*
