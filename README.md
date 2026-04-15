# 🧬 MITHRA — Clinical Pharmacogenomics Intelligence Platform

> **Precision medicine decision support combining Bootstrapped Deep Reinforcement Learning with Large Language Models, grounded in PharmaGKB evidence.**

---

## Screenshots

### Trial Setup
> Select a disease area — drugs, genes, and RL hyperparameters auto-populate from the PharmaGKB knowledge base.

![Trial Setup](screenshots/trial_setup.png)

---

### Patient Input
> Smart patient entry with metabolizer-status dropdowns, synthetic templates, and instant clinical risk badges.

![Patient Input](screenshots/patient_input.png)

---

### Train Agent
> Live reward curve, loss curve, epsilon schedule, and arm distribution — plus a latency benchmark table comparing full, quantized, and pruned models.

![Train Agent](screenshots/train_agent.png)

---

### Patient Report
> GPT-4o-mini streams pharmacogenomic reasoning in real time. RL recommendation shown with confidence % and Bootstrap head vote breakdown. Export to PDF or JSON.

![Patient Report](screenshots/patient_report.png)

---

### Analytics
> Trial-level dashboard: cumulative reward, arm distribution, BDQL++ vs random assignment comparison, and full patient cohort table.

![Analytics](screenshots/analytics.png)

---

## What This Is

Mithra is a production-grade clinical decision support system (DSS) that helps clinicians assign patients to the optimal treatment arm in a pharmacogenomics-guided clinical trial.

It demonstrates three professional AI capabilities in a single, cohesive application:

| Capability | Implementation |
|---|---|
| **Deep Learning Model — Designed & Trained for Production** | Bootstrapped DQN (BDQL++) trained from scratch; live convergence charts; saved checkpoints |
| **Deep Expertise in LLM Architecture** | GPT-4o-mini + RAG over PharmaGKB; ReAct reasoning trace; structured Pydantic output; token budget management |
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

> First install takes 2–4 minutes.

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

### 5. Add your OpenAI API key

```bash
# Windows
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```

### 6. Run the application

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

Mithra is structured as three independent vertical pipelines that share a single knowledge base and converge at the clinical report layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MITHRA APPLICATION                           │
├──────────────────┬──────────────────────┬───────────────────────┤
│   LLM PIPELINE   │    BDQL++ AGENT      │   OPTIMISATION        │
│                  │                      │                       │
│  Patient Input   │  Gene Profile        │  float32 Weights      │
│       ↓          │  Encoded as State    │       ↓               │
│  RAG Retrieval   │       ↓              │  int8 Quantization    │
│  PharmaGKB       │  Shared Encoder      │  4× compression       │
│       ↓          │  (MLP backbone)      │       ↓               │
│  GPT-4o-mini     │       ↓              │  Head Pruning         │
│  ReAct Reasoning │  K Bootstrap Heads   │  K heads → 1 head     │
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

### Why three independent pipelines?

Each pipeline addresses a distinct failure mode. The LLM pipeline can fail at retrieval
or generation — that failure is isolated and does not affect the RL arm assignment. The
optimisation pipeline is a post-training concern that touches only weights, not logic.
This separation means each component can be tested, replaced, or scaled independently,
which is the correct property for any system that will eventually integrate with a
real clinical EHR.

### Data flow through the system

A single patient interaction triggers three concurrent processes:

1. **State encoding** — the patient's gene metabolizer statuses are one-hot encoded
   into a fixed-length state vector and passed to the BDQL++ agent for arm selection.
2. **RAG retrieval** — the same gene/drug profile is used to query the PharmaGKB
   knowledge base, selecting the most relevant drug-gene relationships for the LLM context.
3. **Report assembly** — the RL recommendation and LLM narrative are merged by
   `core/report.py` into a structured clinical output with a full audit trail.

The PharmaGKB knowledge base is the single source of truth for both pipelines,
ensuring the RL agent and the LLM are reasoning from the same evidence.

---

## Model Optimisation — Deep Dive

The optimisation pipeline (`Page 3 — Train Agent`) demonstrates three distinct
techniques relevant to deploying ML models in resource-constrained clinical environments
such as edge servers, hospital on-premise infrastructure, or point-of-care devices.

### 1. float32 → int8 Weight Quantization

After training, every weight matrix in the shared MLP encoder and all Bootstrap heads
is quantized from 32-bit floats to 8-bit integers using symmetric linear quantization:

```
scale = max(|W|) / 127
W_int8 = round(W / scale)
```

This produces approximately **4× model size reduction** with negligible accuracy loss
on the BDQL++ task because the reward signal is coarse (discrete arm labels) and the
weight distributions are well-behaved after training. At inference, weights are
dequantized on the fly — the forward pass remains numerically stable at float32
precision while the stored model footprint is dramatically smaller.

**Why this matters clinically:** many hospital systems run inference on CPU-only
servers with strict storage quotas. A 4× reduction can be the difference between
a model that fits within an IT-approved deployment envelope and one that does not.

### 2. Bootstrap Head Pruning (K → 1)

The BDQL++ agent trains with K Bootstrap heads, each with independent Bernoulli
masks applied to the shared encoder output. At deployment time, the K heads are
pruned to a single head — the one with the highest cumulative validation reward.

This reduces inference cost from O(K) to O(1) forward passes while retaining the
exploration benefits that Bootstrap heads provided during training. The pruned
single-head model is what gets quantized and benchmarked in the latency table.

**Exploration vs exploitation trade-off:** K heads are necessary during training
because disagreement between heads drives Thompson Sampling — uncertain states
get explored more. After training converges, that uncertainty signal is no longer
needed; the best head is a sufficient policy for deployment.

### 3. Latency Benchmarking

The benchmark table on Page 3 reports wall-clock inference latency (mean ± std
over 1,000 forward passes) for three model variants:

| Model | Weights | Heads | Expected Latency |
|---|---|---|---|
| Full (float32, K heads) | float32 | K | baseline |
| Quantized (int8, K heads) | int8 | K | ~40–60% reduction |
| Pruned + Quantized (int8, 1 head) | int8 | 1 | <1ms on CPU |

This table is the kind of output a clinical engineering team would require before
approving a model for deployment — concrete numbers, not just theoretical claims.

---

## Key Technical Decisions

### Why NumPy for the RL agent (not PyTorch)?

The agent is intentionally implemented in pure NumPy to demonstrate understanding
of the mathematics, not just the framework. Every gradient step, every Adam update,
every Huber loss computation is explicit and auditable. For a clinical system,
auditability is not optional — a regulator or clinical team asking "how does it
decide?" deserves a transparent answer, not a black-box framework call.

It also runs in ~90 seconds on CPU with no GPU required, which is the correct
deployment profile for most hospital infrastructure.

### Why RAG instead of a large context window?

PharmaGKB contains 127,000+ drug-gene relationships. Stuffing them all into
context is impossible and expensive. RAG retrieves only what is relevant to the
specific patient's genes and drugs — this is the correct production architecture.
It also means token costs scale with patient complexity, not with database size,
which is the right economic property for a deployed system.

### Why Bootstrap heads instead of ε-greedy exploration only?

Bootstrap heads implement Thompson Sampling naturally — each head develops an
independent value estimate over the shared encoder, and disagreement between
heads signals genuine uncertainty about a treatment arm. This is more principled
than ε-greedy for a clinical trial context, where random exploration has a direct
cost (a patient assigned to a suboptimal arm). Thompson Sampling concentrates
exploration on genuinely uncertain arms rather than exploring uniformly at random.

### Why Pydantic for LLM output parsing?

GPT-4o-mini output is parsed into a typed Pydantic schema before it reaches the
report layer. This means a malformed LLM response fails loudly at the boundary,
not silently inside the report generator. It also makes the ADME breakdown
(Absorption, Distribution, Metabolism, Excretion) programmatically addressable —
each field can be rendered, audited, or exported independently.

### Why disease-agnostic configuration?

`config.py` defines disease areas, drugs, genes, and reward shaping as pure data
— the RL environment and LLM pipeline consume this configuration at runtime.
Adding a new disease area requires no code changes, only a new config block.
This is the correct architecture for a system intended to expand beyond its
initial MDD demo — and it is what makes Mithra commercially extensible.

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

## Disclaimer

Mithra is a research and demonstration system. It is not a certified medical device.
All recommendations require review by a qualified clinical pharmacogenomics specialist
before any clinical action is taken.

---

*Built with PharmaGKB · OpenAI GPT-4o-mini · Streamlit · NumPy · Plotly*