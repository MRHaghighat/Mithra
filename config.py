import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")

OPENAI_MODEL   = "gpt-4o-mini"
DATA_DIR       = "data"

# ── Metabolizer phenotype labels per gene ────────────────────────────────────
GENE_PHENOTYPES = {
    "CYP2C19": ["Poor Metabolizer", "Intermediate Metabolizer", "Normal Metabolizer",
                "Rapid Metabolizer", "Ultrarapid Metabolizer"],
    "CYP2D6":  ["Poor Metabolizer", "Intermediate Metabolizer", "Normal Metabolizer",
                "Ultrarapid Metabolizer"],
    "CYP2C9":  ["Poor Metabolizer", "Intermediate Metabolizer", "Normal Metabolizer"],
    "CYP3A4":  ["Decreased Function", "Normal Function", "Increased Function"],
    "CYP3A5":  ["Poor Metabolizer", "Intermediate Metabolizer", "Normal Metabolizer"],
    "DPYD":    ["Poor Metabolizer (No Function)", "Intermediate Metabolizer",
                "Normal Metabolizer"],
    "TPMT":    ["Poor Metabolizer", "Intermediate Metabolizer", "Normal Metabolizer"],
    "VKORC1":  ["Low Dose Required", "Intermediate Dose Required", "High Dose Required"],
    "SLCO1B1": ["Poor Function", "Decreased Function", "Normal Function",
                "Increased Function"],
    "HLA-B":   ["*57:01 Absent (Negative)", "*57:01 Present (Positive)",
                "*58:01 Absent (Negative)", "*58:01 Present (Positive)"],
    "HLA-A":   ["*31:01 Absent (Negative)", "*31:01 Present (Positive)"],
    "G6PD":    ["Normal Activity", "Deficient Activity"],
}

GENE_DEFAULT_PHENOTYPE = {
    "CYP2C19": "Normal Metabolizer",
    "CYP2D6":  "Normal Metabolizer",
    "CYP2C9":  "Normal Metabolizer",
    "CYP3A4":  "Normal Function",
    "CYP3A5":  "Normal Metabolizer",
    "DPYD":    "Normal Metabolizer",
    "TPMT":    "Normal Metabolizer",
    "VKORC1":  "Intermediate Dose Required",
    "SLCO1B1": "Normal Function",
    "HLA-B":   "*57:01 Absent (Negative)",
    "HLA-A":   "*31:01 Absent (Negative)",
    "G6PD":    "Normal Activity",
}

# ── Disease areas with drugs and genes — all verified in PharmaGKB data ──────
DISEASE_CONFIG = {
    "Major Depressive Disorder": {
        "description": "Pharmacogenomics-guided antidepressant selection based on CYP2C19 and CYP2D6 metabolizer status.",
        "icd10": "F32 / F33",
        "drugs": {
            "Citalopram":    {"genes": ["CYP2C19", "CYP2D6"], "id": "PA449015", "class": "SSRI"},
            "Escitalopram":  {"genes": ["CYP2C19"],            "id": "PA10074",  "class": "SSRI"},
            "Sertraline":    {"genes": ["CYP2C19"],            "id": "PA451866", "class": "SSRI"},
            "Venlafaxine":   {"genes": ["CYP2D6"],             "id": "PA451866", "class": "SNRI"},
            "Nortriptyline": {"genes": ["CYP2C19", "CYP2D6"], "id": "PA31721",  "class": "TCA"},
            "Amitriptyline": {"genes": ["CYP2D6"],             "id": "PA448385", "class": "TCA"},
        },
        "primary_genes": ["CYP2C19", "CYP2D6"],
        "secondary_genes": ["CYP3A4"],
        "outcome_label": "Remission Score (0–100)",
    },
    "Cardiovascular / Anticoagulation": {
        "description": "PGx-guided anticoagulation and antiplatelet therapy to minimise bleeding risk and maximise efficacy.",
        "icd10": "I20–I25 / Z79.01",
        "drugs": {
            "Warfarin":      {"genes": ["CYP2C9", "VKORC1", "CYP2C19"], "id": "PA451906", "class": "Anticoagulant"},
            "Clopidogrel":   {"genes": ["CYP2C19"],                      "id": "PA449053", "class": "Antiplatelet"},
            "Atorvastatin":  {"genes": ["SLCO1B1", "CYP3A4", "CYP2C19"],"id": "PA448500", "class": "Statin"},
            "Simvastatin":   {"genes": ["SLCO1B1", "CYP3A4"],            "id": "PA451485", "class": "Statin"},
            "Rosuvastatin":  {"genes": ["SLCO1B1"],                      "id": "PA134308020","class": "Statin"},
        },
        "primary_genes": ["CYP2C9", "VKORC1", "SLCO1B1", "CYP2C19"],
        "secondary_genes": ["CYP3A4"],
        "outcome_label": "Therapeutic INR Achievement / Event-Free Days",
    },
    "Oncology — Fluoropyrimidines": {
        "description": "DPYD-guided fluoropyrimidine dosing to prevent severe toxicity in colorectal and breast cancer chemotherapy.",
        "icd10": "C18 / C50",
        "drugs": {
            "Fluorouracil":  {"genes": ["DPYD", "TPMT", "CYP2C19"], "id": "PA128406956", "class": "Fluoropyrimidine"},
            "Capecitabine":  {"genes": ["DPYD"],                     "id": "PA448771",    "class": "Fluoropyrimidine"},
            "Mercaptopurine":{"genes": ["TPMT"],                     "id": "PA450379",    "class": "Thiopurine"},
            "Azathioprine":  {"genes": ["TPMT"],                     "id": "PA448515",    "class": "Thiopurine"},
        },
        "primary_genes": ["DPYD", "TPMT"],
        "secondary_genes": ["CYP2C19"],
        "outcome_label": "Toxicity-Free Treatment Days",
    },
    "Pain Management / Opioids": {
        "description": "CYP2D6-guided opioid selection to prevent under-treatment or toxicity based on metabolizer status.",
        "icd10": "R52 / G89",
        "drugs": {
            "Codeine":       {"genes": ["CYP2D6"],             "id": "PA449088", "class": "Opioid"},
            "Tramadol":      {"genes": ["CYP2D6"],             "id": "PA451735", "class": "Opioid"},
            "Ondansetron":   {"genes": ["CYP2D6", "CYP3A4"],  "id": "PA450626", "class": "Antiemetic"},
            "Metoprolol":    {"genes": ["CYP2D6"],             "id": "PA450480", "class": "Beta-blocker"},
        },
        "primary_genes": ["CYP2D6"],
        "secondary_genes": ["CYP3A4"],
        "outcome_label": "Pain Control Score (0–100)",
    },
    "HIV / Infectious Disease": {
        "description": "HLA-B and CYP-guided antiretroviral selection to prevent hypersensitivity reactions.",
        "icd10": "B20 / Z21",
        "drugs": {
            "Abacavir":      {"genes": ["HLA-B"],              "id": "PA448004", "class": "NRTI"},
            "Efavirenz":     {"genes": ["CYP2B6", "CYP2C19"], "id": "PA449441", "class": "NNRTI"},
            "Atazanavir":    {"genes": ["CYP3A4", "G6PD"],    "id": "PA10251",  "class": "PI"},
        },
        "primary_genes": ["HLA-B", "G6PD", "CYP2C19"],
        "secondary_genes": ["CYP3A4"],
        "outcome_label": "Viral Load Suppression Score",
    },
    "Epilepsy / Neurology": {
        "description": "HLA and CYP-guided antiepileptic selection to prevent Stevens-Johnson syndrome and optimise seizure control.",
        "icd10": "G40",
        "drugs": {
            "Carbamazepine": {"genes": ["HLA-B", "HLA-A", "CYP3A5"], "id": "PA448785", "class": "Antiepileptic"},
            "Phenytoin":     {"genes": ["CYP2C9", "HLA-B"],          "id": "PA450940", "class": "Antiepileptic"},
            "Oxcarbazepine": {"genes": ["HLA-B"],                     "id": "PA450657", "class": "Antiepileptic"},
        },
        "primary_genes": ["HLA-B", "HLA-A", "CYP2C9"],
        "secondary_genes": ["CYP3A5"],
        "outcome_label": "Seizure-Free Days",
    },
}

# ── RL hyperparameters (tuned for CPU, fast convergence) ────────────────────
RL_CONFIG = {
    "hidden_size":        128,
    "num_heads":          4,
    "learning_rate":      5e-4,
    "gamma":              0.95,
    "batch_size":         64,
    "memory_size":        10_000,
    "target_update_freq": 100,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      0.992,
    "min_observations":   200,
}

# ── Synthetic patient templates per disease ──────────────────────────────────
SYNTHETIC_PATIENTS = {
    "Major Depressive Disorder": [
        {"name": "Patient A — CYP2C19 Poor Metabolizer",
         "age": 34, "weight": 72, "sex": "Female",
         "diagnosis_notes": "Recurrent MDD, PHQ-9 score 18, no prior antidepressant history.",
         "genes": {"CYP2C19": "Poor Metabolizer", "CYP2D6": "Normal Metabolizer"}},
        {"name": "Patient B — CYP2D6 Ultrarapid",
         "age": 51, "weight": 88, "sex": "Male",
         "diagnosis_notes": "First episode MDD, PHQ-9 score 14, history of chronic pain.",
         "genes": {"CYP2C19": "Normal Metabolizer", "CYP2D6": "Ultrarapid Metabolizer"}},
        {"name": "Patient C — Dual Poor Metabolizer",
         "age": 28, "weight": 61, "sex": "Female",
         "diagnosis_notes": "Treatment-resistant MDD, two prior failed SSRI trials.",
         "genes": {"CYP2C19": "Poor Metabolizer", "CYP2D6": "Poor Metabolizer"}},
    ],
    "Cardiovascular / Anticoagulation": [
        {"name": "Patient A — VKORC1 Sensitive + CYP2C9 IM",
         "age": 67, "weight": 79, "sex": "Male",
         "diagnosis_notes": "AF, needs anticoagulation. eGFR 58, no prior warfarin.",
         "genes": {"CYP2C9": "Intermediate Metabolizer", "VKORC1": "Low Dose Required", "SLCO1B1": "Normal Function"}},
        {"name": "Patient B — CYP2C19 Poor Metabolizer",
         "age": 58, "weight": 83, "sex": "Male",
         "diagnosis_notes": "Post-PCI, dual antiplatelet therapy required.",
         "genes": {"CYP2C9": "Normal Metabolizer", "VKORC1": "Intermediate Dose Required", "SLCO1B1": "Normal Function"}},
    ],
    "Oncology — Fluoropyrimidines": [
        {"name": "Patient A — DPYD Poor Metabolizer",
         "age": 55, "weight": 68, "sex": "Female",
         "diagnosis_notes": "Stage III colorectal cancer, FOLFOX candidate.",
         "genes": {"DPYD": "Poor Metabolizer (No Function)", "TPMT": "Normal Metabolizer"}},
        {"name": "Patient B — TPMT Intermediate",
         "age": 44, "weight": 74, "sex": "Male",
         "diagnosis_notes": "ALL maintenance therapy, thiopurine candidate.",
         "genes": {"DPYD": "Normal Metabolizer", "TPMT": "Intermediate Metabolizer"}},
    ],
}
