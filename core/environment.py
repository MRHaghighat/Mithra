import numpy as np
from typing import Dict, List, Tuple
from config import GENE_PHENOTYPES, GENE_DEFAULT_PHENOTYPE


#  Gene phenotype encoding 

def patientStateEncoder(gene_profile: Dict[str, str], gene_list: List[str]) -> np.ndarray:
    """
    Encode a patient's gene metabolizer phenotypes as a numeric vector.
    Each gene's phenotype is one-hot encoded within its possible values.
    """
    parts = []
    for gene in gene_list:
        phenotypes = GENE_PHENOTYPES.get(gene, ["Unknown"])
        status = gene_profile.get(gene, GENE_DEFAULT_PHENOTYPE.get(gene, phenotypes[0]))
        vec = np.zeros(len(phenotypes), dtype=np.float32)
        if status in phenotypes:
            vec[phenotypes.index(status)] = 1.0
        else:
            vec[0] = 1.0
        parts.append(vec)
    return np.concatenate(parts)


def geneStateDim(gene_list: List[str]) -> int:
    return sum(len(GENE_PHENOTYPES.get(g, ["Unknown"])) for g in gene_list)


#  Reward logic grounded in pharmacogenomics 

# Metabolizer → efficacy/risk multipliers
# Based on established CPIC dosing guidelines
METABOLIZER_EFFICACY = {
    "CYP2C19": {
        "Poor Metabolizer"          : {"efficacy": 0.3, "toxicity": 0.8},
        "Intermediate Metabolizer"  : {"efficacy": 0.7, "toxicity": 0.4},
        "Normal Metabolizer"        : {"efficacy": 1.0, "toxicity": 0.1},
        "Rapid Metabolizer"         : {"efficacy": 0.8, "toxicity": 0.1},
        "Ultrarapid Metabolizer"    : {"efficacy": 0.5, "toxicity": 0.2},
    },
    "CYP2D6": {
        "Poor Metabolizer"          : {"efficacy": 0.2, "toxicity": 0.9},
        "Intermediate Metabolizer"  : {"efficacy": 0.6, "toxicity": 0.5},
        "Normal Metabolizer"        : {"efficacy": 1.0, "toxicity": 0.1},
        "Ultrarapid Metabolizer"    : {"efficacy": 0.4, "toxicity": 0.3},
    },
    "CYP2C9": {
        "Poor Metabolizer"          : {"efficacy": 0.3, "toxicity": 0.85},
        "Intermediate Metabolizer"  : {"efficacy": 0.65, "toxicity": 0.4},
        "Normal Metabolizer"        : {"efficacy": 1.0, "toxicity": 0.1},
    },
    "DPYD": {
        "Poor Metabolizer (No Function)": {"efficacy": 0.5, "toxicity": 0.95},
        "Intermediate Metabolizer"       : {"efficacy": 0.8, "toxicity": 0.5},
        "Normal Metabolizer"             : {"efficacy": 1.0, "toxicity": 0.1},
    },
    "TPMT": {
        "Poor Metabolizer"          : {"efficacy": 0.4, "toxicity": 0.95},
        "Intermediate Metabolizer"  : {"efficacy": 0.75, "toxicity": 0.45},
        "Normal Metabolizer"        : {"efficacy": 1.0, "toxicity": 0.1},
    },
    "VKORC1": {
        "Low Dose Required"         : {"efficacy": 0.9, "toxicity": 0.7},
        "Intermediate Dose Required": {"efficacy": 1.0, "toxicity": 0.2},
        "High Dose Required"        : {"efficacy": 0.7, "toxicity": 0.1},
    },
    "SLCO1B1": {
        "Poor Function"             : {"efficacy": 0.6, "toxicity": 0.7},
        "Decreased Function"        : {"efficacy": 0.8, "toxicity": 0.4},
        "Normal Function"           : {"efficacy": 1.0, "toxicity": 0.1},
        "Increased Function"        : {"efficacy": 0.9, "toxicity": 0.15},
    },
    "HLA-B": {
        "*57:01 Absent (Negative)"  : {"efficacy": 1.0, "toxicity": 0.05},
        "*57:01 Present (Positive)" : {"efficacy": 0.0, "toxicity": 1.0},
        "*58:01 Absent (Negative)"  : {"efficacy": 1.0, "toxicity": 0.05},
        "*58:01 Present (Positive)" : {"efficacy": 0.0, "toxicity": 0.95},
    },
    "HLA-A": {
        "*31:01 Absent (Negative)"  : {"efficacy": 1.0, "toxicity": 0.05},
        "*31:01 Present (Positive)" : {"efficacy": 0.3, "toxicity": 0.85},
    },
    "G6PD": {
        "Normal Activity"           : {"efficacy": 1.0, "toxicity": 0.05},
        "Deficient Activity"        : {"efficacy": 0.4, "toxicity": 0.9},
    },
    "CYP3A4": {
        "Decreased Function"        : {"efficacy": 0.6, "toxicity": 0.5},
        "Normal Function"           : {"efficacy": 1.0, "toxicity": 0.1},
        "Increased Function"        : {"efficacy": 0.75, "toxicity": 0.15},
    },
    "CYP3A5": {
        "Poor Metabolizer"          : {"efficacy": 0.5, "toxicity": 0.5},
        "Intermediate Metabolizer"  : {"efficacy": 0.8, "toxicity": 0.25},
        "Normal Metabolizer"        : {"efficacy": 1.0, "toxicity": 0.1},
    },
}


class ClinicalTrialEnvironment:
    """
    RL environment simulating a pharmacogenomics-guided clinical trial.

    State  : encoded patient gene profile
    Action : drug arm index
    Reward : efficacy_score - toxicity_penalty + noise
             grounded in CPIC/PharmaGKB metabolizer guidance
    """

    def __init__(self, disease_config: dict, gene_list: List[str]):
        self.disease_config = disease_config
        self.gene_list      = gene_list
        self.drug_names     = list(disease_config["drugs"].keys())
        self.n_actions      = len(self.drug_names)
        self.state_dim      = geneStateDim(gene_list)
        self._current_state = None

    def reset(self) -> np.ndarray:
        """Generate a random synthetic patient."""
        profile = self._random_patient()
        self._current_profile = profile
        self._current_state   = patientStateEncoder(profile, self.gene_list)
        return self._current_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        drug_name   = self.drug_names[action]
        drug_info   = self.disease_config["drugs"][drug_name]
        relevant_genes = drug_info["genes"]

        efficacy_scores  = []
        toxicity_scores  = []

        for gene in relevant_genes:
            status = self._current_profile.get(gene, GENE_DEFAULT_PHENOTYPE.get(gene, ""))
            table  = METABOLIZER_EFFICACY.get(gene, {})
            entry  = table.get(status, {"efficacy": 0.7, "toxicity": 0.3})
            efficacy_scores.append(entry["efficacy"])
            toxicity_scores.append(entry["toxicity"])

        if not efficacy_scores:
            efficacy  = 0.7
            toxicity  = 0.3
        else:
            efficacy  = float(np.mean(efficacy_scores))
            toxicity  = float(np.mean(toxicity_scores))

        # Gaussian noise for stochasticity
        noise    = np.random.normal(0, 0.08)
        reward   = np.clip(efficacy - 0.6 * toxicity + noise, -1.0, 1.0)

        next_state = self.reset()
        return next_state, float(reward), True, {
            "drug"    : drug_name,
            "efficacy": efficacy,
            "toxicity": toxicity,
            "profile" : self._current_profile,
        }

    def _random_patient(self) -> Dict[str, str]:
        profile = {}
        for gene in self.gene_list:
            options = GENE_PHENOTYPES.get(gene, ["Normal"])
            # Weighted: normal metabolizer is most common in population
            weights = []
            for opt in options:
                if "Normal" in opt or "Absent" in opt:
                    weights.append(0.5)
                elif "Intermediate" in opt or "Decreased" in opt:
                    weights.append(0.3)
                else:
                    weights.append(0.2)
            total = sum(weights)
            weights = [w / total for w in weights]
            profile[gene] = np.random.choice(options, p=weights)
        return profile

    def encode_patient(self, gene_profile: Dict[str, str]) -> np.ndarray:
        return patientStateEncoder(gene_profile, self.gene_list)
