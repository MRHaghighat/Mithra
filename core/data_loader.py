import os
import pandas as pd
import streamlit as st
from config import DATA_DIR


@st.cache_data(show_spinner=False)
def loadPharmGKB():
    base = DATA_DIR
    drugs_df     = pd.read_csv(os.path.join(base, "drugs.tsv"),         sep="\t", low_memory=False)
    genes_df     = pd.read_csv(os.path.join(base, "genes.tsv"),         sep="\t", low_memory=False)
    phenotypes_df= pd.read_csv(os.path.join(base, "phenotypes.tsv"),    sep="\t", low_memory=False)
    rel_df       = pd.read_csv(os.path.join(base, "relationships.tsv"), sep="\t", low_memory=False)
    return drugs_df, genes_df, phenotypes_df, rel_df


@st.cache_data(show_spinner=False)
def drugGeneMap():
    """
    Build a drug→[genes] lookup from PharmaGKB relationships.
    Includes Gene, Haplotype, and Variant evidence.
    Only key pharmacogenes included.
    """
    _, _, _, rel_df = loadPharmGKB()

    KEY_GENES = [
        "CYP2C19","CYP2D6","CYP2C9","CYP3A4","CYP3A5",
        "DPYD","TPMT","VKORC1","SLCO1B1","HLA-B","HLA-A","G6PD","CFTR"
    ]
    pattern = "|".join(KEY_GENES)

    pgx = rel_df[
        (rel_df["Entity2_type"] == "Chemical") &
        (rel_df["Association"] == "associated") &
        (rel_df["Entity1_type"].isin(["Gene","Haplotype","Variant"])) &
        (rel_df["Entity1_name"].str.contains(pattern, na=False, regex=True))
    ].copy()

    def _extract_gene(name):
        for g in KEY_GENES:
            if name.startswith(g):
                return g
        return name.split("*")[0].split(" ")[0].strip()

    pgx["gene_clean"] = pgx["Entity1_name"].apply(_extract_gene)
    pgx["drug_clean"] = pgx["Entity2_name"].str.lower().str.strip()

    return (pgx.groupby("drug_clean")["gene_clean"]
              .apply(lambda x: sorted(set(x)))
              .to_dict())


def diseaseSummary(disease_name: str, config: dict) -> dict:
    """Return a summary dict for a disease configuration."""
    drugs = config["drugs"]
    all_genes = set()
    for d in drugs.values():
        all_genes.update(d["genes"])
    return {
        "drug_count":   len(drugs),
        "gene_count":   len(all_genes),
        "all_genes":    sorted(all_genes),
        "drug_classes": sorted(set(d["class"] for d in drugs.values())),
    }
