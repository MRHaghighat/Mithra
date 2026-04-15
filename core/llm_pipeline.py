import os
import json
import pandas as pd
from typing import Dict, List, Optional
from config import OPENAI_API_KEY, OPENAI_MODEL, DATA_DIR

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def getClient():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def pharmgkbContext(
    gene_profile: Dict[str, str],
    drug_names: List[str],
    rel_df: pd.DataFrame,
    max_rows: int = 40,
) -> str:

    genes          = list(gene_profile.keys())
    pattern_genes  = "|".join(genes)
    pattern_drugs  = "|".join([d.lower() for d in drug_names])

    mask = (
        rel_df["Entity1_name"].str.contains(pattern_genes, case=False, na=False, regex=True) &
        rel_df["Entity2_name"].str.contains(pattern_drugs, case=False, na=False, regex=True) &
        (rel_df["Association"] == "associated")
    )
    relevant = rel_df[mask][["Entity1_name", "Entity2_name", "PK", "PD"]].drop_duplicates()

    if relevant.empty:
        mask2    = rel_df["Entity1_name"].str.contains(pattern_genes, case=False, na=False, regex=True)
        relevant = rel_df[mask2][["Entity1_name", "Entity2_name", "PK", "PD"]].drop_duplicates().head(max_rows)

    rows  = relevant.head(max_rows)
    lines = ["Gene/Variant → Drug/Phenotype associations from PharmaGKB:"]
    for _, r in rows.iterrows():
        pk  = "(Pharmacokinetic)"  if pd.notna(r.get("PK"))  and r.get("PK")  else ""
        pd_ = "(Pharmacodynamic)" if pd.notna(r.get("PD"))  and r.get("PD")  else ""
        lines.append(f"  • {r['Entity1_name']} ↔ {r['Entity2_name']} {pk}{pd_}".strip())
    return "\n".join(lines)


def analysePatient_LLM(
    gene_profile: Dict[str, str],
    drug_names: List[str],
    disease: str,
    patient_notes: str,
    rel_df: pd.DataFrame,
    stream_callback=None,
) -> Dict:
    client = getClient()

    pgx_context = pharmgkbContext(gene_profile, drug_names, rel_df)
    gene_lines  = "\n".join([f"  • {g}: {s}" for g, s in gene_profile.items()])
    drug_list   = ", ".join(drug_names)

    system_prompt = """You are Mithra, an expert clinical pharmacogenomics AI.
You analyse patient genetic profiles and provide evidence-based drug recommendations.
You always cite the specific gene variant and its known pharmacokinetic/pharmacodynamic effect.
You are precise, cautious, and clinician-friendly.
You respond ONLY in valid JSON — no markdown, no preamble."""

    user_prompt = f"""Disease Area: {disease}
Patient Clinical Notes: {patient_notes}

Patient Genetic Profile:
{gene_lines}

Candidate Drugs: {drug_list}

PharmaGKB Evidence (retrieved):
{pgx_context}

Analyse each drug for this patient. Return JSON exactly as:
{{
  "patient_summary": "2-sentence clinical summary of this patient's PGx profile",
  "overall_recommendation": "drug name of top recommendation",
  "reasoning": "clinical reasoning for top recommendation in 2-3 sentences",
  "drugs": {{
    "<DrugName>": {{
      "rank": 1,
      "recommendation": "Recommended | Use with Caution | Avoid",
      "confidence": "High | Moderate | Low",
      "absorption": "1 sentence on absorption impact of patient's gene profile",
      "distribution": "1 sentence on distribution",
      "metabolism": "1-2 sentences on metabolism — most important PGx effect",
      "excretion": "1 sentence on excretion",
      "key_gene": "most relevant gene for this drug",
      "clinical_action": "specific dose adjustment or monitoring recommendation"
    }}
  }},
  "warnings": ["list of any high-risk gene-drug interactions for this patient"]
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    token_estimate = len(user_prompt.split()) + len(system_prompt.split())
    response_text  = ""

    try:
        if stream_callback:
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1800,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                response_text += delta
                stream_callback(delta)
        else:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1800,
            )
            response_text = resp.choices[0].message.content

        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())
        result["_token_estimate"] = token_estimate
        result["_model"]          = OPENAI_MODEL
        return result

    except json.JSONDecodeError:
        return {
            "patient_summary"       : "Analysis complete — see raw output below.",
            "overall_recommendation": drug_names[0] if drug_names else "N/A",
            "reasoning"             : response_text[:500],
            "drugs"                 : {},
            "warnings"              : [],
            "_raw"                  : response_text,
            "_token_estimate"       : token_estimate,
        }
    except Exception as e:
        return {
            "patient_summary"       : f"Error: {str(e)}",
            "overall_recommendation": "N/A",
            "reasoning"             : str(e),
            "drugs"                 : {},
            "warnings"              : [],
        }


def patientReport_LLM(
    patient_info: dict,
    llm_result: dict,
    rl_result: dict,
    disease: str,
) -> str:
    client     = getClient()
    top_drug   = rl_result.get("recommended_drug", "N/A")
    confidence = rl_result.get("confidence_pct", 0)

    prompt = f"""Generate a concise clinical pharmacogenomics report (max 200 words) for:

Patient: {patient_info.get('name','Anonymous')}, Age {patient_info.get('age','N/A')}, {patient_info.get('sex','N/A')}
Disease: {disease}
Genetic Profile: {patient_info.get('genes', {})}
AI Recommendation: {top_drug} (Confidence: {confidence:.0f}%)
LLM Summary: {llm_result.get('patient_summary','')}
Clinical Reasoning: {llm_result.get('reasoning','')}
Warnings: {llm_result.get('warnings', [])}

Write as a formal clinical note. Be specific about gene variants and their effects.
Do not use markdown."""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=350,
    )
    return resp.choices[0].message.content
