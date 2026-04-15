"""
core/report.py
PDF clinical report generation using fpdf2.
"""
import json
import os
from datetime import datetime
from typing import Dict


def getPdfReport(
    patient_info: dict,
    llm_result: dict,
    rl_result: dict,
    disease: str,
    narrative: str,
    output_path: str = "outputs/patient_report.pdf",
) -> str:
    try:
        from fpdf import FPDF, XPos, YPos

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pdf = FPDF()
        pdf.set_margins(20, 20, 20)
        pdf.add_page()

        #  Header 
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(0, 0, 210, 40, "F")

        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(20, 10)
        pdf.cell(0, 10, "MITHRA", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(148, 163, 184)
        pdf.set_xy(20, 22)
        pdf.cell(0, 6, "Clinical Pharmacogenomics Intelligence Platform")
        pdf.set_xy(130, 22)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R")

        pdf.set_y(50)
        pdf.set_text_color(15, 23, 42)

        def _header(title):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(30, 41, 59)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            pdf.set_text_color(15, 23, 42)
            pdf.ln(2)

        def _body(label, value):
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(55, 6, label + ":", new_x=XPos.RIGHT)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        #  Patient Info 
        _header("PATIENT INFORMATION")
        _body("Patient ID", patient_info.get("name", "Anonymous"))
        _body("Age / Sex", f"{patient_info.get('age','N/A')} / {patient_info.get('sex','N/A')}")
        _body("Disease Area", disease)
        _body("Clinical Notes", patient_info.get("diagnosis_notes", ""))
        pdf.ln(3)

        #  Genetic Profile 
        _header("PHARMACOGENOMIC PROFILE")
        genes = patient_info.get("genes", {})
        for gene, status in genes.items():
            _body(gene, status)
        pdf.ln(3)

        #  AI Recommendation 
        _header("AI TREATMENT RECOMMENDATION")
        top_drug    = rl_result.get("recommended_drug", "N/A")
        confidence  = rl_result.get("confidence_pct", 0)
        head_votes  = rl_result.get("head_votes", {})

        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(34, 197, 94)
        pdf.cell(0, 10, f"  Recommended: {top_drug}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, f"  Confidence Score: {confidence:.1f}%  |  Bootstrap Head Consensus: {head_votes}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        _body("LLM Reasoning", llm_result.get("reasoning", ""))
        pdf.ln(3)

        #  Clinical Narrative 
        _header("CLINICAL NARRATIVE")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, narrative)
        pdf.ln(3)

        #  Per-drug Summary 
        _header("PER-DRUG PHARMACOGENOMIC ANALYSIS")
        drugs_analysis = llm_result.get("drugs", {})
        for drug_name, info in drugs_analysis.items():
            pdf.set_font("Helvetica", "B", 10)
            rec_color = {"Recommended": (34, 197, 94), "Use with Caution": (251, 191, 36), "Avoid": (239, 68, 68)}
            col = rec_color.get(info.get("recommendation", ""), (100, 100, 100))
            pdf.set_text_color(*col)
            pdf.cell(0, 7, f"  {drug_name} - {info.get('recommendation','N/A')} ({info.get('confidence','N/A')} confidence)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(15, 23, 42)
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(0, 5, f"    Key gene: {info.get('key_gene','')} | Action: {info.get('clinical_action','')}",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        #  Warnings 
        warnings = llm_result.get("warnings", [])
        if warnings:
            _header("CLINICAL ALERTS")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(239, 68, 68)
            for w in warnings:
                pdf.multi_cell(0, 5, f"  ⚠  {w}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(15, 23, 42)

        #  Footer 
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 5,
                 "MITHRA Clinical Intelligence Platform - For research and decision support only. "
                 "Not a substitute for clinical judgement. Consult a qualified pharmacogenomics specialist.",
                 align="C")

        pdf.output(output_path)
        return output_path

    except Exception as e:
        return f"PDF generation error: {e}"
