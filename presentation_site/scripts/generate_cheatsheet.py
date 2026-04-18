"""Generate a printable cheat-sheet PDF for the live presentation demo."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_LEFT

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_cheatsheet.pdf"))

# ----- colors tuned to the site -----
BG_DARK = colors.HexColor("#0d1219")
TEXT = colors.HexColor("#1a2332")
DIM = colors.HexColor("#55606e")
ACCENT = colors.HexColor("#0891b2")
WIN = colors.HexColor("#15803d")
FAIL = colors.HexColor("#dc2626")
BORDER = colors.HexColor("#cbd5e1")
ROW_ALT = colors.HexColor("#f1f5f9")
HEADER_BG = colors.HexColor("#0f172a")

styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=20, leading=24,
    textColor=TEXT, alignment=TA_LEFT, spaceAfter=4,
)
SUB = ParagraphStyle(
    "Sub", parent=styles["Normal"],
    fontName="Helvetica", fontSize=10, leading=13,
    textColor=DIM, spaceAfter=16,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=13, leading=16,
    textColor=ACCENT, spaceAfter=6, spaceBefore=10,
)
BODY = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontName="Helvetica", fontSize=9.5, leading=12.5,
    textColor=TEXT,
)
BODY_DIM = ParagraphStyle(
    "BodyDim", parent=BODY,
    textColor=DIM,
)
BODY_BOLD = ParagraphStyle(
    "BodyBold", parent=BODY, fontName="Helvetica-Bold",
)
FLOW_STEP = ParagraphStyle(
    "FlowStep", parent=BODY, fontSize=10, leading=14,
    spaceAfter=5, leftIndent=14, firstLineIndent=-14,
)

def p(text, style=BODY):
    return Paragraph(text, style)

def main():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=14*mm,
        title="Allosteric Demo — Live Cheat-Sheet",
        author="Hassan AL Habeeb",
    )
    story = []

    story.append(p("Live Demo Cheat-Sheet", H1))
    story.append(p(
        "Real site vs. model prediction · presentationsite-six.vercel.app",
        SUB,
    ))

    # --------- PROTEIN TABS ---------
    story.append(p("1. Protein tabs — pick which molecule", H2))
    story.append(p(
        "Each tab loads a different CASBench blind-test protein. "
        "Green badge = success case, red badge = failure mode. The AUROC on the "
        "badge tells you how well the model did on that protein.",
        BODY_DIM,
    ))
    story.append(Spacer(1, 6))

    prot_rows = [
        ["PDB", "AUROC", "Category", "What to say on stage"],
        ["1UU7", "0.99", "Success",
         "Clean success. High-AUROC case, pocket-bound allosteric site. "
         "Pocket geometry, dynamics, and ESM-2 all agree."],
        ["4KSQ", "0.96", "Success",
         "Kinase — the clinically critical family. Kinases are ~35% of drug "
         "targets, and the asciminib story (CML) lives here."],
        ["3ME3", "0.95", "Success",
         "<30% sequence identity to anything in training. Strong prediction "
         "here = real generalization, not memorization."],
        ["1HQ6", "0.26", "Failure",
         "The novel finding. AUROC 0.26 — worse than random. Model gets it "
         "backwards because the real allosteric site sits on a flat protein-"
         "protein interface, not in a pocket."],
        ["3W8L", "0.34", "Failure",
         "Second interface case, different enzyme family, same failure pattern. "
         "This is the pocket-bias blind spot affecting the whole field."],
    ]

    prot_tbl = Table(prot_rows, colWidths=[20*mm, 18*mm, 22*mm, 112*mm], repeatRows=1)
    prot_style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("TEXTCOLOR", (0,1), (-1,-1), TEXT),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (2,-1), "LEFT"),
        ("BOX", (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID", (0,0), (-1,-1), 0.25, BORDER),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("FONTNAME", (0,1), (0,-1), "Courier-Bold"),
        ("FONTNAME", (1,1), (1,-1), "Courier"),
    ])
    # Green text for success rows, red for fail rows (on category column)
    for i, row in enumerate(prot_rows[1:], start=1):
        if row[2] == "Success":
            prot_style.add("TEXTCOLOR", (2,i), (2,i), WIN)
            prot_style.add("TEXTCOLOR", (1,i), (1,i), WIN)
            prot_style.add("FONTNAME", (2,i), (2,i), "Helvetica-Bold")
        else:
            prot_style.add("TEXTCOLOR", (2,i), (2,i), FAIL)
            prot_style.add("TEXTCOLOR", (1,i), (1,i), FAIL)
            prot_style.add("FONTNAME", (2,i), (2,i), "Helvetica-Bold")
        if i % 2 == 0:
            prot_style.add("BACKGROUND", (0,i), (-1,i), ROW_ALT)
    prot_tbl.setStyle(prot_style)
    story.append(prot_tbl)

    # --------- MODE BAR ---------
    story.append(p("2. Mode bar — pick how the protein is colored", H2))
    story.append(p(
        "The mode bar at the top-right of the 3D viewer changes what the colors mean. "
        "Same protein, different questions.",
        BODY_DIM,
    ))
    story.append(Spacer(1, 6))

    mode_rows = [
        ["Mode", "What you see", "When to use it on stage"],
        ["Both (default)",
         "Green = true positive (model caught real site). "
         "Blue = false negative (real site the model missed). "
         "Red = false positive (model predicted, not real). "
         "Orange = active / catalytic site.",
         "Default for success cases. Say: \"Green = model caught it, "
         "blue = missed, red = false alarm, orange = active site.\""],
        ["Ground truth only",
         "Only real allosteric residues (green) + active site (orange). "
         "Hides everything the model did.",
         "\"This is what the experimental literature says is allosteric.\" "
         "Good to set up before revealing what the model predicted."],
        ["Prediction only",
         "Only residues the model flagged as allosteric. Green if correct, "
         "red if wrong.",
         "\"This is what the model produced.\" Especially powerful on 1HQ6 — "
         "you'll see almost nothing, proving the model didn't find the site."],
        ["Probability heatmap",
         "Every residue colored by predicted probability: dark blue (low) → "
         "yellow → red (high). No threshold applied.",
         "\"No threshold — this is the raw confidence gradient.\" Nice closer; "
         "shows allosteric regions as hot spots even on borderline cases."],
    ]
    mode_tbl = Table(mode_rows, colWidths=[32*mm, 68*mm, 72*mm], repeatRows=1)
    mode_style = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("TEXTCOLOR", (0,1), (-1,-1), TEXT),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,1), (0,-1), ACCENT),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("BOX", (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID", (0,0), (-1,-1), 0.25, BORDER),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ])
    for i in range(1, len(mode_rows)):
        if i % 2 == 0:
            mode_style.add("BACKGROUND", (0,i), (-1,i), ROW_ALT)
    mode_tbl.setStyle(mode_style)
    story.append(mode_tbl)

    # --------- COLOR LEGEND ---------
    story.append(p("3. Color legend (always true, any mode)", H2))
    legend = [
        ["<b><font color='#15803d'>Green</font></b>",
         "True positive — real allosteric residue the model correctly caught"],
        ["<b><font color='#1e40af'>Blue</font></b>",
         "False negative — real allosteric residue the model missed"],
        ["<b><font color='#dc2626'>Red</font></b>",
         "False positive — model predicted allosteric, not actually allosteric"],
        ["<b><font color='#b45309'>Orange</font></b>",
         "Active / catalytic site (where the enzyme does its main job)"],
        ["<b><font color='#6b7280'>Grey</font></b>",
         "Other residues (not allosteric, not predicted, not active)"],
    ]
    legend_tbl = Table(
        [[Paragraph(c, BODY), Paragraph(d, BODY)] for c, d in legend],
        colWidths=[20*mm, 152*mm],
    )
    legend_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(legend_tbl)

    story.append(PageBreak())

    # --------- SUGGESTED FLOW ---------
    story.append(p("Suggested demo flow (≈30–40 s per protein)", H2))
    story.append(p(
        "Open the site on whichever tab you land on first, then walk through "
        "in this order. Every tap on a protein tab resets to Both mode.",
        BODY_DIM,
    ))

    flow = [
        ("1. Open on 1UU7, mode = Both",
         "\"Here's a clean hit on a CASBench blind-test protein. "
         "Green dots are where the model correctly caught real allosteric residues.\""),
        ("2. Tap 4KSQ",
         "\"This is a kinase — the family where asciminib was approved for "
         "drug-resistant leukemia. Allosteric drugs here are the future of oncology.\""),
        ("3. Tap 3ME3",
         "\"This protein has less than 30% sequence identity to anything "
         "in my training set. The strong prediction proves the model learned "
         "structural-functional patterns, not just sequence memorization.\""),
        ("4. Tap 1HQ6, keep mode = Both",
         "\"Now watch what happens. AUROC 0.26 — worse than random. "
         "This is my novel finding.\""),
        ("5. Switch to Prediction only",
         "\"The model thought the active site was allosteric. It completely missed "
         "the real site. Why? The real site is on a flat protein-protein interface — "
         "no pocket for FPocket to detect, so the model has no pocket features to fire on.\""),
        ("6. Tap 3W8L",
         "\"Same pattern in a totally different enzyme family. "
         "This is the pocket-bias blind spot nobody has reported — and it affects "
         "every pocket-based predictor in the literature.\""),
        ("7. Tap back to 1UU7, switch to Probability heatmap",
         "\"And finally, here's the raw confidence without any threshold. "
         "The hot spots on this protein exactly match the real allosteric site.\""),
    ]
    for step, line in flow:
        story.append(p(f"<b>{step}</b> — {line}", FLOW_STEP))

    # --------- QUICK TECH RECAP ---------
    story.append(p("One-liner technical recap (in case someone asks)", H2))
    recap = [
        ("Model", "XGBoost, 225 features per residue, Optuna-tuned."),
        ("Training", "2,043 proteins from AlloBench (10× prior work)."),
        ("Blind test", "2,370 independent CASBench proteins across 91 enzyme families."),
        ("Threshold", "0.343 — optimized on validation set for best F1."),
        ("Headline result", "Test AUROC 0.926, CASBench AUROC 0.796. "
                            "Low-homology (<30% ID) subset actually scored higher (0.789)."),
        ("Novel finding", "Per-family stratification reveals 5 enzyme families with "
                          "anti-correlated predictions — all interface-mediated allostery. "
                          "FPocket features (31% importance) return null on flat surfaces."),
    ]
    recap_tbl = Table(
        [[Paragraph(f"<b>{k}</b>", BODY_BOLD), Paragraph(v, BODY)] for k, v in recap],
        colWidths=[34*mm, 138*mm],
    )
    recap_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOX", (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID", (0,0), (-1,-1), 0.25, BORDER),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, ROW_ALT]),
    ]))
    story.append(recap_tbl)

    doc.build(story)
    print(f"PDF written to {OUT}")


if __name__ == "__main__":
    main()
