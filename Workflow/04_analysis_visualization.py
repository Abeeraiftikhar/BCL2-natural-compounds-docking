"""
Script 04: Analysis & Visualization
- Load docking results
- Calculate drug-likeness (Lipinski Ro5) for all compounds
- Generate publication-quality figures:
  1. Horizontal bar chart: binding affinities
  2. Radar chart: drug-likeness properties
  3. 2D molecular structure grid
  4. Interaction summary heatmap
- Save final summary CSV
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

SESSION = "/app/sandbox/session_20260309_105257_f9338604305e"
DATA_DIR = f"{SESSION}/data"
RESULTS_DIR = f"{SESSION}/results"
FIGURES_DIR = f"{SESSION}/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Minimal, clean visual style
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Muted color palette
PALETTE = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#B279A2"]

print("=" * 60)
print("STEP 4: Analysis and Visualization")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/compounds_info.json") as f:
    compounds = json.load(f)

with open(f"{RESULTS_DIR}/docking_results.json") as f:
    docking = json.load(f)

comp_df = pd.DataFrame(compounds)
dock_df = pd.DataFrame(docking)

# Merge
df = dock_df.merge(comp_df.rename(columns={"name": "compound"}),
                   on="compound", how="left")
df_success = df[df["status"] == "success"].copy()
df_success = df_success.sort_values("best_score_kcal_mol")

print(f"\n  Loaded {len(df_success)} successful docking results")

# ─────────────────────────────────────────────────────────
# Drug-likeness calculations
# ─────────────────────────────────────────────────────────
print("\n  Calculating drug-likeness properties...")

def calc_drug_likeness(smiles):
    """Calculate Lipinski Ro5 and additional properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "ArRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Ro5_violations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Descriptors.NumHDonors(mol) > 5,
            Descriptors.NumHAcceptors(mol) > 10,
        ]),
        "Ro5_pass": (
            Descriptors.MolWt(mol) <= 500 and
            Descriptors.MolLogP(mol) <= 5 and
            Descriptors.NumHDonors(mol) <= 5 and
            Descriptors.NumHAcceptors(mol) <= 10
        )
    }

dl_data = []
for _, row in df_success.iterrows():
    smiles = row.get("smiles", "")
    props = calc_drug_likeness(smiles)
    props["compound"] = row["compound"]
    props["binding_affinity"] = row["best_score_kcal_mol"]
    dl_data.append(props)

dl_df = pd.DataFrame(dl_data)
print(f"\n  Drug-likeness summary:")
print(dl_df[["compound", "MW", "LogP", "HBD", "HBA", "TPSA", "Ro5_pass"]].to_string(index=False))

# ─────────────────────────────────────────────────────────
# FIGURE 1: Binding Affinity Bar Chart
# ─────────────────────────────────────────────────────────
print("\n  Creating Figure 1: Binding Affinity Chart...")

fig, ax = plt.subplots(figsize=(8, 4.5))

compounds_sorted = dl_df.sort_values("binding_affinity")
names = compounds_sorted["compound"].tolist()
scores = compounds_sorted["binding_affinity"].tolist()
bar_colors = PALETTE[:len(names)]

bars = ax.barh(names, scores, color=bar_colors, height=0.55, alpha=0.85, edgecolor="none")

# Add value labels
for bar, score in zip(bars, scores):
    ax.text(score - 0.05, bar.get_y() + bar.get_height()/2,
            f"{score:.3f}", va="center", ha="right",
            fontsize=9, color="white", fontweight="bold")

ax.set_xlabel("Binding Affinity (kcal/mol)", fontsize=11, labelpad=8)
ax.set_title("Binding Affinity of Natural Compounds to BCL-2\n(AutoDock Vina · PDB: 6GL8)",
             fontsize=12, fontweight="bold", pad=12)

# Reference line for reference
ax.axvline(x=-7.0, color="#999999", lw=0.8, linestyle="--", alpha=0.7)
ax.text(-7.0, -0.5, "-7.0", fontsize=7, color="#999999", ha="center")

ax.set_xlim(min(scores) - 1.0, -4.0)
ax.tick_params(axis="y", length=0)
ax.set_axisbelow(True)
ax.xaxis.grid(True, alpha=0.3, lw=0.5)

# Legend note
ax.text(0.98, 0.02, "More negative = stronger binding",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#666666", style="italic")

plt.tight_layout(pad=1.5)
fig.savefig(f"{FIGURES_DIR}/binding_affinity.png")
plt.close()
print(f"  Saved: {FIGURES_DIR}/binding_affinity.png")

# ─────────────────────────────────────────────────────────
# FIGURE 2: Drug-Likeness Radar Chart
# ─────────────────────────────────────────────────────────
print("  Creating Figure 2: Drug-Likeness Radar Chart...")

# Properties to show (normalized to 0-1 scale)
RADAR_PROPS = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds"]
LIMITS = {"MW": 500, "LogP": 5, "HBD": 5, "HBA": 10, "TPSA": 140, "RotBonds": 10}
LABELS = ["MW\n(≤500 Da)", "LogP\n(≤5)", "HBD\n(≤5)", "HBA\n(≤10)", "TPSA\n(≤140 Å²)", "Rot. Bonds\n(≤10)"]

N = len(RADAR_PROPS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for i, (_, row) in enumerate(dl_df.iterrows()):
    values = [min(row[p] / LIMITS[p], 1.2) for p in RADAR_PROPS]
    values += values[:1]
    ax.plot(angles, values, "o-", lw=1.8, color=PALETTE[i % len(PALETTE)],
            label=row["compound"], alpha=0.85, ms=4)
    ax.fill(angles, values, color=PALETTE[i % len(PALETTE)], alpha=0.07)

# Lipinski limit line (all at 1.0)
limit_vals = [1.0] * N + [1.0]
ax.plot(angles, limit_vals, "--", lw=1.2, color="#444444", alpha=0.5, label="Ro5 Limit")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(LABELS, size=9)
ax.set_ylim(0, 1.3)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color="#888")
ax.yaxis.set_tick_params(labelsize=7)
ax.set_rlabel_position(10)

ax.spines["polar"].set_alpha(0.2)
ax.grid(alpha=0.25, lw=0.5)

ax.set_title("Drug-Likeness Profile (Lipinski Ro5)\nNormalized to Ro5 Thresholds",
             fontsize=12, fontweight="bold", pad=20)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15),
          ncol=3, fontsize=9, frameon=False,
          title="Compounds", title_fontsize=9)

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/drug_likeness_radar.png")
plt.close()
print(f"  Saved: {FIGURES_DIR}/drug_likeness_radar.png")

# ─────────────────────────────────────────────────────────
# FIGURE 3: 2D Molecular Structure Grid
# ─────────────────────────────────────────────────────────
print("  Creating Figure 3: Molecular Structures Grid...")

# Use rdkit Draw with clean styling
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

with open(f"{DATA_DIR}/compounds_info.json") as f:
    all_compounds = json.load(f)

# Sort compounds in same order as docking results (by affinity)
compound_order = dl_df.sort_values("binding_affinity")["compound"].tolist()
comp_smiles = {c["name"]: c["smiles"] for c in all_compounds}

for i, cname in enumerate(compound_order):
    if i >= 5:
        break
    ax = axes[i]
    smiles = comp_smiles.get(cname, "")
    score = dl_df[dl_df["compound"] == cname]["binding_affinity"].values[0]
    ro5_pass = dl_df[dl_df["compound"] == cname]["Ro5_pass"].values[0]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(280, 180)
            drawer.drawOptions().addStereoAnnotation = False
            drawer.drawOptions().bondLineWidth = 1.5
            drawer.drawOptions().atomLabelFontSize = 0.6
            drawer.drawOptions().padding = 0.15
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            # Convert SVG to PNG via PIL
            from PIL import Image as PILImage
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg.encode())
            img = PILImage.open(io.BytesIO(png_data))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Structure\nunavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
    except Exception as e:
        # Fallback: use rdkit's MolToImage
        try:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(280, 180))
            ax.imshow(img)
        except Exception as e2:
            ax.text(0.5, 0.5, f"{cname}\n(structure unavailable)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

    ax.axis("off")
    rank = compound_order.index(cname) + 1
    ro5_text = "Ro5 ✓" if ro5_pass else "Ro5 ✗"
    ax.set_title(
        f"#{rank}: {cname}\n{score:.3f} kcal/mol | {ro5_text}",
        fontsize=10, fontweight="bold" if rank == 1 else "normal",
        color=PALETTE[i % len(PALETTE)], pad=4
    )

# Hide the 6th subplot (empty)
axes[5].axis("off")

fig.suptitle("2D Structures of Natural Compounds Screened Against BCL-2",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(pad=1.5)
fig.savefig(f"{FIGURES_DIR}/molecular_structures.png")
plt.close()
print(f"  Saved: {FIGURES_DIR}/molecular_structures.png")

# ─────────────────────────────────────────────────────────
# FIGURE 4: Properties Table Heatmap
# ─────────────────────────────────────────────────────────
print("  Creating Figure 4: Properties Summary Table...")

table_cols = ["Binding\nAffinity\n(kcal/mol)", "MW\n(Da)", "LogP", "HBD", "HBA",
              "TPSA\n(Å²)", "Rot.\nBonds", "Ro5\nPass"]
table_data = []
for _, row in dl_df.sort_values("binding_affinity").iterrows():
    table_data.append([
        f"{row['binding_affinity']:.3f}",
        f"{row['MW']:.1f}",
        f"{row['LogP']:.2f}",
        f"{int(row['HBD'])}",
        f"{int(row['HBA'])}",
        f"{row['TPSA']:.1f}",
        f"{int(row['RotBonds'])}",
        "Yes" if row["Ro5_pass"] else "No"
    ])

row_labels = dl_df.sort_values("binding_affinity")["compound"].tolist()

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis("off")

tbl = ax.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=table_cols,
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.3, 1.7)

# Style header
for (r, c), cell in tbl.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor("#E8EDF3")
        cell.set_text_props(fontweight="bold", fontsize=8.5)
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#CCCCCC")
    cell.set_linewidth(0.5)

# Color best affinity row
for c in range(len(table_cols)):
    tbl[1, c].set_facecolor("#EAF4EA")

# Color Ro5 pass column
for r in range(1, len(table_data) + 1):
    val = table_data[r-1][-1]
    if val == "Yes":
        tbl[r, len(table_cols)-1].set_facecolor("#D6EDDA")
    else:
        tbl[r, len(table_cols)-1].set_facecolor("#FDE8E8")

ax.set_title("BCL-2 Docking Results & Drug-Likeness Properties",
             fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/properties_table.png")
plt.close()
print(f"  Saved: {FIGURES_DIR}/properties_table.png")

# ─────────────────────────────────────────────────────────
# FIGURE 5: Combined Summary Dashboard
# ─────────────────────────────────────────────────────────
print("  Creating Figure 5: Summary Dashboard...")

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax_bar = fig.add_subplot(gs[0, :2])
ax_radar = fig.add_subplot(gs[:, 2], polar=True)
ax_scatter = fig.add_subplot(gs[1, :2])

# ── Panel A: Binding Affinity Bars ──────────────────────
compounds_sorted = dl_df.sort_values("binding_affinity")
names = compounds_sorted["compound"].tolist()
scores = compounds_sorted["binding_affinity"].tolist()

bars = ax_bar.barh(names, scores, color=PALETTE[:len(names)], height=0.5,
                    alpha=0.82, edgecolor="none")
for bar, score in zip(bars, scores):
    ax_bar.text(score - 0.04, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")

ax_bar.set_xlabel("Binding Affinity (kcal/mol)", fontsize=9)
ax_bar.set_title("A. Binding Affinities", fontsize=10, fontweight="bold", loc="left")
ax_bar.set_xlim(min(scores) - 0.8, -4.5)
ax_bar.tick_params(axis="y", length=0, labelsize=9)
ax_bar.xaxis.grid(True, alpha=0.3, lw=0.5)
ax_bar.axvline(-7.0, color="#aaa", lw=0.8, ls="--", alpha=0.6)

# ── Panel B: MW vs LogP Scatter ─────────────────────────
for i, (_, row) in enumerate(dl_df.iterrows()):
    ax_scatter.scatter(row["MW"], row["LogP"], s=80,
                       color=PALETTE[i % len(PALETTE)], zorder=5,
                       edgecolors="white", lw=0.5)
    ax_scatter.annotate(row["compound"], (row["MW"], row["LogP"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5, color="#444")

# Lipinski limits
ax_scatter.axvline(500, color="#ccc", lw=1, ls="--")
ax_scatter.axhline(5, color="#ccc", lw=1, ls="--")
ax_scatter.fill_between([0, 500], 0, 5, alpha=0.05, color="green")
ax_scatter.text(490, 4.8, "Ro5 zone", ha="right", va="top", fontsize=7.5, color="#666")
ax_scatter.set_xlabel("Molecular Weight (Da)", fontsize=9)
ax_scatter.set_ylabel("LogP", fontsize=9)
ax_scatter.set_title("B. Chemical Space (MW vs LogP)", fontsize=10, fontweight="bold", loc="left")

# ── Panel C: Radar ───────────────────────────────────────
for i, (_, row) in enumerate(dl_df.iterrows()):
    values = [min(row[p] / LIMITS[p], 1.2) for p in RADAR_PROPS]
    values += values[:1]
    ax_radar.plot(angles, values, "o-", lw=1.5,
                  color=PALETTE[i % len(PALETTE)], label=row["compound"],
                  alpha=0.85, ms=3)
    ax_radar.fill(angles, values, color=PALETTE[i % len(PALETTE)], alpha=0.06)

ax_radar.plot(angles, limit_vals, "--", lw=1, color="#888", alpha=0.5)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds"], size=7.5)
ax_radar.set_ylim(0, 1.3)
ax_radar.set_yticks([0.5, 1.0])
ax_radar.set_yticklabels(["50%", "100%"], size=6, color="#888")
ax_radar.grid(alpha=0.2, lw=0.5)
ax_radar.spines["polar"].set_alpha(0.15)
ax_radar.set_title("C. Drug-Likeness\n(Ro5 Normalized)", fontsize=10, fontweight="bold", pad=15)
ax_radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
                ncol=2, fontsize=7, frameon=False)

fig.suptitle("Computational Screening of Natural Compounds Against BCL-2 Protein\n"
             "PDB: 6GL8 (1.40 Å) · AutoDock Vina · 5 Candidates",
             fontsize=12, fontweight="bold", y=1.01)

fig.savefig(f"{FIGURES_DIR}/summary_dashboard.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/summary_dashboard.png")

# ─────────────────────────────────────────────────────────
# Save Final Summary CSV
# ─────────────────────────────────────────────────────────
print("\n  Saving final summary table...")

final_df = dl_df.sort_values("binding_affinity").reset_index(drop=True)
final_df.insert(0, "Rank", range(1, len(final_df) + 1))

# Add source info
source_map = {c["name"]: c["source"] for c in all_compounds}
final_df["Source"] = final_df["compound"].map(source_map)

final_df.to_csv(f"{RESULTS_DIR}/final_summary.csv", index=False)
print(f"  Saved: {RESULTS_DIR}/final_summary.csv")

# ─────────────────────────────────────────────────────────
# Print Summary Table
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Rank':<5} {'Compound':<13} {'Affinity':<12} {'MW':<8} {'LogP':<7} {'Ro5'}")
print("-" * 52)
for _, row in final_df.iterrows():
    ro5 = "PASS" if row["Ro5_pass"] else "FAIL"
    print(f"{int(row['Rank']):<5} {row['compound']:<13} {row['binding_affinity']:<12.3f} "
          f"{row['MW']:<8.1f} {row['LogP']:<7.2f} {ro5}")

print(f"\n  Top candidate: {final_df.iloc[0]['compound']} "
      f"({final_df.iloc[0]['binding_affinity']:.3f} kcal/mol)")

# Figure summary
print(f"\n  Figures saved to {FIGURES_DIR}/:")
for fname in sorted(os.listdir(FIGURES_DIR)):
    fpath = f"{FIGURES_DIR}/{fname}"
    sz = os.path.getsize(fpath)
    print(f"    {fname} ({sz/1024:.1f} KB)")

print("\nScript 04 complete.")
