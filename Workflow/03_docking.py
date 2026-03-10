"""
Script 03: Molecular Docking with AutoDock Vina
- Load prepared PDBQT files for protein and all 5 ligands
- Define grid box from co-crystallized ligand centroid
- Run AutoDock Vina for each compound
- Save docking scores and best poses
"""

import os
import sys
import json
import time
import pandas as pd

SESSION = "/app/sandbox/session_20260309_105257_f9338604305e"
DATA_DIR = f"{SESSION}/data"
RESULTS_DIR = f"{SESSION}/results"
DOCKING_DIR = f"{RESULTS_DIR}/docking"
os.makedirs(DOCKING_DIR, exist_ok=True)

print("=" * 60)
print("STEP 3: Molecular Docking with AutoDock Vina")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# Load box parameters and ligand info
# ─────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/box_params.json") as f:
    box = json.load(f)

with open(f"{DATA_DIR}/ligand_info.json") as f:
    ligand_info = json.load(f)

PROTEIN_PDBQT = f"{DATA_DIR}/protein.pdbqt"

print(f"\n  Protein: {PROTEIN_PDBQT}")
print(f"  Box center: ({box['center_x']:.2f}, {box['center_y']:.2f}, {box['center_z']:.2f})")
print(f"  Box size: {box['size_x']} x {box['size_y']} x {box['size_z']} Å")
print(f"  Ligands to dock: {len(ligand_info)}")

# ─────────────────────────────────────────────────────────
# Initialize Vina
# ─────────────────────────────────────────────────────────
from vina import Vina

BOX_CENTER = [box["center_x"], box["center_y"], box["center_z"]]
BOX_SIZE = [box["size_x"], box["size_y"], box["size_z"]]
print("\n  AutoDock Vina will be initialized per-ligand.")

# ─────────────────────────────────────────────────────────
# Run docking for each compound
# ─────────────────────────────────────────────────────────
print("\n  Starting docking runs...\n")

docking_results = []
EXHAUSTIVENESS = 8
N_POSES = 5

for i, lig in enumerate(ligand_info):
    name = lig["name"]
    pdbqt_path = lig.get("pdbqt_path", "")
    status = lig.get("status", "failed")

    print(f"  [{i+1}/{len(ligand_info)}] Docking {name}...")

    if status == "failed" or not pdbqt_path or not os.path.exists(pdbqt_path):
        print(f"    SKIP: PDBQT not available")
        docking_results.append({
            "compound": name, "best_score_kcal_mol": None,
            "second_score": None, "third_score": None, "status": "skipped"
        })
        continue

    # Check PDBQT file is valid
    with open(pdbqt_path) as f:
        content = f.read()
    if len(content) < 50:
        print(f"    SKIP: PDBQT file too small ({len(content)} bytes)")
        continue

    t_start = time.time()
    try:
        # Re-initialize Vina for each ligand (required)
        v2 = Vina(sf_name="vina", verbosity=0)
        v2.set_receptor(PROTEIN_PDBQT)
        v2.set_ligand_from_file(pdbqt_path)

        # Compute affinity maps (defines the search box)
        v2.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)

        # Run docking
        v2.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES)

        # Get energies
        energies = v2.energies(n_poses=N_POSES)

        # Extract binding affinities (first column = total energy)
        scores = [round(e[0], 3) for e in energies]
        best_score = scores[0]

        t_elapsed = time.time() - t_start
        print(f"    Best score: {best_score:.3f} kcal/mol | Time: {t_elapsed:.1f}s")
        print(f"    All poses: {scores}")

        # Save best pose
        out_pdbqt = f"{DOCKING_DIR}/{name}_docked.pdbqt"
        v2.write_poses(out_pdbqt, n_poses=1, overwrite=True)
        print(f"    Best pose saved: {out_pdbqt}")

        docking_results.append({
            "compound": name,
            "best_score_kcal_mol": best_score,
            "pose_2_score": scores[1] if len(scores) > 1 else None,
            "pose_3_score": scores[2] if len(scores) > 2 else None,
            "n_poses": len(scores),
            "pose_file": out_pdbqt,
            "status": "success",
            "docking_time_s": round(t_elapsed, 2)
        })

    except Exception as e:
        t_elapsed = time.time() - t_start
        print(f"    ERROR after {t_elapsed:.1f}s: {e}")
        docking_results.append({
            "compound": name, "best_score_kcal_mol": None,
            "status": "error", "error_msg": str(e)
        })

    print()

# ─────────────────────────────────────────────────────────
# Compile and save results
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("DOCKING RESULTS SUMMARY")
print("=" * 60)

df = pd.DataFrame(docking_results)
df_success = df[df["status"] == "success"].copy()

if len(df_success) > 0:
    df_success = df_success.sort_values("best_score_kcal_mol")

    print("\n  Ranking by binding affinity (most negative = strongest):")
    print(f"  {'Rank':<5} {'Compound':<15} {'Affinity (kcal/mol)'}")
    print("  " + "-" * 40)
    for rank, (_, row) in enumerate(df_success.iterrows(), 1):
        print(f"  {rank:<5} {row['compound']:<15} {row['best_score_kcal_mol']:.3f}")
else:
    print("\n  WARNING: No successful docking results!")

# Merge with compound properties
with open(f"{DATA_DIR}/compounds_info.json") as f:
    compounds = json.load(f)

comp_df = pd.DataFrame(compounds)
if "name" in comp_df.columns:
    comp_df = comp_df.rename(columns={"name": "compound"})

# Merge docking scores with compound info
results_merged = df.merge(
    comp_df[["compound", "molecular_weight", "xlogp", "hbd", "hba", "tpsa", "rotatable_bonds"]],
    on="compound", how="left"
)

# Save full results
csv_path = f"{RESULTS_DIR}/docking_results.csv"
results_merged.to_csv(csv_path, index=False)
print(f"\n  Full results saved: {csv_path}")

# Save JSON for visualization script
with open(f"{RESULTS_DIR}/docking_results.json", "w") as f:
    json.dump(docking_results, f, indent=2)

print(f"  JSON results saved: {RESULTS_DIR}/docking_results.json")

# Validate success
success_count = sum(1 for r in docking_results if r.get("status") == "success")
print(f"\n  Successful dockings: {success_count}/{len(ligand_info)}")

if success_count == 0:
    print("\n  CRITICAL: All dockings failed!")
    sys.exit(1)

print("\nScript 03 complete.")
