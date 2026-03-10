"""
Script 02: Structure Preparation
- Clean BCL-2 PDB (remove waters, keep protein + reference ligand for binding site)
- Calculate binding site box from co-crystallized ligand (F3Q) centroid
- Generate 3D conformers for 5 natural compounds using RDKit
- Convert all structures to PDBQT format using meeko
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB import is_aa

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors

# meeko imports
import meeko

SESSION = "/app/sandbox/session_20260309_105257_f9338604305e"
DATA_DIR = f"{SESSION}/data"
LIGANDS_DIR = f"{DATA_DIR}/ligands"
os.makedirs(LIGANDS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 2A: Protein Structure Preparation")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# Parse and clean the BCL-2 PDB file
# ─────────────────────────────────────────────────────────
PDB_RAW = f"{DATA_DIR}/protein_raw.pdb"
PDB_CLEAN = f"{DATA_DIR}/protein_clean.pdb"
PDB_PDBQT = f"{DATA_DIR}/protein.pdbqt"

# Read PDB manually for better control
with open(PDB_RAW) as f:
    raw_lines = f.readlines()

# Extract all ATOM records (protein)
protein_atoms = [l for l in raw_lines if l.startswith("ATOM")]

# Extract reference ligand F3Q for binding site definition
f3q_atoms = [l for l in raw_lines if l.startswith("HETATM") and "F3Q" in l]

print(f"  Protein ATOM records: {len(protein_atoms)}")
print(f"  Co-crystallized ligand (F3Q) atoms: {len(f3q_atoms)}")

# Calculate binding site centroid from F3Q
if f3q_atoms:
    coords = []
    for line in f3q_atoms:
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
        except Exception:
            pass
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    print(f"  Binding site centroid (F3Q): x={centroid[0]:.2f}, y={centroid[1]:.2f}, z={centroid[2]:.2f}")
else:
    # Use known BCL-2 BH3 groove coordinates as fallback
    centroid = np.array([-5.0, 5.0, -8.0])
    print(f"  Using default BCL-2 binding site: {centroid}")

# Box size (generous to cover BH3 binding groove)
BOX_SIZE = [22.0, 22.0, 22.0]
box_center = centroid.tolist()

# Save box parameters
box_params = {
    "center_x": round(box_center[0], 3),
    "center_y": round(box_center[1], 3),
    "center_z": round(box_center[2], 3),
    "size_x": BOX_SIZE[0],
    "size_y": BOX_SIZE[1],
    "size_z": BOX_SIZE[2]
}
with open(f"{DATA_DIR}/box_params.json", "w") as f:
    json.dump(box_params, f, indent=2)
print(f"  Box parameters saved: {box_params}")

# Write clean PDB (protein ATOM records only)
HEADER = [l for l in raw_lines if l.startswith("HEADER") or l.startswith("TITLE") or l.startswith("CRYST")]
with open(PDB_CLEAN, "w") as f:
    for l in HEADER:
        f.write(l)
    for l in protein_atoms:
        f.write(l)
    f.write("END\n")

print(f"  Clean protein PDB saved: {PDB_CLEAN}")

# ─────────────────────────────────────────────────────────
# Convert protein to PDBQT using meeko's protein prep
# ─────────────────────────────────────────────────────────
print("\n  Converting protein to PDBQT format...")

try:
    from meeko import PDBQTWriterLegacy
    from meeko import MoleculePreparation
    print("  Using meeko for protein PDBQT preparation...")
    # meeko doesn't directly prepare proteins from PDB in all versions
    # We'll use a custom approach instead
    raise ImportError("Using custom approach")
except Exception:
    pass

# Custom protein PDB -> PDBQT converter
# Assigns AutoDock atom types based on element + residue context
AD_ATOM_TYPES = {
    "C": "C", "N": "N", "O": "OA", "S": "SA",
    "H": "H", "P": "P", "F": "F", "Cl": "CL",
    "Br": "BR", "I": "I", "CA": "CA",
}

ATOM_CHARGE_APPROX = {
    "C": 0.0, "N": -0.3, "O": -0.3, "S": -0.1,
    "H": 0.1, "P": 0.3, "F": -0.1, "CA": 0.0,
}

def pdb_to_pdbqt_protein(pdb_file, pdbqt_file):
    """Convert a cleaned protein PDB to PDBQT format."""
    with open(pdb_file) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        # Parse atom fields
        record = line[:6].strip()
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21]
        res_seq = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()
        occ = line[54:60].strip() if len(line) > 54 else "1.00"
        bfac = line[60:66].strip() if len(line) > 60 else "0.00"

        # Determine element
        element = ""
        if len(line) >= 78:
            element = line[76:78].strip()
        if not element:
            element = atom_name[0] if atom_name else "C"
            if atom_name[:2] in ["CA", "MG", "ZN", "FE", "CU"]:
                element = atom_name[:2]

        # AutoDock atom type
        ad_type = AD_ATOM_TYPES.get(element, element[:2] if len(element) > 1 else element)
        charge = ATOM_CHARGE_APPROX.get(element, 0.0)

        # Format PDBQT line
        pdbqt_line = (
            f"{record:<6}{line[6:12]}{atom_name:>4} {res_name:<3} {chain}"
            f"{res_seq:>4}    {float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
            f"{float(occ):6.2f}{float(bfac):6.2f}"
            f"    {charge:6.3f} {ad_type:<2}\n"
        )
        out_lines.append(pdbqt_line)

    with open(pdbqt_file, "w") as f:
        f.writelines(out_lines)

    return len(out_lines)

n_atoms = pdb_to_pdbqt_protein(PDB_CLEAN, PDB_PDBQT)
print(f"  Protein PDBQT saved: {PDB_PDBQT} ({n_atoms} atoms)")

# ─────────────────────────────────────────────────────────
# STEP 2B: Ligand Preparation
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2B: Ligand 3D Structure Generation")
print("=" * 60)

# Load compound info
with open(f"{DATA_DIR}/compounds_info.json") as f:
    compounds = json.load(f)

print(f"  Preparing {len(compounds)} ligands...\n")

# Initialize meeko MoleculePreparation
from meeko import MoleculePreparation

mk_prep = MoleculePreparation()

ligand_info = []

for i, comp in enumerate(compounds):
    name = comp["name"]
    smiles = comp.get("smiles", "")
    print(f"  [{i+1}/{len(compounds)}] {name}")

    if not smiles:
        print(f"    ERROR: No SMILES for {name}")
        continue

    # Fix SMILES if needed (remove old canonical form issues)
    # Use isomeric SMILES if available
    smiles_to_use = smiles

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles_to_use)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles_to_use}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D conformer using ETKDGv3
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 4
        result = AllChem.EmbedMolecule(mol, params)

        if result == -1:
            # Fall back to EmbedMolecule with ETKDG
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
        if result == -1:
            result = AllChem.EmbedMolecule(mol)

        if result == -1:
            raise ValueError("Could not generate 3D conformer")

        # Energy minimize with MMFF94
        ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        if ff_result == 1:  # MMFF failed, try UFF
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

        print(f"    3D conformer generated (MMFF94 optimized)")

        # Save SDF
        sdf_path = f"{LIGANDS_DIR}/{name}.sdf"
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        print(f"    SDF saved: {sdf_path}")

        # Convert to PDBQT using meeko
        pdbqt_path = f"{LIGANDS_DIR}/{name}.pdbqt"

        try:
            mk_prep.prepare(mol)
            pdbqt_string = mk_prep.write_pdbqt_string()
            with open(pdbqt_path, "w") as f:
                f.write(pdbqt_string)
            print(f"    PDBQT saved: {pdbqt_path}")
            ligand_info.append({
                "name": name,
                "smiles": smiles,
                "sdf_path": sdf_path,
                "pdbqt_path": pdbqt_path,
                "status": "success"
            })
        except Exception as e_meeko:
            print(f"    meeko error: {e_meeko}, trying alternative PDBQT generation...")
            # Alternative: use meeko MoleculePreparation differently
            try:
                from meeko import MoleculePreparation
                prep2 = MoleculePreparation()
                prep2.prepare(mol)
                pdbqt_str = prep2.write_pdbqt_string()
                with open(pdbqt_path, "w") as f:
                    f.write(pdbqt_str)
                print(f"    PDBQT saved (alt method): {pdbqt_path}")
                ligand_info.append({
                    "name": name, "smiles": smiles,
                    "sdf_path": sdf_path, "pdbqt_path": pdbqt_path,
                    "status": "success"
                })
            except Exception as e2:
                print(f"    Both meeko methods failed: {e2}")
                # Generate minimal PDBQT from SDF manually
                pdbqt_path = convert_sdf_to_pdbqt_manual(sdf_path, pdbqt_path, name)
                ligand_info.append({
                    "name": name, "smiles": smiles,
                    "sdf_path": sdf_path, "pdbqt_path": pdbqt_path,
                    "status": "manual_pdbqt"
                })

    except Exception as e:
        print(f"    ERROR: {e}")
        ligand_info.append({"name": name, "smiles": smiles, "status": "failed"})
        continue

    print()

# ─────────────────────────────────────────────────────────
# Verify outputs
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STRUCTURE PREPARATION SUMMARY")
print("=" * 60)
print(f"\n  Protein PDBQT: {PDB_PDBQT}")
print(f"  Box center: {box_center}")
print(f"  Box size: {BOX_SIZE}")
print(f"\n  Ligands prepared:")
for info in ligand_info:
    status = info.get("status", "unknown")
    pdbqt = info.get("pdbqt_path", "N/A")
    exists = os.path.exists(pdbqt) if pdbqt != "N/A" else False
    size = os.path.getsize(pdbqt) if exists else 0
    print(f"    {info['name']:15s} | {status:15s} | PDBQT: {size} bytes")

# Save ligand info for docking script
with open(f"{DATA_DIR}/ligand_info.json", "w") as f:
    json.dump(ligand_info, f, indent=2)

print(f"\n  Ligand info saved: {DATA_DIR}/ligand_info.json")
print("\nScript 02 complete.")
