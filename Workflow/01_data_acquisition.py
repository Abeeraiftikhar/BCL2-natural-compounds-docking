"""
Script 01: Data Acquisition
- Search PDB for best BCL-2 crystal structure
- Retrieve 5 natural compounds from PubChem
- Save compound metadata and download PDB file
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import pubchempy as pcp

SESSION = "/app/sandbox/session_20260309_105257_f9338604305e"
DATA_DIR = f"{SESSION}/data"
RESULTS_DIR = f"{SESSION}/results"
LOGS_DIR = f"{SESSION}/logs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# STEP 1A: Search PDB for best BCL-2 structure
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1A: Searching PDB for BCL-2 crystal structures")
print("=" * 60)

# Known high-quality BCL-2 structures to evaluate
BCL2_CANDIDATES = [
    {"pdb_id": "6GL8", "description": "BCL-2 + venetoclax, 1.79 Å"},
    {"pdb_id": "6O0K", "description": "BCL-2 + venetoclax analog"},
    {"pdb_id": "4LVT", "description": "BCL-2 + ABT-199 precursor, 2.58 Å"},
    {"pdb_id": "2O2F", "description": "BCL-2 + BH3 peptide, 1.76 Å"},
    {"pdb_id": "3IO9", "description": "BCL-2 BH3 binding domain"},
]

def get_pdb_metadata(pdb_id):
    """Retrieve metadata for a PDB entry via RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Warning: Could not fetch {pdb_id}: {e}")
    return None

def check_pdb_entity(pdb_id):
    """Check if PDB entry contains human BCL-2 protein."""
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # Check for BCL-2 in entity name / description
            entity_name = str(data.get("rcsb_polymer_entity", {}).get("pdbx_description", "")).upper()
            source = data.get("rcsb_entity_source_organism", [{}])
            organism = ""
            if source:
                organism = source[0].get("scientific_name", "")
            return entity_name, organism
    except Exception:
        pass
    return "", ""

# Evaluate candidates
print("\nEvaluating BCL-2 crystal structure candidates...")
best_pdb = None
best_resolution = 999.0
candidates_info = []

for cand in BCL2_CANDIDATES:
    pdb_id = cand["pdb_id"]
    print(f"\n  Checking {pdb_id}: {cand['description']}")
    meta = get_pdb_metadata(pdb_id)
    if meta:
        res = meta.get("rcsb_entry_info", {}).get("resolution_combined")
        method = meta.get("exptl", [{}])[0].get("method", "unknown")
        title = meta.get("struct", {}).get("title", "")
        # resolution_combined can be a list
        if isinstance(res, list):
            res = res[0] if res else None
        print(f"    Resolution: {res} Å | Method: {method}")
        print(f"    Title: {title[:80]}")
        if res and float(res) < best_resolution:
            best_resolution = float(res)
            best_pdb = pdb_id
        candidates_info.append({
            "pdb_id": pdb_id, "resolution": res, "method": method,
            "title": title, "description": cand["description"]
        })
    else:
        # Fall back to requesting directly
        candidates_info.append({
            "pdb_id": pdb_id, "resolution": None,
            "description": cand["description"]
        })
    time.sleep(0.5)

# Default to 6GL8 (known excellent BCL-2 structure) if auto-detection fails
if not best_pdb:
    best_pdb = "6GL8"
    print(f"\n  Auto-detection failed; defaulting to {best_pdb}")

print(f"\n>>> Selected PDB: {best_pdb} (resolution: {best_resolution:.2f} Å)")

# ─────────────────────────────────────────────────────────
# STEP 1B: Download selected PDB file
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 1B: Downloading PDB {best_pdb}")
print("=" * 60)

pdb_url = f"https://files.rcsb.org/download/{best_pdb}.pdb"
pdb_path = f"{DATA_DIR}/protein_raw.pdb"

r = requests.get(pdb_url, timeout=30)
if r.status_code == 200:
    with open(pdb_path, "w") as f:
        f.write(r.text)
    atom_lines = [l for l in r.text.split("\n") if l.startswith("ATOM")]
    hetatm_lines = [l for l in r.text.split("\n") if l.startswith("HETATM")]
    print(f"  Downloaded {best_pdb}.pdb")
    print(f"  ATOM records: {len(atom_lines)}")
    print(f"  HETATM records: {len(hetatm_lines)}")
else:
    # Try backup structure
    print(f"  Warning: Failed to download {best_pdb}, trying 6GL8")
    best_pdb = "6GL8"
    r = requests.get(f"https://files.rcsb.org/download/{best_pdb}.pdb", timeout=30)
    with open(pdb_path, "w") as f:
        f.write(r.text)
    print(f"  Downloaded {best_pdb}.pdb")

# Save the selected PDB ID for downstream scripts
with open(f"{DATA_DIR}/selected_pdb.txt", "w") as f:
    f.write(f"{best_pdb}\n{best_resolution:.2f}\n")

print(f"  PDB saved to: {pdb_path}")

# ─────────────────────────────────────────────────────────
# STEP 1C: Retrieve 5 natural compounds from PubChem
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1C: Retrieving natural compounds from PubChem")
print("=" * 60)

# 5 well-known natural BCL-2 modulating compounds
NATURAL_COMPOUNDS = [
    {
        "name": "Curcumin",
        "source": "Curcuma longa (turmeric)",
        "pubchem_cid": 969516,
        "rationale": "Polyphenol that downregulates BCL-2 expression and induces apoptosis in cancer cells"
    },
    {
        "name": "Quercetin",
        "source": "Various fruits and vegetables",
        "pubchem_cid": 5280343,
        "rationale": "Flavonoid shown to inhibit BCL-2 and induce apoptosis via intrinsic pathway"
    },
    {
        "name": "Resveratrol",
        "source": "Vitis vinifera (grapes), berries",
        "pubchem_cid": 445154,
        "rationale": "Stilbene that downregulates BCL-2 and promotes apoptosis in cancer cells"
    },
    {
        "name": "Apigenin",
        "source": "Apium graveolens (celery), chamomile",
        "pubchem_cid": 5280443,
        "rationale": "Flavone with anti-cancer activity that regulates BCL-2/BAX ratio"
    },
    {
        "name": "Berberine",
        "source": "Berberis vulgaris (barberry), Coptis chinensis",
        "pubchem_cid": 2353,
        "rationale": "Alkaloid that reduces BCL-2 expression and promotes apoptosis in cancer cells"
    }
]

print(f"\nFetching data for {len(NATURAL_COMPOUNDS)} natural compounds...")
compounds_data = []

for i, comp in enumerate(NATURAL_COMPOUNDS):
    print(f"\n  [{i+1}/{len(NATURAL_COMPOUNDS)}] {comp['name']} (CID: {comp['pubchem_cid']})")
    try:
        c = pcp.Compound.from_cid(comp["pubchem_cid"])
        smiles = c.canonical_smiles
        iupac = c.iupac_name
        mw = c.molecular_weight
        formula = c.molecular_formula
        xlogp = c.xlogp
        hbd = c.h_bond_donor_count
        hba = c.h_bond_acceptor_count
        tpsa = c.tpsa
        rot_bonds = c.rotatable_bond_count

        entry = {
            **comp,
            "smiles": smiles,
            "iupac_name": iupac,
            "molecular_formula": formula,
            "molecular_weight": mw,
            "xlogp": xlogp,
            "hbd": hbd,
            "hba": hba,
            "tpsa": tpsa,
            "rotatable_bonds": rot_bonds,
        }
        compounds_data.append(entry)

        print(f"    SMILES: {smiles[:60]}...")
        print(f"    MW={mw}, LogP={xlogp}, HBD={hbd}, HBA={hba}, TPSA={tpsa}")
        time.sleep(0.3)

    except Exception as e:
        print(f"    Error fetching CID {comp['pubchem_cid']}: {e}")
        # Use hardcoded SMILES as fallback
        fallback_smiles = {
            "Curcumin": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
            "Quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
            "Resveratrol": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
            "Apigenin": "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
            "Berberine": "COc1ccc2c(c1OC)C[n+]1cc3cc4c(cc3c[n+]1CC2)OCO4"
        }
        entry = {
            **comp,
            "smiles": fallback_smiles.get(comp["name"], ""),
            "iupac_name": comp["name"],
            "molecular_formula": "N/A",
            "molecular_weight": None,
        }
        compounds_data.append(entry)

# Save to JSON and CSV
json_path = f"{DATA_DIR}/compounds_info.json"
csv_path = f"{RESULTS_DIR}/compounds_info.csv"

with open(json_path, "w") as f:
    json.dump(compounds_data, f, indent=2)

df = pd.DataFrame(compounds_data)
df.to_csv(csv_path, index=False)

print(f"\n  Compound info saved to: {json_path}")
print(f"  Compound table saved to: {csv_path}")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA ACQUISITION COMPLETE")
print("=" * 60)
print(f"  PDB structure: {best_pdb} -> {pdb_path}")
print(f"  Compounds:     {len(compounds_data)} entries -> {json_path}")
print(f"  Results table: {csv_path}")

# Update manifest
manifest = {
    "current_step": "01_data_acquisition",
    "status": "completed",
    "selected_pdb": best_pdb,
    "pdb_resolution": best_resolution,
    "compounds_count": len(compounds_data),
    "outputs": {
        "pdb_raw": pdb_path,
        "compounds_json": json_path,
        "compounds_csv": csv_path
    }
}
with open(f"{SESSION}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nScript 01 complete.")
