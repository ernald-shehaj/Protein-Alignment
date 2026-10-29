# Protein Structure Comparison Tools

Toolkit for comparing protein structures using feature-based Hungarian algorithm alignment with Kabsch superposition.

## Files

- `compare_pdbs_rmsd.py` - Main alignment engine (Hungarian + Kabsch)
- `visualize_match.py` - Side-by-side matched pair visualization
- `protein_pdb_methods.py` - PDB I/O and rendering utilities
- `render_pdb.py` - CLI tool for quick structure visualization

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run from the `src/` directory:

```bash
cd src

# Compare structures
python compare_pdbs_rmsd.py ../data/3nir.pdb ../data/4nir.pdb --out mapping.csv

# Visualize alignment
python visualize_match.py ../data/3nir.pdb ../data/4nir.pdb mapping.csv

# Render single structure (interactive HTML files)
python render_pdb.py ../data/3nir.pdb --all --output-dir ../figures/
```

Example results for 3NIR vs 4NIR are in `data/` (`out_mapping_3nir_4nir.csv`, `match.png`) and `src/fig_match_3nir_4nir.png`.

## Key Features

- Feature-based matching: kNN fingerprints, secondary structure, chemical groups
- Hungarian algorithm for optimal residue pairing
- Kabsch alignment with iterative refinement
- Multiple visualization modes (backbone, atoms, bonds, interfaces)

## References

- Cover & Hart (1967). Nearest neighbor pattern classification. *IEEE Trans. Info Theory*, 13(1), 21-27.
- Taylor (1986). The classification of amino acid conservation. *J. Theor. Biol.*, 119(2), 205-218.
- Hungarian Algorithm: https://www.geeksforgeeks.org/dsa/hungarian-algorithm-assignment-problem-set-1-introduction/
- Kabsch Algorithm: https://hunterheidenreich.com/posts/kabsch-algorithm/
- PDB methods: https://medium.com/@jgbrasier/working-with-pdb-files-in-python-7b538ee1b5e4
