# Protein Alignment – Projektarbeit

Ernald Shehaj, University of Bonn

Feature-based protein structure comparison: residues are matched with the Hungarian algorithm on a cost built from kNN fingerprints, secondary structure and residue chemistry, then superposed with iterative Kabsch alignment.

## Contents

| Folder / file | What it is |
| --- | --- |
| [`3D_Shape_Matching_Ernald_Shehaj/`](3D_Shape_Matching_Ernald_Shehaj/) | Python code, example PDB data (3NIR, 4NIR) and results. See its README for setup and usage. |
| [`Projektarbeit_Ernald_Shehaj-2/`](Projektarbeit_Ernald_Shehaj-2/) | Written report: LaTeX source (`main.tex`, `references.bib`, figures) and compiled PDF. |
| [`Presentation_Final_Ernald_Shehaj.pdf`](Presentation_Final_Ernald_Shehaj.pdf) | Final presentation slides. |

## Quick start

```bash
cd 3D_Shape_Matching_Ernald_Shehaj
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python compare_pdbs_rmsd.py ../data/3nir.pdb ../data/4nir.pdb --out mapping.csv
```

## Building the report

```bash
cd Projektarbeit_Ernald_Shehaj-2
pdflatex main && biber main && pdflatex main && pdflatex main
```
