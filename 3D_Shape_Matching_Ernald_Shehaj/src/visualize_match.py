#!/usr/bin/env python3
"""
Side-by-side visualization of two proteins with colors transferred along the
A↔B residue mapping from Hungarian algorithm results.

- Left: Protein A (colored by normalized XYZ → RGB)  
- Right: Protein B (colored by the color of its matched residue in A)
- Unmatched residues are light gray

Usage:
  python visualize_match.py data/3nir.pdb data/4nir.pdb mapping.csv --out figure.png
"""
from __future__ import annotations
import csv
import argparse
import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_ca_coords(pdb_path: Path) -> Tuple[np.ndarray, List[Tuple[str,int,str]]]:
    """
    Return:
      xyz  : (N,3) float array of CA coordinates
      meta : list of (chain, resseq, resname) for each row in xyz
    """
    xyz = []
    meta = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if len(line) < 54:  # malformed
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            resname = line[17:20].strip()
            chain   = line[21].strip() or "-"
            try:
                resseq  = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            xyz.append((x,y,z))
            meta.append((chain, resseq, resname))
    if not xyz:
        raise SystemExit(f"No CA atoms found in {pdb_path}")
    return np.array(xyz, dtype=float), meta

def load_mapping_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load the mapping CSV from your comparison script"""
    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows in mapping CSV: {csv_path}")
    return rows

def meta_key(chain: str, resseq: int) -> str:
    """Create unique key for residue identification"""
    return f"{chain}:{resseq}"

def xyz_to_rgb_unit(xyz: np.ndarray) -> np.ndarray:
    """
    Map xyz (N,3) to rgb (N,3) in [0,1] by min-max normalizing each column.
    This creates the "xyz → rgb" color mapping like in your example.
    """
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    rgb = (xyz - mins) / denom
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

def run_comparison_if_needed(pdbA: Path, pdbB: Path, mapping_csv: Path) -> Optional[dict]:
    """Run your comparison script if mapping doesn't exist"""
    if mapping_csv.exists():
        return None
        
    print(f"Running comparison between {pdbA.name} and {pdbB.name}...")
    
    try:
        result = subprocess.run([
            sys.executable, 'compare_pdbs_rmsd.py',
            str(pdbA), str(pdbB),
            '--out', str(mapping_csv),
            '--iters', '3',
            '--use-local-rmsd'
        ], capture_output=True, text=True, check=True)
        
        # Parse the JSON output for stats
        comparison_results = json.loads(result.stdout)
        print(f"Comparison complete! Similarity: {comparison_results['similarity_score']}%")
        return comparison_results
        
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not run comparison script: {e}")
        print("Please ensure compare_pdbs_rmsd.py is available and working")
        return None

def plot_side_by_side(
    xyzA: np.ndarray, colorsA: np.ndarray,
    xyzB: np.ndarray, colorsB: np.ndarray,
    out_path: Path, title: str = "", stats: Optional[dict] = None
):
    """Create the side-by-side visualization like your tutor's example"""
    
    # Shift B so they're side-by-side with a gap
    gap = max(15.0, (xyzA[:,0].max()-xyzA[:,0].min())*1.5)
    xyzB_shifted = xyzB.copy()
    xyzB_shifted[:,0] += xyzA[:,0].max() - xyzA[:,0].min() + gap

    fig = plt.figure(figsize=(14, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-45)

    # Plot both proteins
    ax.scatter(xyzA[:,0], xyzA[:,1], xyzA[:,2], s=30, c=colorsA, 
              depthshade=True, alpha=0.8, edgecolors='black', linewidths=0.3)
    ax.scatter(xyzB_shifted[:,0], xyzB_shifted[:,1], xyzB_shifted[:,2], s=30, c=colorsB,
              depthshade=True, alpha=0.8, edgecolors='black', linewidths=0.3)

    # Add "Matching" label between proteins like in your example
    mid_x = (xyzA[:,0].max() + xyzB_shifted[:,0].min()) / 2
    mid_y = (xyzA[:,1].mean() + xyzB_shifted[:,1].mean()) / 2
    mid_z = (xyzA[:,2].max() + xyzB_shifted[:,2].max()) / 2
    ax.text(mid_x, mid_y, mid_z, 'Matching', fontsize=16, ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    # Clean styling
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    
    # Remove axis lines for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w') 
    ax.zaxis.pane.set_edgecolor('w')
    
    if title:
        title_text = title
        if stats:
            title_text += f"\nSimilarity: {stats.get('similarity_score', 'N/A')}%"
            if stats.get('global_rmsd'):
                title_text += f" | RMSD: {stats['global_rmsd']} Å"
        ax.set_title(title_text, pad=20, fontsize=14)

    # Equal aspect ratio
    all_pts = np.vstack([xyzA, xyzB_shifted])
    mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
    ranges = maxs - mins; max_range = ranges.max()
    centers = (maxs + mins) / 2
    for i, ctr in enumerate(centers):
        lo = ctr - max_range/2; hi = ctr + max_range/2
        if i == 0: ax.set_xlim(lo, hi)
        if i == 1: ax.set_ylim(lo, hi)
        if i == 2: ax.set_zlim(lo, hi)

    # Add subtle grid
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor='white', dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description="Visualize Hungarian algorithm protein matching with color transfer"
    )
    ap.add_argument("pdbA", type=Path, nargs='?', default=None)
    ap.add_argument("pdbB", type=Path, nargs='?', default=None)
    ap.add_argument("mapping_csv", type=Path, nargs='?', default=Path("mapping.csv"))
    ap.add_argument("--out", type=Path, default=None, help="output figure (.png)")
    ap.add_argument("--gray_unmatched", type=float, default=0.85,
                    help="gray level for unmatched residues in B (0–1)")
    ap.add_argument("--run-comparison", action="store_true",
                    help="Force run comparison even if CSV exists")
    args = ap.parse_args()

    # Auto-detect PDB file locations if not specified
    if args.pdbA is None or args.pdbB is None:
        possible_paths = [
            ("data/3nir.pdb", "data/4nir.pdb"),
            ("../data/3nir.pdb", "../data/4nir.pdb"),
            ("3nir.pdb", "4nir.pdb"),
            ("./3nir.pdb", "./4nir.pdb")
        ]
        
        found = False
        for pdb1, pdb2 in possible_paths:
            if Path(pdb1).exists() and Path(pdb2).exists():
                args.pdbA = Path(pdb1)
                args.pdbB = Path(pdb2)
                print(f"Found PDB files: {pdb1}, {pdb2}")
                found = True
                break
        
        if not found:
            print("Could not find 3nir.pdb and 4nir.pdb files.")
            print("Tried looking in:")
            for pdb1, pdb2 in possible_paths:
                print(f"  {pdb1}, {pdb2}")
            print("\nPlease specify file paths manually:")
            print("  python visualize_match.py path/to/3nir.pdb path/to/4nir.pdb")
            return 1

    # Check if files exist
    if not args.pdbA.exists():
        print(f"PDB file not found: {args.pdbA}")
        return 1
    if not args.pdbB.exists():
        print(f"PDB file not found: {args.pdbB}")
        return 1

    # Run comparison if needed
    stats = None
    if args.run_comparison or not args.mapping_csv.exists():
        stats = run_comparison_if_needed(args.pdbA, args.pdbB, args.mapping_csv)
    
    if not args.mapping_csv.exists():
        print(f"Mapping CSV not found: {args.mapping_csv}")
        print("Try running with --run-comparison to generate it automatically")
        return 1

    print(f"Loading structures and mapping...")
    
    # Load structures
    xyzA, metaA = load_ca_coords(args.pdbA)
    xyzB, metaB = load_ca_coords(args.pdbB)
    rows = load_mapping_csv(args.mapping_csv)

    print(f"Protein A: {len(xyzA)} CA atoms")
    print(f"Protein B: {len(xyzB)} CA atoms") 
    print(f"Mappings: {len(rows)} pairs")

    # Create residue index maps
    idxA_by_key = {meta_key(c,r): i for i,(c,r,_) in enumerate(metaA)}
    idxB_by_key = {meta_key(c,r): i for i,(c,r,_) in enumerate(metaB)}

    # Color protein A by its coordinates (xyz → rgb)
    colorsA = xyz_to_rgb_unit(xyzA)

    # Default protein B colors = light gray
    gray = np.clip(args.gray_unmatched, 0.0, 1.0)
    colorsB = np.ones_like(xyzB) * gray

    # Transfer colors based on mapping
    matched_count = 0
    for row in rows:
        # Use the column names from your CSV output
        ak = meta_key(row["A_chain"], int(row["A_resseq"]))
        bk = meta_key(row["B_chain"], int(row["B_resseq"]))
        
        ia = idxA_by_key.get(ak, None)
        ib = idxB_by_key.get(bk, None)
        
        if ia is not None and ib is not None:
            # Transfer color from A to B
            colorsB[ib] = colorsA[ia]
            matched_count += 1

    print(f"Successfully matched {matched_count}/{len(rows)} residue pairs")

    # Create output path
    out = args.out or Path(f"fig_match_{args.pdbA.stem}_{args.pdbB.stem}.png")
    
    # Create the visualization
    title = f"{args.pdbA.stem}  →  {args.pdbB.stem}   (Hungarian Algorithm)"
    plot_side_by_side(xyzA, colorsA, xyzB, colorsB, out, title=title, stats=stats)
    
    print(f"Saved figure: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())