#!/usr/bin/env python3
"""
Command-line tool for rendering PDB structures in multiple ways.

Usage:
    python render_pdb.py <pdb_file> [options]
    
Examples:
    python render_pdb.py 1zni.pdb --all
    python render_pdb.py 1zni.pdb --backbone --atoms
    python render_pdb.py 1zni.pdb --interface --output-dir figures/
"""

import argparse
import os
from pathlib import Path
from protein_pdb_methods import (
    read_pdb_to_dataframe,
    process_dataframe,
    build_backbone_graph,
    visualize_backbone_graph,
    plot_atoms_scatter3d,
    plot_atoms_with_bonds,
    plot_two_chains_with_interface,
    get_contact_atoms
)

def main():
    parser = argparse.ArgumentParser(
        description="Render PDB structures with multiple visualization modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization modes:
  --backbone        Residue-level graph with backbone connectivity
  --atoms           3D scatter plot of all atoms
  --bonds           Atoms with bond network from CONECT records
  --interface       Chain-chain interface analysis (multi-chain only)
  --all             Generate all applicable visualizations

Examples:
  python render_pdb.py 1zni.pdb --all
  python render_pdb.py 1zni.pdb --backbone --atoms
  python render_pdb.py 3i40.pdb --interface --output-dir figures/
        """
    )
    
    parser.add_argument("pdb_file", help="Path to PDB file")
    parser.add_argument("--backbone", action="store_true", help="Backbone graph visualization")
    parser.add_argument("--atoms", action="store_true", help="Atom scatter plot")
    parser.add_argument("--bonds", action="store_true", help="Atoms with bonds")
    parser.add_argument("--interface", action="store_true", help="Chain interface analysis")
    parser.add_argument("--all", action="store_true", help="All visualizations")
    parser.add_argument("--color-by", default="element_symbol", 
                       help="Coloring scheme for atoms: element_symbol, chain_id, residue_name, b_factor")
    parser.add_argument("--threshold", type=float, default=7.0,
                       help="Distance threshold for interface contacts (Angstroms)")
    parser.add_argument("--output-dir", help="Save HTML files instead of showing (optional)")
    parser.add_argument("--granularity", default="CA", help="Atom selection for backbone graph (default: CA)")
    
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.pdb_file):
        print(f"Error: PDB file not found: {args.pdb_file}")
        return 1
    
    # If --all, enable everything
    if args.all:
        args.backbone = True
        args.atoms = True
        args.bonds = True
        args.interface = True
    
    # If nothing selected, show help
    if not any([args.backbone, args.atoms, args.bonds, args.interface]):
        parser.print_help()
        return 0
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Will save outputs to: {args.output_dir}/")
    
    # Load PDB
    print(f"Loading {args.pdb_file}...")
    try:
        df, header, bonds = read_pdb_to_dataframe(
            args.pdb_file, 
            parse_header=True, 
            include_bonds=args.bonds
        )
    except Exception as e:
        print(f"Error loading PDB: {e}")
        return 1
    
    pdb_name = Path(args.pdb_file).stem
    print(f"Loaded {len(df)} atoms")
    print(f"Chains: {sorted(df['chain_id'].unique())}")
    if bonds:
        print(f"Bonds: {len(bonds)}")
    
    # 1. Backbone graph
    if args.backbone:
        print("\n[1/4] Generating backbone graph...")
        try:
            df_ca = process_dataframe(df, granularity=args.granularity)
            graph = build_backbone_graph(df_ca, df, pdb_code=pdb_name.upper(), 
                                        granularity=args.granularity)
            fig = visualize_backbone_graph(graph, title=f"{pdb_name} Backbone Structure")
            
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{pdb_name}_backbone.html")
                fig.write_html(output_path)
                print(f"   Saved: {output_path}")
            else:
                fig.show()
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. Atom scatter
    if args.atoms:
        print("\n[2/4] Generating atom scatter plot...")
        try:
            fig = plot_atoms_scatter3d(df, color_by=args.color_by)
            fig.update_layout(title=f"{pdb_name} - All Atoms")
            
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{pdb_name}_atoms.html")
                fig.write_html(output_path)
                print(f"   Saved: {output_path}")
            else:
                fig.show()
        except Exception as e:
            print(f"   Error: {e}")
    
    # 3. Bonds
    if args.bonds:
        print("\n[3/4] Generating atoms with bonds...")
        if not bonds:
            print("   Warning: No CONECT records found, loading bonds...")
            _, _, bonds = read_pdb_to_dataframe(args.pdb_file, include_bonds=True)
        
        try:
            fig = plot_atoms_with_bonds(df, bonds, color_by=args.color_by)
            fig.update_layout(title=f"{pdb_name} - Atoms and Bonds")
            
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{pdb_name}_bonds.html")
                fig.write_html(output_path)
                print(f"   Saved: {output_path}")
            else:
                fig.show()
        except Exception as e:
            print(f"   Error: {e}")
    
    # 4. Interface
    if args.interface:
        print("\n[4/4] Analyzing chain interface...")
        chains = sorted(df['chain_id'].unique())
        
        if len(chains) < 2:
            print(f"   Skipped: Only {len(chains)} chain found, need at least 2")
        else:
            try:
                # Use first two chains
                df_a = df[df['chain_id'] == chains[0]]
                df_b = df[df['chain_id'] == chains[1]]
                
                print(f"   Analyzing {chains[0]} - {chains[1]} interface...")
                contact_df, pairs = get_contact_atoms(df_a, df_b, threshold=args.threshold)
                
                print(f"   Found {len(pairs)} residue-residue contacts")
                print(f"   Contact atoms: {len(contact_df)}")
                
                fig = plot_two_chains_with_interface(df_a, df_b, contact_df)
                fig.update_layout(title=f"{pdb_name} - Chain {chains[0]}/{chains[1]} Interface")
                
                if args.output_dir:
                    output_path = os.path.join(args.output_dir, 
                                              f"{pdb_name}_interface_{chains[0]}{chains[1]}.html")
                    fig.write_html(output_path)
                    print(f"   Saved: {output_path}")
                else:
                    fig.show()
            except Exception as e:
                print(f"   Error: {e}")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    exit(main())