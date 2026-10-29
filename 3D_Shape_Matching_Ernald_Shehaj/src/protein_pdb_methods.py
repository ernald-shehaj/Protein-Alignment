from __future__ import annotations

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx

from biopandas.pdb import PandasPdb
from prody import parsePDBHeader

from graphein.protein.graphs import (
    label_node_id,
    initialise_graph_with_metadata,
    add_nodes_to_graph,
)
from graphein.protein.visualisation import plotly_protein_structure_graph

import plotly.express as px
import plotly.graph_objects as go


# --------------------------- Core I/O with CONECT --------------------------- #

def parse_conect_records(pdb_path: str) -> List[Tuple[int, int]]:
    """Parse CONECT records from PDB file."""
    bonds = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('CONECT'):
                try:
                    atom1 = int(line[6:11].strip())
                    for i in range(11, len(line), 5):
                        atom_chunk = line[i:i+5].strip()
                        if atom_chunk and atom_chunk.isdigit():
                            atom2 = int(atom_chunk)
                            if atom1 < atom2:  # Avoid duplicates
                                bonds.append((atom1, atom2))
                except (ValueError, IndexError):
                    continue
    return bonds


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    model_index: int = 1,
    parse_header: bool = True,
    include_bonds: bool = False,
) -> Tuple[pd.DataFrame, Optional[dict], List[Tuple[int, int]]]:
    """Read a PDB file and return (df, header_dict, bonds_list)."""
    if pdb_path is None:
        raise ValueError("Please provide a local PDB file path.")
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    ppdb = PandasPdb().read_pdb(pdb_path)
    header = parsePDBHeader(pdb_path) if parse_header else None

    model = ppdb.get_model(model_index)
    if len(model.df["ATOM"]) == 0 and len(model.df["HETATM"]) == 0:
        raise ValueError(f"No atoms found for model index {model_index} in {pdb_path}")

    df = pd.concat([model.df["ATOM"], model.df["HETATM"]], ignore_index=True)
    bonds = parse_conect_records(pdb_path) if include_bonds else []
    
    return df, header, bonds


# --------------------- DataFrame processing -------------------- #

def process_dataframe(df: pd.DataFrame, granularity: str = "CA") -> pd.DataFrame:
    """Reduce ambiguity (alt_loc) and filter to desired granularity (default 'CA')."""
    if 'alt_loc' in df.columns:
        df = df.copy()
        df['alt_loc'] = df['alt_loc'].replace('', 'A')
        df = df.loc[df['alt_loc'] == 'A']

    df = label_node_id(df, granularity)

    if 'atom_name' not in df.columns:
        raise KeyError("Expected column 'atom_name' not found in DataFrame.")
    return df.loc[df['atom_name'] == granularity]


# -------------------- Graph construction/vis ------------------- #

def add_backbone_edges(G: nx.Graph) -> nx.Graph:
    """Connect sequential residues within each chain."""
    chain_ids = G.graph.get("chain_ids")
    if chain_ids is None:
        chain_ids = sorted({v.get('chain_id') for _, v in G.nodes(data=True) if 'chain_id' in v})
        G.graph["chain_ids"] = chain_ids

    for chain_id in chain_ids:
        residues = [(n, v) for n, v in G.nodes(data=True) if v.get("chain_id") == chain_id]
        residues.sort(key=lambda nv: nv[1].get("residue_number", 0))
        for (curr_name, curr_data), (next_name, next_data) in zip(residues, residues[1:]):
            if abs(curr_data.get("residue_number", 0) - next_data.get("residue_number", 0)) == 1:
                if G.has_edge(curr_name, next_name):
                    kind = G.edges[curr_name, next_name].get("kind", set())
                    kind.add("backbone_bond")
                    G.edges[curr_name, next_name]["kind"] = kind
                else:
                    G.add_edge(curr_name, next_name, kind={"backbone_bond"})
    return G


def build_backbone_graph(df: pd.DataFrame, raw_df: pd.DataFrame, pdb_code: str = "unknown", granularity: str = "CA") -> nx.Graph:
    """Create Graphein graph with nodes and backbone edges."""
    g = initialise_graph_with_metadata(
        protein_df=df,
        raw_pdb_df=raw_df,
        pdb_code=pdb_code,
        granularity=granularity,
    )
    g.graph['chain_ids'] = sorted(df['chain_id'].unique().tolist()) if 'chain_id' in df.columns else []
    g = add_nodes_to_graph(g)
    g = add_backbone_edges(g)
    return g


def visualize_backbone_graph(g: nx.Graph, title: str = "Protein Backbone Graph"):
    """Plotly visualization for the residue graph."""
    return plotly_protein_structure_graph(
        g,
        colour_edges_by="kind",
        colour_nodes_by="seq_position",
        label_node_ids=False,
        plot_title=title,
        node_size_multiplier=1,
    )


# ---------------------- Atom-level plotting -------------------- #

def plot_atoms_scatter3d(df: pd.DataFrame, color_by: str = "element_symbol"):
    """Plot atoms as 3D scatter plot with proper color handling."""
    if color_by not in df.columns:
        raise KeyError(f"Column '{color_by}' not found in DataFrame.")
    
    fig = px.scatter_3d(df, x='x_coord', y='y_coord', z='z_coord', color=color_by)
    fig.update_traces(marker_size=4)
    return fig


def plot_atoms_with_bonds(df: pd.DataFrame, bonds: List[Tuple[int, int]], color_by: str = "element_symbol"):
    """Plot atoms with bond connections."""
    import plotly.express as px
    
    fig = go.Figure()
    
    # Create atom number to index mapping
    atom_to_idx = {row['atom_number']: idx for idx, row in df.iterrows()}
    
    # Handle coloring
    if color_by in df.columns:
        if color_by == "element_symbol":
            # Use plotly express to get proper color mapping for elements
            temp_fig = px.scatter_3d(df, x='x_coord', y='y_coord', z='z_coord', 
                                   color=color_by, color_discrete_sequence=px.colors.qualitative.Set1)
            color_data = temp_fig.data[0].marker.color
        else:
            # For other categorical columns, use plotly express
            temp_fig = px.scatter_3d(df, x='x_coord', y='y_coord', z='z_coord', color=color_by)
            color_data = temp_fig.data[0].marker.color
    else:
        color_data = 'blue'  # Default color
    
    # Plot atoms
    fig.add_trace(go.Scatter3d(
        x=df['x_coord'],
        y=df['y_coord'],
        z=df['z_coord'],
        mode='markers',
        marker=dict(size=5, color=color_data),
        name='Atoms',
        text=df.get('atom_number', range(len(df)))
    ))
    
    # Plot bonds
    if bonds:
        bond_x, bond_y, bond_z = [], [], []
        for atom1, atom2 in bonds:
            if atom1 in atom_to_idx and atom2 in atom_to_idx:
                idx1, idx2 = atom_to_idx[atom1], atom_to_idx[atom2]
                bond_x.extend([df.iloc[idx1]['x_coord'], df.iloc[idx2]['x_coord'], None])
                bond_y.extend([df.iloc[idx1]['y_coord'], df.iloc[idx2]['y_coord'], None])
                bond_z.extend([df.iloc[idx1]['z_coord'], df.iloc[idx2]['z_coord'], None])
        
        fig.add_trace(go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z,
            mode='lines',
            line=dict(color='gray', width=2),
            name='Bonds'
        ))
    
    fig.update_layout(
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


def plot_two_chains_with_interface(df_A: pd.DataFrame, df_B: pd.DataFrame, contact_atoms: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(name='Chain A', x=df_A['x_coord'], y=df_A['y_coord'], z=df_A['z_coord'],
                               mode='markers', marker=dict(size=4, color='blue', opacity=0.5)))
    fig.add_trace(go.Scatter3d(name='Chain B', x=df_B['x_coord'], y=df_B['y_coord'], z=df_B['z_coord'],
                               mode='markers', marker=dict(size=4, color='red', opacity=0.5)))
    if len(contact_atoms) > 0:
        fig.add_trace(go.Scatter3d(name='Contact Atoms', x=contact_atoms['x_coord'], y=contact_atoms['y_coord'], z=contact_atoms['z_coord'],
                                   mode='markers', marker=dict(size=6, color='orange')))
    fig.update_layout(scene=dict(aspectmode="cube"), margin=dict(l=0, r=0, b=0, t=30))
    return fig





# -------------------- Chain interfaces ------------------- #

def get_contact_atoms(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold: float = 7.0,
    coord_names: List[str] = ["x_coord", "y_coord", "z_coord"],
):
    """Return (contact_df, node_pairs) for atoms within threshold Å across chains."""
    coords1 = df1[coord_names].to_numpy(float)
    coords2 = df2[coord_names].to_numpy(float)
    dist = np.sqrt(((coords1[:, None, :] - coords2[None, :, :]) ** 2).sum(axis=2))
    pairs = np.argwhere(dist < threshold)
    if pairs.size == 0:
        import numpy as _np
        return pd.concat([df1.iloc[[]], df2.iloc[[]]]), _np.empty((0, 2), dtype=object)

    atoms1 = df1.iloc[pairs[:, 0]]
    atoms2 = df2.iloc[pairs[:, 1]]
    atoms1_id = atoms1['chain_id'].astype(str) + ":" + atoms1['residue_name'].astype(str) + ":" + atoms1['residue_number'].astype(str)
    atoms2_id = atoms2['chain_id'].astype(str) + ":" + atoms2['residue_name'].astype(str) + ":" + atoms2['residue_number'].astype(str)

    import numpy as _np
    node_pairs = _np.vstack((atoms1_id.values, atoms2_id.values)).T

    contact_df = pd.concat([df1.iloc[np.unique(pairs[:, 0])], df2.iloc[np.unique(pairs[:, 1])]], ignore_index=True)
    return contact_df, node_pairs