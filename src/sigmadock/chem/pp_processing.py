"""
Protein-protein graph builder.

Takes two DataFrames (receptor, binder) from a DIPS .dill file and produces
a torch_geometric Data object compatible with the SigmaDock denoiser.

Convention:
  - Receptor: only Cα atoms (one node per residue). Fixed, frag_idx_map = -1.
  - Binder:   backbone atoms only (N, CA, C, O). Diffused, frag_idx_map = residue_index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from sigmadock.chem import RESIDUE_MAP
from sigmadock.chem.utils import get_fourier_embeddings
from sigmadock.oracle import HPARAMS

ELEMENT_TO_ATOMIC_NUM: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15,
    "F": 9, "CL": 17, "BR": 35, "I": 53, "SE": 34,
}

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}

# Bond topology within a backbone frame: N-CA, CA-C, C-O
BACKBONE_BONDS = [("N", "CA"), ("CA", "C"), ("C", "O")]


def _element_to_features(element: str, atom_name: str) -> np.ndarray:
    """
    10-dim atom feature vector from element + atom_name (no RDKit needed).
    Mirrors get_atom_features() layout: [atomic_num, degree, charge, hybridization,
    implicit_valence, explicit_valence, is_aromatic, is_in_ring, num_rings, chiral]
    """
    el = element.strip().upper()
    atomic_num = ELEMENT_TO_ATOMIC_NUM.get(el, 6)
    hyb_map = {"CA": 3, "C": 2, "N": 3, "O": 2}
    hyb = hyb_map.get(atom_name.strip(), 3)
    return np.array([atomic_num, 0, 0, hyb, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def build_receptor_graph(rec_df: pd.DataFrame) -> dict:
    """
    Receptor: keep only Cα atoms. One node per residue.
    All nodes are protein_virtual (5). frag_idx_map = -1 (not diffused).
    CA-CA edges within 18 Å cutoff.
    """
    ca_df = rec_df[rec_df["atom_name"].str.strip() == "CA"].reset_index(drop=True)
    N = len(ca_df)

    pos = torch.tensor(ca_df[["x", "y", "z"]].values, dtype=torch.float32)

    x = torch.tensor(
        np.stack([_element_to_features(row.element, "CA") for row in ca_df.itertuples()]),
        dtype=torch.float32,
    )

    res_types = torch.tensor(
        [RESIDUE_MAP.get(str(r).strip(), RESIDUE_MAP["UNK"]) for r in ca_df["resname"]],
        dtype=torch.long,
    )

    # All receptor nodes are protein_virtual
    node_entity = torch.full((N,), HPARAMS.get_node_idx("protein_virtual"), dtype=torch.long)

    # Positional embedding: depth = 0 for all CA
    depths = torch.zeros(N, dtype=torch.long)
    positional_embeddings = get_fourier_embeddings(depths, sigma=1 / 1024, num_features=16)

    # CA-CA edges within 18 Å
    tree = cKDTree(pos.numpy())
    pairs = tree.query_pairs(r=18.0)
    virt_edges = []
    for i, j in pairs:
        virt_edges += [[i, j], [j, i]]

    if virt_edges:
        edge_index = torch.tensor(virt_edges, dtype=torch.long).t()
        edge_attr = torch.zeros((edge_index.size(1), 4), dtype=torch.float32)
        edge_entity = torch.full(
            (edge_index.size(1),), HPARAMS.get_edge_idx("protein_v2v"), dtype=torch.long
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
        edge_entity = torch.empty((0,), dtype=torch.long)

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_entity": node_entity,
        "edge_entity": edge_entity,
        "esm_embeddings": None,
        "positional_embeddings": positional_embeddings,
        "mask": torch.ones(N, dtype=torch.bool),
        "residue_types": res_types,
        "n_atoms": N,
    }


def build_binder_graph(bind_df: pd.DataFrame) -> dict:
    """
    Binder: keep only backbone atoms (N, CA, C, O). One rigid frame per residue.
    node_entity:
      CA atoms  → protein_virtual (5)
      N, C, O   → protein_atom (4)
    frag_idx_map = sequential residue index (0, 1, 2, ...)
    """
    bb_df = bind_df[bind_df["atom_name"].str.strip().isin(BACKBONE_ATOM_NAMES)].reset_index(drop=True)
    N = len(bb_df)

    pos = torch.tensor(bb_df[["x", "y", "z"]].values, dtype=torch.float32)

    x = torch.tensor(
        np.stack([
            _element_to_features(row.element, row.atom_name)
            for row in bb_df.itertuples()
        ]),
        dtype=torch.float32,
    )

    res_types = torch.tensor(
        [RESIDUE_MAP.get(str(r).strip(), RESIDUE_MAP["UNK"]) for r in bb_df["resname"]],
        dtype=torch.long,
    )

    # Fragment index: one per residue
    residue_keys = list(dict.fromkeys(zip(bb_df["chain"], bb_df["residue"])))
    key_to_frag = {k: i for i, k in enumerate(residue_keys)}
    frag_idx_map = torch.tensor(
        [key_to_frag[(c, r)] for c, r in zip(bb_df["chain"], bb_df["residue"])],
        dtype=torch.long,
    )
    frag_counter = frag_idx_map.clone()

    frag_atom_idx = torch.zeros(N, dtype=torch.long)
    counts: dict[int, int] = {}
    for i, f in enumerate(frag_idx_map.tolist()):
        frag_atom_idx[i] = counts.get(f, 0)
        counts[f] = counts.get(f, 0) + 1

    # Node entity: CA → protein_virtual, others → protein_atom
    node_entity = torch.full((N,), HPARAMS.get_node_idx("protein_atom"), dtype=torch.long)
    ca_local = bb_df["atom_name"].str.strip() == "CA"
    node_entity[ca_local.values] = HPARAMS.get_node_idx("protein_virtual")
    ca_indices = np.where(ca_local.values)[0]

    # Positional embeddings: CA=0, N/C/O=1
    depth_map = {"CA": 0, "N": 1, "C": 1, "O": 1}
    depths = torch.tensor(
        [depth_map.get(str(n).strip(), 1) for n in bb_df["atom_name"]],
        dtype=torch.long,
    )
    positional_embeddings = get_fourier_embeddings(depths, sigma=1 / 1024, num_features=16)

    # Bond edges per residue: N-CA, CA-C, C-O
    edge_pairs: list[list[int]] = []
    edge_feats: list[np.ndarray] = []
    bf = np.array([1, 0, 0, 0], dtype=np.float32)
    for _, grp in bb_df.groupby(["chain", "residue"], sort=False):
        name_to_local = {
            n.strip(): i for i, n in enumerate(grp["atom_name"])
        }
        offset = int(grp.index[0])  # already reset_index so this is the bb_df position
        for a, b in BACKBONE_BONDS:
            if a in name_to_local and b in name_to_local:
                i = name_to_local[a] + offset
                j = name_to_local[b] + offset
                edge_pairs += [[i, j], [j, i]]
                edge_feats += [bf, bf]

    if edge_pairs:
        edge_index_chem = torch.tensor(edge_pairs, dtype=torch.long).t()
        edge_attr_chem = torch.tensor(np.stack(edge_feats), dtype=torch.float32)
    else:
        edge_index_chem = torch.empty((2, 0), dtype=torch.long)
        edge_attr_chem = torch.empty((0, 4), dtype=torch.float32)

    edge_entity_chem = torch.full(
        (edge_index_chem.size(1),), HPARAMS.get_edge_idx("protein_bonds"), dtype=torch.long
    )

    # CA-CA virtual edges within 18 Å
    ca_pos = pos[ca_indices].numpy()
    tree = cKDTree(ca_pos)
    pairs = tree.query_pairs(r=18.0)
    virt_edges = []
    for i, j in pairs:
        gi, gj = int(ca_indices[i]), int(ca_indices[j])
        virt_edges += [[gi, gj], [gj, gi]]

    if virt_edges:
        edge_index_v = torch.tensor(virt_edges, dtype=torch.long).t()
        edge_attr_v = torch.zeros((edge_index_v.size(1), 4), dtype=torch.float32)
        edge_entity_v = torch.full(
            (edge_index_v.size(1),), HPARAMS.get_edge_idx("protein_v2v"), dtype=torch.long
        )
    else:
        edge_index_v = torch.empty((2, 0), dtype=torch.long)
        edge_attr_v = torch.empty((0, 4), dtype=torch.float32)
        edge_entity_v = torch.empty((0,), dtype=torch.long)

    edge_index = torch.cat([edge_index_chem, edge_index_v], dim=1)
    edge_attr = torch.cat([edge_attr_chem, edge_attr_v], dim=0)
    edge_entity = torch.cat([edge_entity_chem, edge_entity_v], dim=0)

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_entity": node_entity,
        "edge_entity": edge_entity,
        "esm_embeddings": None,
        "positional_embeddings": positional_embeddings,
        "mask": torch.ones(N, dtype=torch.bool),
        "residue_types": res_types,
        "frag_counter": frag_counter,
        "frag_atom_idx": frag_atom_idx,
        "frag_idx_map": frag_idx_map,
        "n_atoms": N,
    }


def build_pp_complex_graph(
    rec_df: pd.DataFrame,
    bind_df: pd.DataFrame,
    inter_cutoff: float = 4.0,
    coordinate_noise: float | None = None,
) -> Data:
    """
    Merge receptor (CA only) + binder (backbone only) into a single complex graph.
    Returns a torch_geometric Data compatible with SigmaDockDenoiser.
    """
    rec = build_receptor_graph(rec_df)
    bnd = build_binder_graph(bind_df)

    Np = rec["x"].size(0)

    if coordinate_noise is not None:
        rec["pos"] = rec["pos"] + torch.randn_like(rec["pos"]) * coordinate_noise
        bnd["pos"] = bnd["pos"] + torch.randn_like(bnd["pos"]) * coordinate_noise

    x = torch.cat([rec["x"], bnd["x"]], dim=0)
    pos = torch.cat([rec["pos"], bnd["pos"]], dim=0)
    node_entity = torch.cat([rec["node_entity"], bnd["node_entity"]], dim=0)
    mask = torch.cat([rec["mask"], bnd["mask"]], dim=0)
    res_types = torch.cat([rec["residue_types"], bnd["residue_types"]], dim=0)

    frag_counter = torch.cat([
        torch.full((Np,), -1, dtype=torch.long),
        bnd["frag_counter"],
    ], dim=0)
    frag_atom_idx = torch.cat([
        torch.full((Np,), -1, dtype=torch.long),
        bnd["frag_atom_idx"],
    ], dim=0)
    frag_idx_map = torch.cat([
        torch.full((Np,), -1, dtype=torch.long),
        bnd["frag_idx_map"],
    ], dim=0)

    bnd_ei_shifted = bnd["edge_index"] + Np
    edge_index = torch.cat([rec["edge_index"], bnd_ei_shifted], dim=1)
    edge_attr = torch.cat([rec["edge_attr"], bnd["edge_attr"]], dim=0)
    edge_entity = torch.cat([rec["edge_entity"], bnd["edge_entity"]], dim=0)

    # Inter-complex edges: receptor CA ↔ binder backbone within cutoff
    rec_pos_np = rec["pos"].numpy()
    bnd_pos_np = bnd["pos"].numpy()
    tree_bnd = cKDTree(bnd_pos_np)
    pairs = tree_bnd.query_ball_point(rec_pos_np, r=inter_cutoff)

    src_list, dst_list = [], []
    for i, js in enumerate(pairs):
        for j in js:
            src_list.append(i)
            dst_list.append(j + Np)

    if src_list:
        src = torch.tensor(src_list + dst_list, dtype=torch.long)
        dst = torch.tensor(dst_list + src_list, dtype=torch.long)
        inter_ei = torch.stack([src, dst], dim=0)
        inter_ea = torch.zeros((inter_ei.size(1), 4), dtype=torch.float32)
        inter_ee = torch.full(
            (inter_ei.size(1),), HPARAMS.get_edge_idx("inter_complex"), dtype=torch.long
        )
        edge_index = torch.cat([edge_index, inter_ei], dim=1)
        edge_attr = torch.cat([edge_attr, inter_ea], dim=0)
        edge_entity = torch.cat([edge_entity, inter_ee], dim=0)

    pocket_com = bnd["pos"].mean(dim=0, keepdim=True)

    pos_emb = torch.cat([
        rec["positional_embeddings"],
        bnd["positional_embeddings"],
    ], dim=0)

    data = Data(
        x=x,
        ref_pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_entity=node_entity,
        edge_entity=edge_entity,
        mask=mask,
        frag_counter=frag_counter,
        frag_atom_idx=frag_atom_idx,
        frag_idx_map=frag_idx_map,
        residue_types=res_types,
        pocket_com=pocket_com,
        protein_embeddings={
            "esm_embeddings": None,
            "positional_embeddings": pos_emb,
        },
        triangulation_indexes=None,
        overconstrained_anchors=torch.zeros(0, dtype=torch.bool),
        overconstrained_dummies=torch.zeros(0, dtype=torch.bool),
        overconstrained_mask=torch.ones(x.size(0), dtype=torch.bool),
        frag_sizes=torch.zeros(0, dtype=torch.long),
        dummy_frag_sizes=torch.zeros(0, dtype=torch.long),
    )
    return data