from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem


def one_hot(value, choices):
    return [float(value == choice) for choice in choices]


def graph_2d(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Match prepare_data.py exactly: RDKit preserves the original heavy-atom
    # order and appends explicit hydrogens, giving a deterministic 2D/3D atom map.
    mol = Chem.AddHs(mol)
    node_features = []
    atomic_numbers = []
    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())
        node_features.append(
            one_hot(min(atom.GetDegree(), 5), range(6))
            + [float(np.clip(atom.GetFormalCharge(), -5, 5)) / 5.0]
            + one_hot(
                str(atom.GetHybridization()),
                ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"],
            )
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
            + one_hot(min(atom.GetTotalNumHs(), 4), range(5))
            + one_hot(
                str(atom.GetChiralTag()),
                [
                    "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW",
                    "CHI_TETRAHEDRAL_CCW", "CHI_OTHER",
                ],
            )
        )
    edges = []
    edge_features = []
    bond_types = [
        Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
    ]
    for bond in mol.GetBonds():
        feature = one_hot(bond.GetBondType(), bond_types) + [
            float(bond.GetIsConjugated()), float(bond.IsInRing()),
        ]
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(begin, end), (end, begin)])
        edge_features.extend([feature, feature])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T.contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_features, dtype=torch.float32)
        if edge_features else torch.empty((0, 6), dtype=torch.float32)
    )
    return {
        "z_2d": torch.tensor(atomic_numbers, dtype=torch.long),
        "x_2d": torch.tensor(node_features, dtype=torch.float32),
        "edge_index_2d": edge_index,
        "edge_attr_2d": edge_attr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/npmmpd_3d.pt"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/npmmpd_2d3d_explicit_h.pt"),
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    for index, graph in enumerate(payload["graphs"].values(), start=1):
        topology = graph_2d(graph["smiles"])
        if not graph["conformers"]:
            raise ValueError(f"No conformers: {graph['smiles']}")
        for conformer in graph["conformers"]:
            if not torch.equal(conformer["z"], topology["z_2d"]):
                raise ValueError(f"2D/3D atomic numbers/order mismatch: {graph['smiles']}")
            if conformer["pos"].shape != (len(topology["z_2d"]), 3):
                raise ValueError("Expected one xyz coordinate per explicit-H atom")
            if not torch.isfinite(conformer["pos"]).all():
                raise ValueError("Non-finite conformer coordinates")
        graph.update(topology)
        if index % 500 == 0:
            print(f"processed {index}/{len(payload['graphs'])}", flush=True)
    payload["metadata"] = dict(payload.get("metadata", {})) | {
        "two_d_features": "explicit-H atom topology + bond type/conjugation/ring; generated from canonical SMILES",
        "two_d_node_feature_dim": 24,
        "two_d_edge_feature_dim": 6,
        "two_d_explicit_hydrogens": True,
        "atom_alignment": "exact RDKit Chem.AddHs atom order shared by 2D and 3D graphs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"FINAL {args.output} n={len(payload['graphs'])}", flush=True)


if __name__ == "__main__":
    main()
