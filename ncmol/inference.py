"""Checkpoint loading and explicit-H conformer generation for NCMol inference."""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from prepare_data import graph_2d
from .model import NCMol


def load_model(checkpoint, device="cpu"):
    # Legacy training checkpoints contain argparse Path objects: load only trusted files.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"model", "target_mean", "target_std"} <= payload.keys():
        raise ValueError("Expected a trained NCMol best.pt with model and training-label statistics")
    mean, std = float(payload["target_mean"]), float(payload["target_std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid checkpoint target_mean/target_std")
    state = payload["model"]
    jepa = any(k.startswith("jepa_predictor.") for k in state)
    options = payload.get("args", {})
    if not isinstance(options, dict):
        options = vars(options)
    hidden = state["jepa_predictor.1.weight"].shape[0] if jepa else 256
    model = NCMol(checkpoint=None, dropout=float(options.get("dropout", 0.2)),
                  jepa=jepa, jepa_hidden_dim=hidden)
    # Loading the complete trained state avoids requiring a separate initialization file.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, mean, std


def graph_from_smiles(smiles, candidates=10, keep=3, cutoff=5.0, max_neighbors=32):
    if not 1 <= keep <= candidates or cutoff <= 0 or max_neighbors < 1:
        raise ValueError("Require 1 <= keep <= candidates, positive cutoff and max_neighbors")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    key = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = zlib.crc32(key.encode()) & 0x7FFFFFFF
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    params.useMacrocycleTorsions = True
    params.useMacrocycle14config = True
    params.useSmallRingTorsions = True
    params.maxIterations = 2000
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=candidates, params=params))
    if not ids:
        params.useRandomCoords = True
        params.randomSeed = (params.randomSeed + 104729) & 0x7FFFFFFF
        params.pruneRmsThresh = -1.0
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(keep, 3), params=params))
    if not ids:
        raise ValueError(f"Conformer generation failed: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        optimized = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500,
                                                    mmffVariant="MMFF94s")
    else:
        optimized = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)
    valid = [(float(energy), cid, status) for cid, (status, energy) in zip(ids, optimized)
             if status >= 0 and math.isfinite(energy)]
    if not valid:
        raise ValueError(f"No usable optimized conformers: {smiles}")
    graph = graph_2d(smiles)
    graph.update(smiles=smiles, conformers=[])
    for energy, cid, status in sorted(valid)[:keep]:
        pos = np.asarray(mol.GetConformer(cid).GetPositions(), dtype=np.float32)
        distances = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        pairs = []
        for receiver in range(len(pos)):
            neighbors = np.flatnonzero(distances[receiver] < cutoff)
            neighbors = neighbors[np.argsort(distances[receiver, neighbors], kind="stable")]
            pairs.extend((receiver, int(sender)) for sender in neighbors[:max_neighbors])
        graph["conformers"].append({
            "z": graph["z_2d"].clone(), "pos": torch.from_numpy(pos),
            "edge_index": torch.tensor(pairs, dtype=torch.long).T.contiguous(),
            "energy": energy, "optimization_status": int(status),
        })
    return graph
