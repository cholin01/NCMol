"""Identity-level split and initialization provenance safeguards."""
import hashlib

from rdkit import Chem


def molecule_id(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid molecule in split manifest")
    # Atom-map annotations must not allow identical molecules to evade checks.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_splits(payload, strategy):
    identities = {}
    for name in ("train", "val", "test"):
        rows = payload["splits"][strategy][name]
        if not rows:
            raise ValueError(f"Empty {name} partition")
        ids = [molecule_id(row["smiles"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate molecules within {name}; aggregate before splitting")
        for row, identity in zip(rows, ids):
            graph = payload["graphs"][row["key"]]
            if "smiles" not in graph or molecule_id(graph["smiles"]) != identity:
                raise ValueError("Graph/row molecule identity mismatch or missing graph SMILES")
        identities[name] = sorted(ids)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(identities[left]) & set(identities[right]):
            raise ValueError(f"Molecule overlap: {left}/{right}")
    return identities


def initialization_exposure(checkpoint, target_ids, policy="molecule-disjoint", target_task=None):
    manifest = checkpoint.get("protocol", {})
    if manifest.get("version") not in (1, 2) or "selection_exposure_ids" not in manifest:
        raise ValueError("Initialization lacks audited training/selection provenance. "
                         "Legacy weights remain usable for inference, but cannot certify leakage-free transfer.")
    exposed = set(manifest["selection_exposure_ids"])
    checked = exposed
    if policy == "cross-task":
        by_task = manifest.get("exposure_by_task", {})
        if not target_task or not by_task or "unspecified" in by_task:
            raise ValueError("Cross-task transfer requires explicit source and target task IDs")
        if set().union(*(set(ids) for ids in by_task.values())) != exposed:
            raise ValueError("Incomplete task exposure manifest")
        checked = set(by_task.get(target_task, []))
    elif policy != "molecule-disjoint":
        raise ValueError("Unknown transfer policy")
    # Validation also influences checkpoint selection; propagate it through transfers.
    for split in ("val", "test"):
        if checked & set(target_ids[split]):
            raise ValueError(f"Pretraining/selection molecules overlap target {split}")
    return exposed
