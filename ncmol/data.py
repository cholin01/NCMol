"""Dataset and atom-aligned multi-conformer collation."""
import torch
from torch.utils.data import Dataset

class TwoDThreeDDataset(Dataset):
    def __init__(self, payload, strategy: str, split: str):
        self.graphs = payload["graphs"]
        self.rows = payload["splits"][strategy][split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.graphs[row["key"]], row


def collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    zs, positions, edges, atom_batches = [], [], [], []
    molecule_indices, ys, rows = [], [], []
    z2d, x2d, edges2d, attrs2d, batches2d = [], [], [], [], []
    offset3d = 0
    offset2d = 0
    graph_index = 0
    aligned_3d_indices, aligned_to_2d_indices = [], []
    for molecule_index, (graph, row) in enumerate(items):
        n2d = len(graph["z_2d"])
        if not n2d or not graph["conformers"]:
            raise ValueError("Each molecule requires atoms and at least one conformer")
        for conformer in graph["conformers"]:
            z = conformer["z"]
            if len(z) != n2d or not torch.equal(z, graph["z_2d"]):
                raise RuntimeError(
                    "Explicit-H 2D/3D atom ordering mismatch; interaction is unsafe"
                )
            zs.append(z)
            positions.append(conformer["pos"])
            edges.append(conformer["edge_index"] + offset3d)
            atom_batches.append(torch.full((len(z),), graph_index, dtype=torch.long))
            molecule_indices.append(molecule_index)
            aligned_3d_indices.append(torch.arange(offset3d, offset3d + n2d))
            aligned_to_2d_indices.append(torch.arange(offset2d, offset2d + n2d))
            offset3d += len(z)
            graph_index += 1
        z2d.append(graph["z_2d"])
        x2d.append(graph["x_2d"])
        edges2d.append(graph["edge_index_2d"] + offset2d)
        attrs2d.append(graph["edge_attr_2d"])
        batches2d.append(torch.full((n2d,), molecule_index, dtype=torch.long))
        offset2d += n2d
        ys.append(row["y"])
        rows.append(row)
    return {
        "z": torch.cat(zs), "pos": torch.cat(positions),
        "edge_index": torch.cat(edges, dim=1),
        "atom_batch": torch.cat(atom_batches),
        "molecule_index": torch.tensor(molecule_indices, dtype=torch.long),
        "z_2d": torch.cat(z2d), "x_2d": torch.cat(x2d),
        "edge_index_2d": torch.cat(edges2d, dim=1),
        "edge_attr_2d": torch.cat(attrs2d), "batch_2d": torch.cat(batches2d),
        "aligned_3d_index": torch.cat(aligned_3d_indices),
        "aligned_to_2d_index": torch.cat(aligned_to_2d_indices),
        "y": torch.tensor(ys, dtype=torch.float32), "rows": rows,
        "n_graphs": graph_index, "n_molecules": len(items),
    }
