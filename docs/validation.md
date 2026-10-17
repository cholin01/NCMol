# Data format and verification

## Processed datasets

Datasets are trusted PyTorch files with:

- `graphs`: molecule key to a graph with `smiles`, `conformers`, `z_2d`,
  `x_2d`, `edge_index_2d` and `edge_attr_2d`.
- Each conformer: tensor `z`, `pos` and `edge_index`, plus optional energy metadata.
- `splits[strategy][partition]`: lists of `{key, smiles, y}` records, where
  partition is `train`, `val` or `test`.

Atom descriptors have 24 features, bond descriptors have six. Explicit-H atom
ordering must match between the 2D and 3D representations. Element checks alone
cannot detect swaps of identical elements; generate both from the same RDKit
molecule. Spatial graphs use a 5 Å cutoff and up to 32 neighbors including self.

Only load trusted `.pt` inputs, because legacy files use Python pickle.
Regenerating conformers with a different RDKit version can change predictions.
Mean aggregation is not an estimate of thermodynamic conformer populations.

## Local integration tests

The integration suite uses temporary toy data and the included six-layer
geometric encoder. It checks one-epoch NCMol, JEPA and transfer training,
checkpoint reload, standalone inference at two batch sizes, topology preparation,
raw-SMILES inference, conformer means, train-only scaling and invalid inputs.
Temporary data and weights are removed after the test.

```bash
python tests/smoke_inference.py
```

Tested CPU environment: Python 3.12.13, PyTorch 2.4.0, PyTorch Geometric 2.7.0,
NumPy 2.4.6, SciPy 1.15.3, scikit-learn 1.7.2, RDKit 2025.09.1.
These tests validate execution, not scientific accuracy. CUDA and fresh
dependency installation are not covered by this local check.

## Existing experimental checkpoints

The integrated geometry implementation was checked against the source used by
the retained remote NCMol experiments. Original geometry source hashes matched
LiTEN-FF revision `ee38b4ceda1b4f699d8e6b294f38255a70746cc1`.
Class names and module paths were reorganized; parameter keys and mathematical
operations were retained to allow strict loading of historical NCMol weights.
Unused legacy heads and normalization parameters remain for this compatibility.

On 2026-10-17, all 46 retained experimental checkpoints passed strict loading
and finite-parameter checks: six PAMPA models (three splits, with/without JEPA),
20 directly trained cellular models and 20 PAMPA-adapted cellular models.
No missing/unexpected parameter keys or shape mismatches were tolerated.
Eight evenly spaced PAMPA random-test molecules were checked for each variant
against saved outputs: maximum absolute difference was 4.76837158e-7 for NCMol
and 0 for NCMol+JEPA on the original modelling scale. These are sampled
prediction checks, not a full rerun of the 512-molecule test sets.

Batch loading can be repeated with:

```bash
python tests/check_collection.py runs/pampa runs/cellular runs/transfer
```

To compare a real checkpoint to saved prediction records:

```bash
python tests/check_checkpoint.py --checkpoint runs/pampa/best.pt \
  --data data/pampa.pt --reference runs/pampa/predictions_test.csv \
  --strategy random --limit 0
```

The test checks molecule ordering and absolute prediction differences below
1e-4. It reads datasets, weights and references without modifying them.
Use the saved experimental conformers for this comparison, not regenerated ones.

## Reproduction boundaries

The repository contains code rather than experiment data or weights. Direct
training initializes randomly unless an initialization checkpoint is supplied.
To reproduce experiments that used a separately generated random backbone,
supply that file with `--checkpoint`; omitting it changes the RNG sequence.
Use the original fixed partitions and conformers to compare published results.

Training defaults follow `configs/experiment_protocol.json`. The publication
protocol logs test metrics every epoch, but checkpoint selection uses only
validation RMSE. For new experiments, do not use test logs to tune settings.
Undefined correlations are reported as NaN for singleton or constant inputs.
