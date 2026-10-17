# NCMol

**Topology-conditioned multi-conformer learning for molecular permeability prediction across artificial and cellular membranes**

NCMol predicts permeability of nonpeptidic macrocycles by combining an explicit-hydrogen
molecular graph with multiple 3D conformers. The package supports PAMPA, Caco-2 and MDCK
regression tasks, optional cross-conformer JEPA, and PAMPA-to-cellular adaptation.

## Model

The model combines:

- **Recurrent topology learning:** gated directed bond messages, degree-normalized
  aggregation, GRU atom updates and evolving bond states.
- **Topology-conditioned geometry:** atom-aligned gated residual updates after
  geometric layers 3 and 5.
- **Multi-conformer prediction:** gated 2D–3D fusion, a shared regression head and
  averaging of conformer predictions.
- **Optional JEPA:** an EMA target predicts representations across conformers during
  training; inference uses only the online prediction path.

The full encoder implementation is included in `ncmol/`; no separate source checkout
or `vendor` directory is needed. The geometric interaction equations are retained
from the trained model, while the recurrent topology branch and cross-modal updates
implement the NCMol-specific changes. Geometric module provenance is documented below.
Parameter keys remain compatible with existing NCMol checkpoints.

## Install

Python 3.10 or later:

```bash
git clone https://github.com/cholin01/NCMol.git
cd NCMol
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU execution, install a CUDA-compatible PyTorch build. CPU execution is supported.
No SPICE initialization is required. Trained weights and experimental datasets are
not included in this repository.

## Prepare data

Provide a CSV with `smiles`, `y` and `split` columns. Use labels on the original
modelling scale, such as log permeability, and assign `train`, `val` and `test`
before preparation. Aggregate repeated measurements before splitting.

```text
smiles,y,split
CCO,-6.0,train
CCN,-5.0,train
CCCC,-6.4,val
CCCO,-5.2,test
```

The rows above illustrate the format only.

```bash
python prepare_csv.py --input labelled.csv --output data/pampa.pt \
  --strategy random --candidate-conformers 10 --keep-conformers 3
```

Preparation generates explicit-H ETKDGv3 conformers with macrocycle settings,
optimizes them with MMFF94s (UFF fallback), and retains up to three lowest-energy
conformers. RMS pruning can produce fewer than three. The 2D and 3D representations
share atom ordering. Invalid molecules and duplicate canonical isomeric SMILES
are rejected. The strategy flag labels the provided partitions; it does not create
random, activity-cliff or scaffold splits.

Existing processed 3D datasets can be augmented with `prepare_data.py --input
data/three_d.pt --output data/explicit_h.pt`. Dataset format and evaluation notes
are in [docs/validation.md](docs/validation.md).

## Train

```bash
python train.py --data data/pampa.pt --strategy random \
  --output-dir runs/pampa --device cpu --num-workers 0
```

Defaults: 30 epochs, batch size 16, learning rate 1e-4, no gradient accumulation.
All encoders are trainable. Add `--jepa-weight 0.1` to train NCMol+JEPA.
Use `--device cuda:0` for GPU execution.

For PAMPA-to-cellular adaptation:

```bash
python train.py --data data/cellular.pt --strategy random \
  --model-init runs/pampa/best.pt --output-dir runs/cellular_transfer \
  --device cpu --num-workers 0
```

The task head is initialized anew; representation modules are transferred and
fine-tuned. Omit `--model-init` for direct training. An optional `--checkpoint`
backbone initialization file is supported for legacy reproduction.

Labels are standardized with training-set statistics. Loss uses standardized
labels; metrics and exported predictions use the original modelling scale.
The lowest validation RMSE selects `best.pt`. Test metrics are recorded each epoch
but must not guide selection. Non-finite updates are skipped and recorded.
Existing training runs are protected against accidental overwrite.

Each run produces `best.pt`, `config.json`, `history.jsonl`, `metrics.json`,
`predictions_val.csv` and `predictions_test.csv`.

## Predict

A trained NCMol checkpoint is required. Supply a CSV with a `smiles` column:

```bash
python predict.py --checkpoint runs/pampa/best.pt \
  --input-csv molecules.csv --output outputs/predictions.csv --device cpu
```

To evaluate saved conformers without regeneration:

```bash
python predict.py --checkpoint runs/pampa/best.pt --data data/pampa.pt \
  --strategy random --split test --output outputs/test.csv
```

Both standard and JEPA checkpoints are detected automatically. Predictions are
averaged over conformers and restored using the checkpoint's label statistics.
Output includes `input_index`, `smiles`, `y_pred` and `n_conformers`; dataset
evaluation also includes `y_true`. Log-scale outputs are not exponentiated.
Only load trusted checkpoint and dataset files.

## Verify

```bash
python tests/smoke_inference.py
```

This checks short NCMol, JEPA and transfer training runs, checkpoint loading,
SMILES inference, conformer averaging, scaling and prediction consistency.
[Validation details](docs/validation.md) distinguish software tests from experiments.

## Repository

```text
ncmol/
  model.py          NCMol, recurrent topology, cross-modal updates and JEPA
  geometry.py       Integrated geometric encoder and interactions
  backbone.py       Geometric encoding and checkpoint-compatible pooling
  data.py           Dataset and atom-aligned collation
  inference.py      Checkpoint loading and conformer generation
  utils.py          Metrics, seeding and device helpers
train.py            Training and evaluation
predict.py          Standalone inference
prepare_csv.py      Preparation from labelled SMILES
prepare_data.py     Topology augmentation of stored 3D inputs
scripts/run_all.sh  Retained experiment grid
```

## License and attribution

NCMol is released under the [MIT license](LICENSE). The integrated geometry
implementation is adapted from [LiTEN-FF](https://github.com/lingcon01/LiTEN-FF),
revision `ee38b4ceda1b4f699d8e6b294f38255a70746cc1`; its original MIT copyright
and license are preserved in [licenses/LiTEN-FF-MIT.txt](licenses/LiTEN-FF-MIT.txt).
