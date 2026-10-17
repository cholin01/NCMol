# NCMol

Topology-conditioned multi-conformer learning for nonpeptidic macrocycle
permeability prediction. This code-only release contains NCMol-specific model,
training, data preparation, inference and evaluation workflows. Datasets,
trained weights, experiment outputs and third-party source code are excluded.

## Installation

Python 3.10+ is required. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git clone https://github.com/lingcon01/LiTEN-FF.git vendor/LiTEN-FF
git -C vendor/LiTEN-FF checkout ee38b4ceda1b4f699d8e6b294f38255a70746cc1
```

LiTEN-FF supplies the external geometric architecture; consult its repository
for terms of use. Its source remains outside this code release, in the ignored
vendor directory. This does not require SPICE weights. On CUDA, install a
PyTorch build compatible with your driver before installing other requirements.
Native PyTorch scatter fallbacks and precomputed spatial edges support the
non-periodic path without torch-scatter or torch-cluster.

## Models

- NCMol: supervised regression and bidirectional 2D–3D representation alignment.
- NCMol+JEPA: adds an EMA target and cross-conformer latent prediction.
- PAMPA adaptation: initializes representation modules from a trained PAMPA
  NCMol model, resets the task regression head, and fine-tunes all encoders.

Direct runs start from random initialization when --checkpoint is omitted.
An optional backbone checkpoint supports legacy initialization. The complete
trained best.pt is sufficient for inference; the original initialization
checkpoint is not needed. Adapted models use task-trained PAMPA weights.

## Prepare data

Supply a CSV with smiles,y,split columns. Labels must be on the desired
endpoint modelling scale (e.g. log permeability), not z-scores. Assign train,
val and test splits before preparation. A format-only example:

```text
smiles,y,split
CCO,-6.0,train
CCN,-5.0,train
CCCC,-6.4,val
CCCO,-5.2,test
```

These are toy examples, not scientific training data. Aggregate repeated assay
measurements before splitting. Duplicate canonical isomeric SMILES and invalid
labels are rejected rather than silently removed.

```bash
python prepare_csv.py --input labelled.csv --output data/my_endpoint.pt \
  --strategy random --candidate-conformers 10 --keep-conformers 3
```

The strategy flag names the supplied partition; it does not construct a
random, cliff or scaffold split. ETKDGv3 generates explicit-H conformers with
macrocycle settings, followed by MMFF94s or UFF when MMFF parameters are
unavailable. Up to three lowest-energy conformers are retained; RMS pruning
may yield fewer. Finite non-converged optimization results retain their status
in the saved data. Preparation fails if no usable conformer is available.
Spatial edges use a 5 Å cutoff and at most 32 neighbors including self edges.
Both streams share RDKit atom order.

Existing trusted 3D payloads can instead be augmented:

```bash
python prepare_data.py --input data/three_d.pt --output data/explicit_h.pt
```

Payloads contain graphs and splits[strategy][train|val|test]. Each row has
key,smiles,y. Each graph has smiles,conformers; each conformer has tensor
z,pos,edge_index. Topology preparation adds z_2d,x_2d,edge_index_2d,edge_attr_2d.
Do not reorder conformer atoms independently of topology.

## Train and adapt

```bash
python train.py --data data/my_endpoint.pt --strategy random \
  --vendor-dir vendor/LiTEN-FF --output-dir runs/direct \
  --device cpu --epochs 30 --batch-size 16 --num-workers 0

python train.py --data data/my_endpoint.pt --strategy random \
  --vendor-dir vendor/LiTEN-FF --output-dir runs/direct_jepa \
  --device cpu --epochs 30 --batch-size 16 --num-workers 0 --jepa-weight 0.1

python train.py --data data/cellular.pt --strategy random \
  --vendor-dir vendor/LiTEN-FF --output-dir runs/transfer \
  --device cpu --num-workers 0 --model-init runs/pampa/best.pt
```

Use --device cuda:0 for a suitable GPU. Default learning rate is 1e-4; no
gradient accumulation is used. Label mean/std are fitted on training labels
only. Supervised loss uses standardized labels; metrics and exported
predictions are restored to the original modelling scale. Minimum validation
RMSE selects the checkpoint. The retained protocol logs test metrics each
epoch; they must not be used for model selection. Non-finite losses, gradients
or gradient norms skip the batch; an epoch with no valid update fails.
Existing output runs are not overwritten.

Outputs: config.json, history.jsonl, best.pt, metrics.json, predictions_val.csv
and predictions_test.csv. Undefined singleton/constant correlations report NaN.

Run bash scripts/run_all.sh for the retained experiment grid after obtaining
processed PAMPA and cellular inputs. Environment overrides are PYTHON_BIN,
VENDOR_DIR, CELL_SOURCE_DIR, DEVICE and optional RANDOM_INIT.
python summarize_results.py aggregates this repository's runs directory.

## Inference

Provide a trained NCMol checkpoint for the desired endpoint; none is bundled.
Only load trusted .pt files: legacy checkpoints use Python pickle, which can
execute code. The loader checks the complete parameter state and label
statistics; it never falls back to random weights. Base and JEPA checkpoints
are detected automatically.

For a CSV containing a smiles column, with no labels required:

```bash
python predict.py --checkpoint runs/direct/best.pt \
  --vendor-dir vendor/LiTEN-FF --input-csv molecules.csv \
  --output outputs/predictions.csv --device cpu --batch-size 16
```

For exact reevaluation using saved training conformers:

```bash
python predict.py --checkpoint runs/direct/best.pt \
  --vendor-dir vendor/LiTEN-FF --data data/my_endpoint.pt \
  --strategy random --split test --output outputs/test.csv
```

Inference averages conformer predictions and applies checkpoint mean/std.
Output columns: input_index,smiles,y_pred,n_conformers; processed-data mode
also exports y_true. Input order and duplicate input rows are retained.
Invalid molecules and non-finite predictions fail explicitly.
y_pred stays on the checkpoint modelling scale, e.g. log permeability; it is
not automatically exponentiated. Regenerated conformers can yield different
results from stored conformers, particularly across RDKit versions.

## Verification

```bash
python tests/smoke_inference.py --vendor-dir vendor/LiTEN-FF
```

The test uses the real geometric encoder and temporary toy data. It runs one
epoch each for NCMol, NCMol+JEPA and transfer initialization, reloads each
checkpoint and compares CLI inference with training evaluation at two batch
sizes. It also checks raw-SMILES inference, conformer means, train-only label
scaling and invalid input handling. Temporary weights are removed. Execution
tests do not establish paper-checkpoint accuracy. See VALIDATION.md.

## Source layout

- train.py: architecture, collation, objectives and training.
- ncmol/backbone.py: external geometry encoder adapter.
- ncmol/inference.py and predict.py: checkpoint loading and inference.
- prepare_csv.py and prepare_data.py: data preparation.
- ncmol/compat.py and ncmol/utils.py: compatibility, scaling helpers and metrics.
- configs/experiment_protocol.json: retained optimization settings.
- tests/smoke_inference.py: end-to-end verification.

Do not commit datasets, external source trees, weights, manuscripts or
experimental prediction tables to this code-only release.
