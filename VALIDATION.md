# Validation record

Date: 2026-10-17. These checks validate software behavior, not benchmark accuracy.

## Tested environment

- macOS, CPU, Python 3.12.13
- PyTorch 2.4.0; PyTorch Geometric 2.7.0
- NumPy 2.4.6; SciPy 1.15.3; scikit-learn 1.7.2
- RDKit 2025.09.1
- External LiTEN-FF commit ee38b4ceda1b4f699d8e6b294f38255a70746cc1

The real upstream six-layer geometric encoder was used, with native PyTorch
scatter fallbacks. No mock geometric network was substituted.

## End-to-end checks passed

1. Labelled CSV preparation, explicit-H atom correspondence and 3D conformers;
   round-trip topology augmentation using the legacy preparation entry point.
2. One training epoch with backpropagation for NCMol, NCMol+JEPA and
   task-checkpoint transfer initialization. No non-finite batch skips occurred.
3. Strict checkpoint reload and standalone processed-data inference for all
   three variants; predictions agreed with the training evaluation export
   at batch sizes 1 and 2 (absolute tolerance 2e-5, relative tolerance 1e-5).
4. Macrocycle multi-conformer prediction: per-conformer mean equals molecular
   prediction; inverse scaling uses training-only label statistics.
5. Raw-SMILES CSV to conformer generation to JEPA checkpoint inference and CSV.
6. Invalid SMILES fail without creating an output; zero scaling std is rejected;
   mismatched 2D/3D atomic numbers and incomplete model states are rejected;
   singleton correlations return NaN.

Run from the release root:

```bash
python tests/smoke_inference.py --vendor-dir /path/to/LiTEN-FF
```

The test creates temporary toy training data and weights and removes them at
completion. The toy labels have no experimental interpretation. OpenMP shared
memory was unavailable in the execution sandbox; the successful test ran as a
normal local process outside that sandbox.

## Fixes made during release preparation

- Added independent inference and labelled-CSV preparation entry points.
- Removed mandatory initialization-file dependency for scratch training and
  trained-checkpoint inference; retained support for legacy initialization.
- Fixed result aggregation to use the release root instead of its parent.
- Added guards against split-key overlap, empty datasets, invalid labels,
  overwritten training runs, invalid scaling and mismatched transfer states.
- Added finite gradient-norm checks and explicit failure when an epoch has no
  usable training update.
- Checkpoint selection now follows the exact lowest validation RMSE, without
  the previous 1e-5 improvement threshold.
- Replaced unavailable strict future dependency pins with documented ranges.
- Recorded and pinned the external encoder revision in installation instructions.

## Boundaries

Paper-trained checkpoints and the original processed experiment datasets were
not available locally. Historical checkpoint compatibility and published
prediction reproduction therefore remain unverified. CUDA, other operating
systems and a fresh dependency installation have not been tested. These tests
do not establish identical conformer generation across RDKit versions.

Element-order checks cannot detect permutations between identical elements;
both streams must be generated from the same RDKit atom indexing. Random
initialization without a legacy initialization file changes the RNG sequence
relative to experiments that loaded a separately generated random backbone.
For exact legacy reproduction, supply the original initialization and data.
The code-only release does not contain paper weights or external source code.
