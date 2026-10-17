#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON_BIN:-python}"
vendor_dir="${VENDOR_DIR:-vendor/LiTEN-FF}"
random_init="${RANDOM_INIT:-}"
cell_source_dir="${CELL_SOURCE_DIR:-data/processed/cell_liten}"

common=(
  --vendor-dir "${vendor_dir}"
  --device "${DEVICE:-cuda:0}"
  --seed 42 --batch-size 16 --epochs 30 --lr 1e-4
  --alignment-weight 0.1 --num-workers 4
)
if [[ -n "${random_init}" ]]; then
  common+=(--checkpoint "${random_init}")
fi

for split in random cliff scaffold; do
  "${python_bin}" train.py \
    --data data/processed/npmmpd_2d3d_explicit_h.pt \
    --strategy "${split}" \
    --output-dir "runs/ncmol_pampa/${split}" \
    "${common[@]}"
  "${python_bin}" train.py \
    --data data/processed/npmmpd_2d3d_explicit_h.pt \
    --strategy "${split}" \
    --output-dir "runs/ncmol_pampa_jepa/${split}" \
    --jepa-weight 0.1 --jepa-momentum 0.996 \
    "${common[@]}"
done

endpoints=(
  caco2_log_papp_ab caco2_log_papp_ba caco2_log10_er
  mdck_log_papp_ab mdck_log10_er
)
mkdir -p data/processed/cell_2d3d_explicit_h
for endpoint in "${endpoints[@]}"; do
  "${python_bin}" prepare_data.py \
    --input "${cell_source_dir}/${endpoint}.pt" \
    --output "data/processed/cell_2d3d_explicit_h/${endpoint}.pt"
done

for endpoint in "${endpoints[@]}"; do
  for split in random scaffold; do
    for variant in base jepa; do
      extra=()
      [[ "${variant}" == "jepa" ]] && extra+=(--jepa-weight 0.1 --jepa-momentum 0.996)
      "${python_bin}" train.py \
        --data "data/processed/cell_2d3d_explicit_h/${endpoint}.pt" \
        --strategy "${split}" \
        --output-dir "runs/ncmol_cell/${endpoint}/${split}/${variant}" \
        "${extra[@]}" "${common[@]}"
    done

    for variant in base jepa; do
      extra=()
      source="runs/ncmol_pampa/${split}/best.pt"
      if [[ "${variant}" == "jepa" ]]; then
        source="runs/ncmol_pampa_jepa/${split}/best.pt"
        extra+=(--jepa-weight 0.1 --jepa-momentum 0.996)
      fi
      "${python_bin}" train.py \
        --data "data/processed/cell_2d3d_explicit_h/${endpoint}.pt" \
        --strategy "${split}" --model-init "${source}" \
        --output-dir "runs/ncmol_pampa_transfer/${endpoint}/${split}/${variant}" \
        "${extra[@]}" "${common[@]}"
    done
  done
done

"${python_bin}" summarize_results.py
