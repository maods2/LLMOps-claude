#!/usr/bin/env bash
# Training container entry point.
# Dispatches to the correct training script based on the STAGE env var or $1 arg.
set -euo pipefail

STAGE="${1:-${STAGE:-pretrain}}"
echo "[entrypoint] Starting training stage: $STAGE"

case "$STAGE" in
  pretrain)
    exec python3 -m services.training.pretrain.train \
      --config configs/pretrain.yaml \
      --output_dir "${OUTPUT_DIR}/pretrain"
    ;;
  sft)
    exec python3 -m services.training.sft.train \
      --config configs/sft.yaml \
      --output_dir "${OUTPUT_DIR}/sft"
    ;;
  lora)
    exec python3 -m services.training.lora.train \
      --config configs/lora.yaml \
      --output_dir "${OUTPUT_DIR}/lora"
    ;;
  dpo)
    exec python3 -m services.training.dpo.train \
      --config configs/dpo.yaml \
      --output_dir "${OUTPUT_DIR}/dpo"
    ;;
  evaluate)
    exec python3 -m evaluation.benchmarks.run \
      --config configs/eval.yaml \
      --model_path "${MODEL_PATH:-${OUTPUT_DIR}/pretrain/best}"
    ;;
  *)
    echo "[entrypoint] Unknown stage: $STAGE"
    echo "Available stages: pretrain | sft | lora | dpo | evaluate"
    exit 1
    ;;
esac
