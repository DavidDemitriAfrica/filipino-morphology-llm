#!/bin/bash
# Convert all 3 trained NeMo checkpoints to HuggingFace format
set -e

echo "========================================="
echo "Converting ALL NeMo Checkpoints to HF"
echo "========================================="

CHECKPOINTS=(
    "vanilla:logs/checkpoints/gemma2-2b-vanilla/val_loss=2.99-step=4999-consumed_samples=320000.0"
    "stochastok:logs/checkpoints/gemma2-2b-stochastok/val_loss=3.34-step=4999-consumed_samples=320000.0"
    "patok:logs/checkpoints/gemma2-2b-patok/val_loss=3.42-step=4999-consumed_samples=320000.0"
)

for ckpt_info in "${CHECKPOINTS[@]}"; do
    name="${ckpt_info%%:*}"
    path="${ckpt_info#*:}"

    echo ""
    echo "========================================="
    echo "Processing: $name"
    echo "========================================="

    # Step 1: Extract raw weights
    echo "1. Extracting raw weights..."
    python scripts/inspect_and_load_checkpoint.py \
        --checkpoint "$path" \
        --output "checkpoints/raw_nemo/${name}-step4999.pt"

    # Step 2: Convert to HF
    echo "2. Converting to HuggingFace format..."
    python scripts/convert_megatron_to_hf.py \
        --megatron-checkpoint "checkpoints/raw_nemo/${name}-step4999.pt" \
        --base-model google/gemma-2-2b \
        --output "checkpoints/hf/gemma2-2b-${name}-step4999"

    echo "✓ Done: $name"
done

echo ""
echo "========================================="
echo "All checkpoints converted!"
echo "========================================="
ls -lh checkpoints/hf/
