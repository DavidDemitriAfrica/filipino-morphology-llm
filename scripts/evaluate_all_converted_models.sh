#!/bin/bash
# Evaluate all converted checkpoints on all 5 benchmarks

set -e

echo "========================================="
echo "Evaluating All Converted Models"
echo "========================================="

# Define models
MODELS=(
    "vanilla-step4999:checkpoints/hf/gemma2-2b-vanilla-step4999"
    "stochastok-step4999:checkpoints/hf/gemma2-2b-stochastok-step4999"
    "patok-step4999:checkpoints/hf/gemma2-2b-patok-step4999"
)

# Define benchmarks
BENCHMARKS="pacute hierarchical cute langgame multi-digit-addition"

# Evaluate each model
for model_info in "${MODELS[@]}"; do
    name="${model_info%%:*}"
    path="${model_info#*:}"

    echo ""
    echo "========================================="
    echo "Evaluating: $name"
    echo "========================================="

    python scripts/evaluate_converted_models.py \
        --model-path "$path" \
        --model-name "$name" \
        --benchmarks $BENCHMARKS \
        --output-dir results/converted_models

    echo "✓ Completed: $name"
done

echo ""
echo "========================================="
echo "All Evaluations Complete!"
echo "========================================="

# Show results summary
echo ""
echo "Results saved in:"
ls -lh results/converted_models/*/evaluation_results_*.json
