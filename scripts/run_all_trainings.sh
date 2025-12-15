#!/bin/bash
# Run all three tokenization trainings sequentially
# Usage: nohup ./scripts/run_all_trainings.sh > logs/all_trainings.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MAX_STEPS="${1:-5000}"

echo "=============================================="
echo "Filipino Morphology LLM - Full Training Suite"
echo "=============================================="
echo "Max steps per model: $MAX_STEPS"
echo "Start time: $(date)"
echo "=============================================="

# 1. Vanilla training
echo ""
echo "[1/3] Starting VANILLA training..."
echo "=============================================="
./scripts/run_cpt_training.sh vanilla "$MAX_STEPS"
echo "Vanilla training completed at $(date)"

# 2. Stochastok training
echo ""
echo "[2/3] Starting STOCHASTOK training..."
echo "=============================================="
./scripts/run_cpt_training.sh stochastok "$MAX_STEPS"
echo "Stochastok training completed at $(date)"

# 3. Patok training
echo ""
echo "[3/3] Starting PATOK training..."
echo "=============================================="
./scripts/run_cpt_training.sh patok "$MAX_STEPS"
echo "Patok training completed at $(date)"

echo ""
echo "=============================================="
echo "All trainings completed!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Checkpoints saved to:"
echo "  - /logs/checkpoints/gemma2-2b-vanilla"
echo "  - /logs/checkpoints/gemma2-2b-stochastok"
echo "  - /logs/checkpoints/gemma2-2b-patok"
