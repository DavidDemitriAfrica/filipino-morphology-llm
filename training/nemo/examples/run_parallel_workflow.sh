#!/bin/bash
# Example workflow: Parallel preprocessing + training with chunks
#
# This demonstrates the complete workflow from raw JSONL to training
# using parallel chunk preprocessing for optimal performance.

set -e

echo "=========================================="
echo "Parallel Data Preprocessing + Training"
echo "=========================================="
echo ""

# Configuration
NUM_CHUNKS=20
INPUT_JSONL="/scratch_aisg/SPEC-SF-AISG/railey/data/corpora/seapile-v2.jsonl"
TOKENIZER="google/gemma-3-1b-pt"
TOKENIZER_NAME=$(echo ${TOKENIZER} | sed 's/\//-/g')  # Convert slashes to dashes
CHUNK_DIR="/scratch_aisg/SPEC-SF-AISG/railey/data/chunks/${TOKENIZER_NAME}"
OUTPUT_DIR="/scratch_aisg/SPEC-SF-AISG/railey/data/processed/${TOKENIZER_NAME}"

echo "Configuration:"
echo "  Input: ${INPUT_JSONL}"
echo "  Tokenizer: ${TOKENIZER}"
echo "  Tokenizer Name: ${TOKENIZER_NAME}"
echo "  Chunks: ${NUM_CHUNKS}"
echo "  Chunk dir: ${CHUNK_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Step 1: Split JSONL into chunks
echo "Step 1: Splitting JSONL into ${NUM_CHUNKS} chunks..."
echo "----------------------------------------"
python training/nemo/data/split_jsonl.py \
    --input "${INPUT_JSONL}" \
    --output-dir "$(dirname ${CHUNK_DIR})" \
    --num-chunks ${NUM_CHUNKS} \
    --tokenizer "${TOKENIZER}"

echo "✓ Split complete"
echo ""

# Step 2: Submit parallel preprocessing jobs
echo "Step 2: Submitting parallel preprocessing jobs..."
echo "----------------------------------------"
PREPROCESS_JOB=$(qsub -J 1-${NUM_CHUNKS} -v TOKENIZER="${TOKENIZER}" jobs/preprocess_data_parallel.pbs)
PREPROCESS_JOB_ID=$(echo $PREPROCESS_JOB | cut -d'[' -f1)

echo "✓ Submitted job: ${PREPROCESS_JOB}"
echo ""
echo "Monitoring preprocessing progress:"
echo "  qstat -t ${PREPROCESS_JOB_ID}"
echo "  # Or watch continuously:"
echo "  watch -n 5 'qstat -t ${PREPROCESS_JOB_ID}'"
echo ""
echo "Waiting for preprocessing to complete..."
echo "(Press Ctrl+C if you want to monitor manually)"
echo ""

# Wait for preprocessing to complete
while qstat -t "${PREPROCESS_JOB_ID}" &> /dev/null; do
    sleep 30
    echo -n "."
done
echo ""
echo "✓ Preprocessing complete"
echo ""

# Step 3: Verify all chunks were created
echo "Step 3: Verifying preprocessed chunks..."
echo "----------------------------------------"
missing_chunks=0
for i in $(seq -f "%04g" 1 ${NUM_CHUNKS}); do
    chunk_prefix="${OUTPUT_DIR}/chunk_${i}"
    if [ ! -f "${chunk_prefix}_text_document.bin" ] || [ ! -f "${chunk_prefix}_text_document.idx" ]; then
        echo "✗ Missing: ${chunk_prefix}_text_document.{bin,idx}"
        missing_chunks=$((missing_chunks + 1))
    fi
done

if [ $missing_chunks -gt 0 ]; then
    echo ""
    echo "✗ Error: ${missing_chunks} chunks are missing!"
    echo "Check preprocessing logs in: /scratch_aisg/SPEC-SF-AISG/railey/logs/preprocessing/"
    exit 1
fi

echo "✓ All ${NUM_CHUNKS} chunks verified"
echo ""

# Step 4: Generate chunk paths for training
echo "Step 4: Generating chunk paths for training..."
echo "----------------------------------------"
CHUNK_PATHS=$(training/nemo/data/generate_chunk_paths.sh ${NUM_CHUNKS} "${TOKENIZER}")
echo "✓ Chunk paths generated"
echo ""

# Step 5: Submit training job
echo "Step 5: Submitting training job with ${NUM_CHUNKS} chunks..."
echo "----------------------------------------"

TRAIN_JOB=$(qsub -v DATA_PATH="${CHUNK_PATHS}" jobs/run_cpt.pbs)

echo "✓ Submitted training job: ${TRAIN_JOB}"
echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Training job submitted: ${TRAIN_JOB}"
echo ""
echo "Monitor training:"
echo "  qstat ${TRAIN_JOB}"
echo "  tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/${TRAIN_JOB}.OU"
echo ""
echo "View WandB logs:"
echo "  Check your WandB project at: https://wandb.ai"
echo ""
