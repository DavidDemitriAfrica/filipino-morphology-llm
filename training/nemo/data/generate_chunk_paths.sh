#!/bin/bash
# Helper script to generate DATA_PATH string for multiple chunks
#
# Usage:
#   # Generate paths for 20 chunks with default tokenizer
#   ./generate_chunk_paths.sh 20
#
#   # Generate paths for specific tokenizer
#   ./generate_chunk_paths.sh 20 google/gemma-3-1b-pt
#   ./generate_chunk_paths.sh 20 meta-llama/Llama-3.3-70B
#
#   # Use in qsub:
#   DATA_PATH=$(./generate_chunk_paths.sh 20) qsub jobs/run_cpt.pbs

set -euo pipefail

NUM_CHUNKS=${1:-20}
TOKENIZER=${2:-google/gemma-3-1b-pt}
TOKENIZER_NAME=$(echo ${TOKENIZER} | sed 's/\//-/g')  # Convert slashes to dashes
PREFIX="/workspace/data/processed/${TOKENIZER_NAME}/chunk"

paths=""
for i in $(seq -f "%04g" 1 $NUM_CHUNKS); do
    if [ -z "$paths" ]; then
        paths="${PREFIX}_${i}"
    else
        paths="${paths} ${PREFIX}_${i}"
    fi
done

echo "$paths"
