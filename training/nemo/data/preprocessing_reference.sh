#!/bin/bash
# Quick reference for preprocessing with different tokenization modes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo "Data Preprocessing Quick Reference"
echo "================================================================"
echo ""
echo "This script provides examples for preprocessing with different"
echo "tokenization modes: vanilla (default) and stochastok."
echo ""
echo "================================================================"
echo ""

# Function to print example commands
print_section() {
    echo -e "${BLUE}$1${NC}"
    echo "----------------------------------------------------------------"
}

print_command() {
    echo -e "${GREEN}# $1${NC}"
    echo -e "${YELLOW}$2${NC}"
    echo ""
}

# Single File Preprocessing
print_section "📄 SINGLE FILE PREPROCESSING"
echo ""

print_command "Vanilla tokenization (default):" \
"qsub jobs/preprocess_data.pbs"

print_command "Stochastok tokenization (10% expansion):" \
"qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data.pbs"

print_command "Stochastok tokenization (20% expansion):" \
"qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.2 jobs/preprocess_data.pbs"

print_command "Stochastok with custom seed:" \
"qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.15,SEED=12345 jobs/preprocess_data.pbs"

echo ""

# Parallel Chunk Preprocessing
print_section "⚡ PARALLEL CHUNK PREPROCESSING (FASTER)"
echo ""

print_command "Vanilla tokenization (20 chunks):" \
"qsub -J 1-20 jobs/preprocess_data_parallel.pbs"

print_command "Stochastok tokenization (20 chunks, 10% expansion):" \
"qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data_parallel.pbs"

print_command "Stochastok tokenization (20 chunks, 15% expansion):" \
"qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.15 jobs/preprocess_data_parallel.pbs"

echo ""

# Test with Single Chunk
print_section "🧪 TEST WITH SINGLE CHUNK FIRST"
echo ""

print_command "Test vanilla preprocessing:" \
"qsub jobs/preprocess_test_chunk1.pbs"

print_command "Test stochastok preprocessing:" \
"qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_test_chunk1.pbs"

echo ""

# Experimental Workflow
print_section "🔬 RECOMMENDED EXPERIMENTAL WORKFLOW"
echo ""

echo "1. Test preprocessing on a single chunk:"
echo "   qsub jobs/preprocess_test_chunk1.pbs"
echo "   qsub -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_test_chunk1.pbs"
echo ""

echo "2. If successful, preprocess all chunks in parallel:"
echo "   # Vanilla"
echo "   qsub -J 1-20 jobs/preprocess_data_parallel.pbs"
echo ""
echo "   # Stochastok"
echo "   qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.1 jobs/preprocess_data_parallel.pbs"
echo ""

echo "3. Train models on each preprocessed dataset:"
echo "   # Model A: Vanilla tokenization"
echo "   qsub -v DATA_PATH=/workspace/data/processed/google-gemma-3-1b-pt jobs/run_cpt.pbs"
echo ""
echo "   # Model B: Stochastok tokenization"
echo "   qsub -v DATA_PATH=/workspace/data/processed/google-gemma-3-1b-pt_stochastok_0.1 jobs/run_cpt.pbs"
echo ""

echo "4. Evaluate and compare model performance"
echo ""

# Configuration Options
print_section "⚙️  CONFIGURATION OPTIONS"
echo ""

echo "Common Parameters:"
echo "  TOKENIZER          - HuggingFace tokenizer (default: google/gemma-3-1b-pt)"
echo "  WORKERS            - Number of workers (default: 64)"
echo "  JSON_KEY           - JSON key for text (default: text)"
echo ""

echo "Stochastok Parameters:"
echo "  TOKENIZATION_MODE  - 'vanilla' or 'stochastok' (default: vanilla)"
echo "  EXPAND_PROP        - Expansion proportion (default: 0.1 = 10%)"
echo "  SEED               - Random seed (default: 42)"
echo ""

echo "Example with custom configuration:"
echo "  qsub -J 1-20 -v TOKENIZATION_MODE=stochastok,EXPAND_PROP=0.15,SEED=99,WORKERS=32 jobs/preprocess_data_parallel.pbs"
echo ""

# Output Organization
print_section "📁 OUTPUT ORGANIZATION"
echo ""

echo "Preprocessed data is automatically organized by mode:"
echo ""
echo "data/processed/"
echo "├── google-gemma-3-1b-pt/                    # Vanilla"
echo "│   ├── chunk_0001_text_document.bin"
echo "│   └── chunk_0001_text_document.idx"
echo "├── google-gemma-3-1b-pt_stochastok_0.1/     # Stochastok 10%"
echo "│   ├── chunk_0001_text_document.bin"
echo "│   └── chunk_0001_text_document.idx"
echo "└── google-gemma-3-1b-pt_stochastok_0.2/     # Stochastok 20%"
echo "    ├── chunk_0001_text_document.bin"
echo "    └── chunk_0001_text_document.idx"
echo ""

# Monitoring
print_section "👀 MONITORING JOBS"
echo ""

echo "Check job status:"
echo "  qstat -u \$USER"
echo ""

echo "Check specific array job:"
echo "  qstat -t <job_id>"
echo ""

echo "View logs:"
echo "  tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<job_id>.OU"
echo ""

echo "Check preprocessing progress in detail:"
echo "  tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/preprocessing/<job_id>/preprocessing.log"
echo ""

# Tips
print_section "💡 TIPS"
echo ""

echo "• Always test on a single chunk first (preprocess_test_chunk1.pbs)"
echo "• Use parallel processing for large datasets (preprocess_data_parallel.pbs)"
echo "• Stochastok preprocessing is slower - be patient!"
echo "• Check expansion statistics in logs to verify stochastok is working"
echo "• Use different EXPAND_PROP values (0.1, 0.15, 0.2) to compare effects"
echo "• Set different SEED values for multiple runs"
echo ""

echo "================================================================"
echo "For more details, see: training/nemo/data/README_TOKENIZATION.md"
echo "================================================================"
