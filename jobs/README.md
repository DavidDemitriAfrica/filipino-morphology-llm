# PBS Job Scripts

This directory contains PBS job scripts for running various tasks on the Hopper cluster.

## Available Jobs

### Training Jobs

#### `run_cpt.pbs`
Continued pretraining job using NeMo Framework container with Enroot.

**Usage:**
```bash
# Run with defaults from .env
qsub jobs/run_cpt.pbs

# Override hyperparameters
qsub -v MAX_STEPS=1000,GBS=512,LR=5e-5 jobs/run_cpt.pbs
```

**Key Parameters:**
- `MAX_STEPS`: Training steps (default: 100)
- `GBS`: Global batch size (default: 256)
- `MBS`: Micro batch size (default: 2)
- `LR`: Learning rate (default: 1e-4)
- `RESUME_FROM`: Model checkpoint to resume from (default: google/gemma-3-1b-pt)

**Requirements:**
- Enroot container must be set up (`setup_enroot.sh`)
- `.env` file must be configured
- Data must be preprocessed (see `preprocess_data.pbs`)

---

### Data Preprocessing Jobs

#### `preprocess_data.pbs`
Preprocesses JSONL data into Megatron binary format (.bin + .idx).

**Usage:**
```bash
# Run with defaults from .env
qsub jobs/preprocess_data.pbs

# Override paths
qsub -v INPUT=/path/to/data.jsonl,OUTPUT_PREFIX=/path/to/output jobs/preprocess_data.pbs
```

**Key Parameters:**
- `INPUT`: Input JSONL file (default: /workspace/data/corpora/seapile-v2.jsonl)
- `OUTPUT_PREFIX`: Output file prefix (default: /workspace/data/processed/seapile-v2)
- `TOKENIZER`: Tokenizer model (default: google/gemma-3-1b-pt)
- `WORKERS`: Number of workers (default: 64)

**Note:** Run this ONCE before training.

---

### Evaluation Jobs

#### `run_evaluation_test.pbs`
Quick evaluation test with a small model and limited samples.

**Usage:**
```bash
# Run quick test with GPT-2
qsub jobs/run_evaluation_test.pbs

# Test with different model
qsub -v TEST_MODEL=qwen-2.5-0.5b jobs/run_evaluation_test.pbs

# Test with different benchmark
qsub -v TEST_BENCHMARKS=hierarchical jobs/run_evaluation_test.pbs
```

**Key Parameters:**
- `TEST_MODEL`: Model to test (default: gpt2)
- `TEST_BENCHMARKS`: Benchmarks to test (default: pacute)
- `MAX_SAMPLES`: Sample limit (default: 100)

**Resources:**
- 1 node, 8 CPUs, 1 GPU, 32GB RAM
- 1 hour walltime
- Good for testing before full batch

---

#### `run_evaluation_batch.pbs`
Batch evaluation on multiple models and benchmarks.

**Usage:**
```bash
# Run all models on all benchmarks
qsub jobs/run_evaluation_batch.pbs

# Run specific model groups
qsub -v MODELS_GPT2="gpt2",MODELS_QWEN="qwen-2.5-0.5b" jobs/run_evaluation_batch.pbs

# Run on specific benchmarks
qsub -v BENCHMARKS="pacute hierarchical" jobs/run_evaluation_batch.pbs

# Limit samples for testing
qsub -v MAX_SAMPLES=100 jobs/run_evaluation_batch.pbs
```

**Key Parameters:**
- `MODELS_GPT2`: GPT-2 models (default: gpt2 gpt2-medium gpt2-large)
- `MODELS_QWEN`: Qwen models (default: qwen-2.5-0.5b qwen-2.5-0.5b-it qwen-2.5-1.5b qwen-2.5-1.5b-it)
- `MODELS_CEREBRAS`: Cerebras models (default: cerebras-gpt-111m cerebras-gpt-256m cerebras-gpt-590m)
- `MODELS_LLAMA`: Llama models (default: llama-3.2-1b llama-3.2-1b-it)
- `MODELS_GEMMA`: Gemma models (default: gemma-2b gemma-2b-it)
- `BENCHMARKS`: Benchmarks to run (default: pacute cute hierarchical langgame math)
- `MAX_SAMPLES`: Sample limit (empty = all)
- `OUTPUT_DIR`: Output directory (default: results/benchmark_evaluation)

**Resources:**
- 1 node, 16 CPUs, 1 GPU, 64GB RAM
- 12 hours walltime

**Note:** Some models (Llama, Gemma) may require HuggingFace authentication.

---

## Workflow

### 1. Setup Environment
```bash
# First time setup
bash scripts/setup_environment.sh

# For container-based training
bash scripts/setup_enroot.sh
```

### 2. Preprocess Data (for training)
```bash
# Preprocess training data
qsub jobs/preprocess_data.pbs

# Check output
ls -lh data/processed/seapile-v2*
```

### 3. Run Training
```bash
# Quick test
qsub -v MAX_STEPS=10 jobs/run_cpt.pbs

# Full training
qsub jobs/run_cpt.pbs
```

### 4. Run Evaluation
```bash
# Quick test
qsub jobs/run_evaluation_test.pbs

# Full batch evaluation
qsub jobs/run_evaluation_batch.pbs
```

---

## Monitoring Jobs

```bash
# Check job status
qstat -u $USER

# View job details
qstat -f <job_id>

# Monitor logs (replace JOB_ID with actual ID)
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<JOB_ID>/training.log
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/evaluation/<JOB_ID>/evaluation_test.log

# Check GPU usage
ssh <node> nvidia-smi
```

---

## Common Issues

### Container Not Found
```bash
Error: Container nemo_framework not found
```
**Solution:** Run `bash scripts/setup_enroot.sh` to create the container.

### Environment Not Found
```bash
Error: Conda environment not found
```
**Solution:** Create the environment: `conda create -p env python=3.11`

### Out of Memory
**Solution:** Reduce batch size or use smaller model:
```bash
qsub -v MBS=1,GBS=128 jobs/run_cpt.pbs
```

### HuggingFace Authentication Required
**Solution:** Set HF_TOKEN in `.env` file or login:
```bash
huggingface-cli login
```

---

## Resource Guidelines

| Job Type | CPUs | GPUs | Memory | Walltime | Notes |
|----------|------|------|--------|----------|-------|
| Quick test | 8 | 1 | 32GB | 1h | Testing only |
| Evaluation | 16 | 1 | 64GB | 12h | Standard eval |
| Preprocessing | 64 | 0 | 512GB | 8h | One-time setup |
| Training (1B) | 32 | 4 | 256GB | 4h+ | Multi-GPU |
| Training (7B+) | 64 | 8 | 512GB | 24h+ | Large models |

---

## Environment Variables

Key environment variables (defined in `.env`):

**Paths:**
- `DATA_PATH`: Training data path
- `CKPT_DIR`: Checkpoint directory
- `LOG_DIR`: Log directory
- `HF_HOME`: HuggingFace cache
- `WANDB_API_KEY`: WandB authentication

**Enroot:**
- `ENROOT_PATH`: Enroot data path
- `ENROOT_CACHE_PATH`: Container cache
- `SQSH_PATH`: Container images path

**Training:**
- `MAX_STEPS`, `GBS`, `MBS`, `LR`, etc.

See `.env.example` for full list.
