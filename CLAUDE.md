# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project testing whether **morpheme-aware tokenization** improves LLM performance on Filipino morphological tasks. Compares three tokenization approaches (Baseline BPE, Stochastok expansion, Patok affix-aware) on Gemma 2 2B using SEA-PILE v2 Filipino corpus (7.4GB) and 15,023 morphological evaluation tasks.

## Common Development Commands

### Environment Setup (One-Time)
```bash
# Configure environment variables
cp .env.example .env
nano .env  # Add: HF_TOKEN, WANDB_API_KEY, container paths
source .env

# Option A: Docker (recommended for cloud/local)
docker pull nvcr.io/nvidia/nemo:24.07
# Creates run_in_docker.sh automatically

# Option B: Enroot (recommended for HPC clusters)
bash training/nemo/setup/setup_enroot.sh
enroot list | grep nemo_framework
./run_in_enroot.sh python -c "import nemo; print(nemo.__version__)"

# Option C: Singularity/Apptainer (alternative HPC)
bash training/nemo/setup/setup_singularity.sh
```

### Data Preprocessing
```bash
# Download corpus
python scripts/download_seapile.py

# Batch preprocessing scripts (parallel, ~1hr each)
bash scripts/preprocess_all_vanilla.sh     # Baseline BPE
bash scripts/preprocess_all_stochastok.sh  # Stochastok expansion (~10%)
bash scripts/preprocess_all_patok.sh       # Patok morphology-aware (30%+30%)

# Or use PBS jobs for individual chunks
qsub jobs/preprocess_test_chunk1.pbs      # Test single chunk first
qsub -J 1-20 jobs/preprocess_data_parallel.pbs  # Full parallel processing

# Verify outputs
ls -lh data/processed/*.bin | wc -l  # Should show preprocessed chunks
```

### Training
```bash
# Quick test (10-100 steps) - Docker
./run_in_docker.sh python training/nemo/run_cpt.py --max-steps 100

# Quick test - Enroot
./run_in_enroot.sh python training/nemo/run_cpt.py --max-steps 100

# Full training via PBS
qsub jobs/run_cpt_test.pbs  # Test first
qsub jobs/run_cpt.pbs       # Full training

# Custom hyperparameters
qsub -v MAX_STEPS=5000,GBS=512,LR=5e-5 jobs/run_cpt.pbs

# Monitor training
qstat -u $USER
tail -f nemo_experiments/<experiment_name>/nemo_log_globalrank-0_localrank-0.txt
```

### Evaluation
```bash
# Generate all benchmarks
python scripts/generate_benchmarks.py

# Evaluate single model
python scripts/run_evaluation.py --models gpt2 --benchmarks pacute

# Evaluate specific format (MCQ or GEN)
python scripts/run_evaluation.py --models gpt2 --eval-mode mcq

# Full evaluation suite
bash scripts/run_full_evaluation.sh

# Analyze results
python scripts/analyze_inference_results.py

# Create visualizations
python scripts/create_visualizations.py

# View results
cat results/benchmark_evaluation/*.json
```

### Testing
```bash
# Run single test file
pytest tests/test_patok_morphology.py -v

# Run specific test
pytest tests/test_affixation.py::test_affix_identification -v

# Run all tests
pytest tests/ -v
```

## Architecture Overview

### Tokenization Pipeline

The core innovation is comparing three tokenization approaches:

1. **Baseline (Vanilla BPE)**: Standard tokenization using HuggingFace tokenizers
2. **Stochastok**: Stochastic token expansion (~10%) - randomly splits tokens into subword units
3. **Patok**: Affix-aware expand-contract (30%+30%) - preferentially splits/merges at morpheme boundaries

**Key Files:**
- `src/tokenization/base_processor.py` - Base class for tokenization processors
- `src/tokenization/stochastok_processor.py` - Implements stochastic expansion
- `src/tokenization/patok_morphology.py` - Filipino affix detection and morpheme boundary identification (RECOMMENDED)
- `src/tokenization/patok_processor.py` - DEPRECATED - use patok_morphology.py instead
- `src/tokenization/affix_decomposition.py` - Affix decomposition utilities

**Critical Implementation Detail**: Tokenization processors apply DURING preprocessing, not during model training. The `preprocess_data.py` script uses these processors to create different versions of the training data:
- Vanilla: `data/processed/vanilla/chunk_*.bin`
- Stochastok: `data/processed/stochastok/chunk_*.bin`
- Patok: `data/processed/patok/chunk_*.bin`

**Preprocessing Scripts**: Three batch preprocessing scripts automate the full preprocessing workflow:
- `scripts/preprocess_all_vanilla.sh` - Processes all chunks with vanilla tokenization
- `scripts/preprocess_all_stochastok.sh` - Processes with stochastic expansion
- `scripts/preprocess_all_patok.sh` - Processes with morphology-aware tokenization

### Training Architecture

Training uses **NeMo Framework 2.1+** which requires:
- **Megatron binary format** (.bin + .idx files) - NOT raw JSONL or HuggingFace Arrow
- Preprocessing MUST run inside the NeMo container where Megatron tools are available
- Data is memory-mapped for efficient multi-GPU distributed training

**Key Files:**
- `training/nemo/run_cpt.py` - Main CPT training script using NeMo 2.0 API
- `training/nemo/data/preprocess_data.py` - Converts JSONL → Megatron binary format
- `training/nemo/data/split_jsonl.py` - Splits large corpus into chunks for parallel preprocessing
- `scripts/convert_hf_to_nemo.py` - Converts HuggingFace checkpoints to NeMo format for distributed training
- `scripts/run_cpt_training.sh` - Wrapper script for training with different tokenization modes

**Critical Architecture Decision**: NeMo 2.0+ uses a different API than NeMo 1.x:
- Uses `nemo.collections.llm` package (not `nemo_nlp`)
- Model definition via `llm.GemmaConfig2()` and `llm.GemmaModel2()`
- Training via `nl.Trainer()` with strategy="ddp"
- Data loading via `PreTrainingDataModule` expecting Megatron format

**Distributed Training**: For multi-GPU training, HF checkpoints must be pre-converted to NeMo format:
```bash
./run_in_docker.sh python scripts/convert_hf_to_nemo.py --model google/gemma-2-2b
./run_in_docker.sh torchrun --nproc_per_node=8 training/nemo/run_cpt.py \
    --resume-from /workspace/checkpoints/nemo/google_gemma-2-2b
```

### Evaluation Architecture

Evaluation uses a **hierarchical diagnostic framework** to identify where models fail:

**Benchmark Hierarchy:**
- **PACUTE** (4,080 tasks): Filipino-specific morphology tests
  - Affixation: Identify/apply Filipino affixes (prefixes, infixes, suffixes)
  - Composition: Character counting, diacritics, word formation
  - Manipulation: Insert/delete/swap/replace characters
  - Syllabification: Syllable counting, stress, reduplication

- **Hierarchical** (1,198 tasks): 6-level diagnostic cascade
  - Level 0: Character Recognition → Level 1: Character Manipulation
  - Level 2: **Morpheme Decomposition** (critical bottleneck) → Level 3: Morpheme Manipulation
  - Level 4: Morpheme Composition → Level 5: Complex Reasoning
  - **Key Insight**: Level 2 failures cascade through Levels 3-5, making morpheme boundary understanding critical

- **CUTE** (1,400 tasks): Character understanding across 14 task types
- **LangGame** (2,000 tasks): Subword understanding via word games
- **Multi-digit Addition** (2,000 tasks): Numerical reasoning baseline

**Format Types:**
- **MCQ**: Log probability-based selection from 4 options
- **GEN**: Free-form generation with exact match scoring

**Key Files:**
- `src/evaluation/loaders/` - Benchmark loading (pacute.py, hierarchical.py, etc.)
- `src/evaluation/loaders/registry.py` - Registry pattern with EVALS_DICT
- `src/evaluation/datasets/generators/` - Generate benchmark tasks
- `src/evaluation/evaluators/` - Evaluation logic (MCQ vs GEN)
- `scripts/generate_benchmarks.py` - Generate all benchmark JSONL files
- `scripts/run_evaluation.py` - Run evaluation on models
- `scripts/run_full_evaluation.sh` - Comprehensive evaluation workflow
- `scripts/analyze_inference_results.py` - Analyze evaluation outputs
- `scripts/create_visualizations.py` - Generate performance visualizations

**Critical Evaluation Pattern**: Each benchmark supports filtering by evaluation mode via `--eval-mode mcq|gen|both`. Generators create both MCQ and GEN versions with task IDs linking equivalent questions. The `EVALS_DICT` in `src/evaluation/loaders/registry.py` maps benchmark names to loaders.

### PBS Job System

The codebase uses **template-based PBS job generation** to avoid committing cluster-specific paths:

- **Templates** (version-controlled): `job_templates/*.template.pbs` with placeholder variables
- **Generated Jobs** (gitignored): `jobs/*.pbs` with actual paths substituted
- **Setup Wizard**: `job_templates/setup_jobs.sh` - Interactive script to generate jobs from templates

**Environment Variables Pattern**: All cluster-specific configuration goes in `.env`:
```bash
HF_HOME=/scratch/$USER/cache/huggingface
WANDB_DIR=/scratch/$USER/logs/wandb
ENROOT_PATH=/scratch/$USER/enroot/
BIND_MOUNTS=/scratch/$USER/cache:/cache,/scratch/$USER/logs:/logs
```

This `.env` file is sourced by PBS jobs and provides configuration to the container runtime.

### Container Architecture

Three container runtimes supported:

**Docker** (Recommended for cloud/local):
- Uses Docker runtime with `nvidia-docker` or Docker with GPU support
- Container run via `./run_in_docker.sh` wrapper script
- GPU access via `--gpus all` flag
- Most portable and widely supported

**Enroot** (Recommended for HPC):
- Uses `.sqsh` (squashfs) images stored at `$ENROOT_PATH`
- Container managed via `enroot` commands
- Executed via `./run_in_enroot.sh` wrapper script
- GPU access automatic (no special flags needed)

**Singularity/Apptainer** (Alternative HPC):
- Uses `.sif` (Singularity Image Format) files
- Executed via `./run_in_singularity.sh` with `--nv` flag for GPU
- Container stored at `$CONTAINER_CACHEDIR`

**Critical Container Workflow**:
1. Container setup is ONE-TIME per user (creates reusable image)
2. Preprocessing MUST run inside container (Megatron tools only available there)
3. Training MUST run inside container (NeMo + optimizations)
4. Evaluation CAN run outside container (only needs transformers)

**Mount Points**: Project directory auto-mounted as `/workspace`, shared directories mounted via `$BIND_MOUNTS`:
- `/workspace` → current project directory
- `/cache` → HuggingFace model cache
- `/logs` → training outputs

## Code Organization Patterns

### Tokenization Processor Pattern
All tokenization processors inherit from `TokenizerProcessor` base class:
```python
class MorphologyAwarePatokProcessor:
    def __init__(self, tokenizer, prefix_file, infix_file, suffix_file,
                 expand_prop=0.1, contract_prop=0.9, affix_awareness=0.95):
        self.tokenizer = tokenizer
        self.affixes = self._build_affix_list()
        self.affix_finder = self._build_affix_finder(self.affixes)  # Aho-Corasick automaton

    def contract_expand(self, token_ids, contract_prop, expand_prop):
        # Contract: merge tokens respecting morpheme boundaries
        # Expand: split tokens at morpheme boundaries
```

Processors are applied DURING preprocessing, not during training. They use **Aho-Corasick automaton** for efficient affix detection.

### Benchmark Generator Pattern
Benchmark generators follow consistent structure:
```python
def generate_<benchmark_name>_benchmark(num_samples=1000, seed=42):
    """Generate benchmark tasks in standardized format."""
    tasks = []
    for i in range(num_samples):
        task = {
            "id": f"{benchmark_name}_{i}",
            "question": "...",
            "answer": "...",
            "options": ["A", "B", "C", "D"],  # MCQ only
            "category": "...",
            "difficulty": "easy|medium|hard"
        }
        tasks.append(task)
    return tasks
```

All benchmarks saved as JSONL in `data/benchmarks/`.

### Evaluation Loader Pattern
Benchmark loaders registered in `src/evaluation/loaders/registry.py`:
```python
from functools import partial
from evaluation.loaders.pacute import load_pacute

EVALS_DICT = {
    "pacute": partial(load_pacute, split="test"),
    "pacute-mcq": partial(load_pacute, split="test"),
    "pacute-gen": partial(load_pacute, split="test", format="gen"),
    "pacute-affixation": partial(load_pacute, split="test", categories=["affixation"]),
    # ... more variants
}

def load_benchmark(benchmark_name):
    return EVALS_DICT[benchmark_name]()
```

This allows `run_evaluation.py` to dynamically load benchmarks via `load_benchmark(name)`.

## Important Development Constraints

### Data Format Requirements
- **Training input**: JSONL with `{"text": "..."}` format (one document per line)
- **Preprocessing output**: Megatron binary (`.bin` + `.idx` pairs)
- **Training data paths**: Must be prefixes WITHOUT `_text_document` suffix
- **Benchmark format**: JSONL with standardized task schema

### Container Workflow Requirements
- **NEVER** run preprocessing on login node (needs container)
- **ALWAYS** source `.env` before submitting PBS jobs or running container scripts
- **NEVER** commit generated helper scripts (`run_in_docker.sh`, `run_in_enroot.sh`, `run_in_singularity.sh`)
- **ALWAYS** use PBS jobs for preprocessing and training on HPC (not interactive)
- **Docker** is preferred for cloud/local development
- **Enroot** is preferred for HPC clusters

### Security Requirements
- **NEVER** commit `.env` file (use `.env.example` for templates)
- **NEVER** hardcode cluster paths in version-controlled files
- **NEVER** commit API keys, tokens, or secrets
- **ALWAYS** use templates for PBS jobs with placeholder variables

### Testing Requirements
- Test preprocessing with single chunk before full 20-chunk processing
- Test training with `run_cpt_test.pbs` (10 steps) before full runs
- Test evaluation with `--max-samples 100` before full benchmarks

## File Structure Patterns

```
filipino-morphology-llm/
├── src/                          # Source code (importable package)
│   ├── tokenization/            # Tokenization processors
│   │   ├── base_processor.py
│   │   ├── stochastok_processor.py
│   │   ├── patok_morphology.py  # RECOMMENDED
│   │   ├── patok_processor.py   # DEPRECATED
│   │   └── affix_decomposition.py
│   ├── evaluation/              # Evaluation framework
│   │   ├── loaders/             # Benchmark loaders (registry pattern)
│   │   │   ├── registry.py      # EVALS_DICT mapping
│   │   │   ├── pacute.py
│   │   │   ├── hierarchical.py
│   │   │   └── ...
│   │   ├── datasets/generators/ # Task generators
│   │   └── evaluators/          # Evaluation logic
│   └── analysis/                # Analysis tools
├── training/                     # Training implementations
│   ├── nemo/                    # NeMo Framework CPT (ACTIVE)
│   │   ├── run_cpt.py           # Main training script
│   │   ├── data/                # Data preprocessing
│   │   │   ├── preprocess_data.py
│   │   │   └── split_jsonl.py
│   │   └── setup/               # Container setup
│   └── stochastok/              # DEPRECATED - Legacy GPT-2 training
├── scripts/                      # Executable utilities
│   ├── generate_benchmarks.py   # Generate evaluation datasets
│   ├── run_evaluation.py        # Run model evaluation
│   ├── run_full_evaluation.sh   # Comprehensive evaluation
│   ├── analyze_inference_results.py  # Analyze outputs
│   ├── create_visualizations.py # Generate plots
│   ├── convert_hf_to_nemo.py    # Convert HF → NeMo format
│   ├── preprocess_all_vanilla.sh    # Batch preprocess vanilla
│   ├── preprocess_all_stochastok.sh # Batch preprocess stochastok
│   └── preprocess_all_patok.sh      # Batch preprocess patok
├── job_templates/               # PBS job templates (version-controlled)
├── jobs/                        # Generated PBS jobs (gitignored)
├── data/                        # Data files
│   ├── benchmarks/             # Evaluation tasks (JSONL)
│   ├── chunks/                 # Split training data
│   ├── processed/              # Megatron binary format
│   │   ├── vanilla/            # Baseline tokenization
│   │   ├── stochastok/         # Stochastic expansion
│   │   └── patok/              # Morphology-aware
│   └── affixes/                # Filipino affix lists
└── docs/                        # Documentation
```

## Common Development Workflows

### Adding a New Tokenization Method
1. Create processor in `src/tokenization/<name>_processor.py` (inherit from base if needed)
2. Implement token expansion/contraction methods
3. Add tokenization mode to `preprocess_data.py` argument parser
4. Add conditional logic in `preprocess_data.py` to instantiate processor
5. Create batch preprocessing script `scripts/preprocess_all_<name>.sh`
6. Test with single chunk preprocessing
7. Generate full preprocessed dataset
8. Train model with new tokenization
9. Evaluate and compare results

### Adding a New Benchmark
1. Create generator in `src/evaluation/datasets/generators/<name>.py`
2. Implement `generate_<name>_benchmark()` returning list of standardized tasks
3. Add loader in `src/evaluation/loaders/<name>.py`
4. Register in `src/evaluation/loaders/registry.py` EVALS_DICT
5. Add to `scripts/generate_benchmarks.py` generation workflow
6. Test generation: `python scripts/generate_benchmarks.py --benchmarks <name>`
7. Add to evaluation: modify `scripts/run_evaluation.py` or use existing framework
8. Run evaluation: `python scripts/run_evaluation.py --benchmarks <name>`

### Debugging Training Issues
1. Check container exists: `enroot list | grep nemo_framework` or `docker images | grep nemo`
2. Verify preprocessed data exists: `ls -lh data/processed/*.bin`
3. Test interactively: `./run_in_docker.sh python training/nemo/run_cpt.py --max-steps 10`
4. Check training logs: `tail -f nemo_experiments/<name>/nemo_log_*.txt`
5. Monitor W&B dashboard for metrics
6. For distributed training: ensure HF checkpoint converted to NeMo format first

### Comparing Tokenization Methods
1. Preprocess same data with different modes (vanilla vs stochastok vs patok)
2. Train models with same hyperparameters on each tokenized dataset
3. Evaluate all models on same benchmarks
4. Analyze results: `python scripts/analyze_inference_results.py`
5. Create visualizations: `python scripts/create_visualizations.py`
6. Focus comparison on:
   - PACUTE Affixation (morpheme understanding)
   - Hierarchical Level 2 (morpheme decomposition bottleneck)
   - PACUTE Manipulation (character operations)
7. Expected improvements: Patok > Stochastok > Baseline

### Full Experiment Workflow
```bash
# 1. Preprocess data (all three tokenizations)
bash scripts/preprocess_all_vanilla.sh
bash scripts/preprocess_all_stochastok.sh
bash scripts/preprocess_all_patok.sh

# 2. Train models (submit PBS jobs or run locally)
./run_in_docker.sh python training/nemo/run_cpt.py \
    --data-path /workspace/data/processed/vanilla/chunk_001 \
    --max-steps 5000

# 3. Run full evaluation
bash scripts/run_full_evaluation.sh

# 4. Analyze results
python scripts/analyze_inference_results.py
python scripts/create_visualizations.py
```

## Key Documentation References

- `README.md` - Project overview, quick start, baseline results, visualizations
- `SETUP.md` - Complete environment setup and container installation
- `docs/RESEARCH.md` - Research question, tokenization methods, experimental design
- `docs/TRAINING.md` - Training workflows, PBS jobs, monitoring
- `docs/EVALUATION.md` - Benchmark generation, evaluation procedures, analysis
- `docs/GEMMA3_MONKEY_PATCH.md` - Known Gemma3 bugs and workarounds
- `docs/BENCHMARK_FORMATS.md` - Benchmark format specifications (MCQ vs GEN)
- `training/nemo/data/DATA_PREPROCESSING.md` - Data preprocessing guide
- `job_templates/README.md` - PBS job template system
- `docs/SECURITY.md` - Security best practices

## Model-Specific Notes

### Gemma2 2B (Current Model)
- Used for continued pretraining experiments
- Requires conversion to NeMo format for distributed training: `scripts/convert_hf_to_nemo.py`
- Config via `llm.GemmaConfig2()` in NeMo
- Pre-tokenized with SentencePiece tokenizer

### Gemma3 Issues (If Using)
- `Gemma3SelfAttention.forward()` signature mismatch may require monkey patch
- Gemma3 uses custom local/global RoPE, not standard rotary embeddings
- See `docs/GEMMA3_MONKEY_PATCH.md` for full details and workarounds
- NeMo 2.1+ has better Gemma3 support

### NeMo 2.0+ API
- Use `nemo.collections.llm` not `nemo.collections.nlp`
- Model config via `llm.GemmaConfig2()` not YAML files
- Trainer via `nl.Trainer()` with Lightning 2.0 API
- Data module must be `PreTrainingDataModule` with Megatron format

### Preprocessing Requirements
- Must specify `--tokenizer-type HuggingFaceTokenizer` with `--tokenizer-model`
- Text key must match JSONL structure (default: "text")
- Output prefix must NOT include `_text_document` suffix
- Workers should be 64 for optimal parallel tokenization

## Container Runtime Selection Guide

| Environment | Recommended Runtime | Why |
|-------------|-------------------|-----|
| Cloud (AWS/GCP/Azure) | Docker | Most portable, standard |
| Local workstation | Docker | Easy setup, widely supported |
| HPC with Enroot | Enroot | Fast, no root needed, efficient |
| HPC without Enroot | Singularity | Standard HPC container runtime |

All three runtimes use the same NeMo container image but with different wrapper scripts.
