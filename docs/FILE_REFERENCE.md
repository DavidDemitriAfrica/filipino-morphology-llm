# Quick Reference: Key Files and Their Purpose

This document provides a quick lookup of important files and their roles in the project.

---

## 📖 Documentation (Start Here!)

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start guide |
| `docs/RESEARCH_OVERVIEW.md` | 📌 **COMPLETE research design** - read this first! |
| `docs/EXPERIMENTAL_FLOW.md` | Pipeline visualization and workflow |
| `docs/SETUP.md` | Environment setup and data preprocessing |
| `docs/USAGE.md` | Training workflow and PBS jobs |
| `docs/AFFIX_PROCESSING.md` | Patok implementation details |
| `docs/HIERARCHICAL_TASKS.md` | Hierarchical benchmark design |
| `scripts/EVALUATION_README.md` | 📌 **Benchmark generation and evaluation guide** |

---

## 🔬 Tokenization Methods

| File | Tokenization Approach | Description |
|------|----------------------|-------------|
| N/A (default) | **Baseline** | Standard BPE (GPT-2/Gemma tokenizer as-is) |
| `src/tokenization/stochastok_processor.py` | **StochasTok** | Stochastic token expansion (~10%) |
| `src/tokenization/patok_processor.py` | **Patok** | Affix-aware expand-contract (30%+30%, affix_pref=0.7) |

### Usage Examples
```python
# StochasTok
from src.tokenization.stochastok_processor import StochastokProcessor
processor = StochastokProcessor(tokenizer, expand_prop=0.1)
expanded_ids = processor.expand(token_ids)

# Patok
from src.tokenization.patok_processor import PatokProcessor
processor = PatokProcessor(tokenizer, expand_prop=0.3, contract_prop=0.3, affix_preference=0.7)
processed_ids = processor.expand_contract(token_ids)
```

---

## 🏃 Training Systems

### Small-Scale Training (117M params)
| File | Purpose |
|------|---------|
| `training/stochastok/models/` | GPT-2 architecture implementation |
| `training/stochastok/training/` | Custom training loop |
| `training/stochastok/experiments/train.py` | Training entrypoint |
| `training/stochastok/experiments/eval.py` | Evaluation entrypoint |
| `training/stochastok/data_processing/` | Data preprocessing for memmap format |

### Large-Scale Training (1B params) - **Current Focus**
| File | Purpose |
|------|---------|
| `training/nemo/run_cpt.py` | 📌 **Main training script** (NeMo CPT) |
| `training/nemo/setup/setup_enroot.sh` | Container setup script |
| `training/nemo/setup/setup_env.sh` | Conda environment setup |
| `training/nemo/data/preprocess_data.py` | Convert JSONL → Megatron binary |
| `training/nemo/data/split_jsonl.py` | Split large JSONL into chunks |

---

## 📊 Evaluation Benchmarks

### Benchmark Generation Scripts (One-Stop)
| File | Purpose |
|------|---------|
| `scripts/generate_all_benchmarks.py` | 📌 **Generate ALL benchmarks** (master script) |
| `src/evaluation/datasets/scripts/generate_pacute_benchmarks.py` | Generate PACUTE benchmarks |
| `src/evaluation/datasets/scripts/generate_hierarchical_benchmark.py` | Generate hierarchical tasks |
| `src/evaluation/datasets/scripts/generate_langgame_benchmark.py` | Generate LangGame dataset |
| `src/evaluation/datasets/scripts/generate_math_benchmark.py` | Generate multi-digit addition |

### PACUTE (5,845 tasks)
| File | Task Type | Count (MCQ + Gen) |
|------|-----------|-------------------|
| `src/evaluation/datasets/generators/affixation.py` | Prefix/infix/suffix identification | 140 + 140 |
| `src/evaluation/datasets/generators/composition.py` | Character counting, word formation | 2,505 + 1,400 |
| `src/evaluation/datasets/generators/manipulation.py` | Insert/delete/swap operations | 2,560 + 2,560 |
| `src/evaluation/datasets/generators/syllabification.py` | Syllable counting/extraction | 640 + 640 |

**Data**: `data/benchmarks/{name}_{mcq|gen}.jsonl`

### Hierarchical Benchmark (1,198 tasks)
| File | Purpose |
|------|---------|
| `src/evaluation/datasets/generators/hierarchical.py` | Task generator (6 levels, 0-5) |
| `src/evaluation/evaluators/hierarchical.py` | Evaluation and diagnostic analysis |
| `scripts/demo_hierarchical_tasks.py` | Demo and usage examples |

**Data**: `data/benchmarks/hierarchical_{mcq|gen}.jsonl`

**Levels**:
- Level 0: Character Recognition
- Level 1: Character Manipulation  
- Level 2: Morpheme Decomposition
- Level 3: Morpheme Manipulation
- Level 4: Morpheme Composition
- Level 5: Complex Morphological Reasoning

### LangGame (3,000 tasks)
| File | Purpose |
|------|---------|
| `src/evaluation/datasets/scripts/generate_langgame_benchmark.py` | 📌 **Generate dataset** |

**Data**: `data/benchmarks/langgame_{train|val}.jsonl` (2000 train, 1000 val)

**Tasks**: Contains, starts with, ends with, longest, shortest, most of letter

### Multi-Digit Addition (3,000 tasks)
| File | Purpose |
|------|---------|
| `src/evaluation/datasets/scripts/generate_math_benchmark.py` | 📌 **Generate dataset** |

**Data**: `data/benchmarks/multi_digit_addition_{train|val}.jsonl` (2000 train, 1000 val)

**Format**: 3-digit addition problems

### Morphological Analysis
| File | Purpose |
|------|---------|
| `src/analysis/morphological_metrics.py` | 📌 **MorphScore, Boundary F1, Fragmentation** |
| `src/analysis/information_theory.py` | Entropy and compression analysis |
| `src/analysis/tokenization/` | Tokenization comparison tools |
| `src/analysis/affixes/` | Affix coverage analysis |
| `src/analysis/datasets/` | Dataset comparison tools |
| `scripts/run_analysis.py` | 📌 **Unified analysis runner** |

**Data**: `data/corpora/affix_annotations.jsonl` (472 annotated words)

---

## 🔧 Utility Scripts

### Core Scripts
| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/download_seapile.py` | Download seapile-v2 dataset | Before preprocessing |
| `scripts/verify_setup.py` | Installation check | After environment setup |
| `scripts/create_affix_annotations.py` | Generate morpheme annotations | Already done (472 words) |

### Evaluation Scripts
| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/generate_all_benchmarks.py` | 📌 **Generate ALL benchmarks** | Creates all evaluation datasets |
| `scripts/run_benchmark_evaluation.py` | 📌 **Evaluate models on benchmarks** | Test model performance |
| `scripts/demo_hierarchical_tasks.py` | Demo hierarchical framework | See examples of hierarchical tasks |
| `scripts/evaluate_downstream.py` | Downstream task evaluation | Evaluate on specific benchmarks |

### Analysis Scripts
| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/run_analysis.py` | 📌 **Unified analysis runner** | Run all analysis tools |

**Analysis modules** are in `src/analysis/`:
- `tokenization/`: Tokenization comparison (simple, comprehensive, compare)
- `affixes/`: Affix coverage analysis
- `datasets/`: Dataset comparison tools

---

## 📦 Data Files

### Training Data
| Path | Description | Size |
|------|-------------|------|
| `data/corpora/seapile-v2.jsonl` | Raw training corpus | 7.4GB |
| `data/chunks/chunk_*.jsonl` | Split corpus (20 chunks) | ~370MB each |
| `data/processed/*.bin` + `*.idx` | Megatron binary format | For NeMo |

### Evaluation Data
| Path | Description | Count |
|------|-------------|-------|
| `data/benchmarks/affixation_{mcq\|gen}.jsonl` | Affix tasks | 140 + 140 |
| `data/benchmarks/composition_{mcq\|gen}.jsonl` | Composition tasks | 2,505 + 1,400 |
| `data/benchmarks/manipulation_{mcq\|gen}.jsonl` | Manipulation tasks | 2,560 + 2,560 |
| `data/benchmarks/syllabification_{mcq\|gen}.jsonl` | Syllabification tasks | 640 + 640 |
| `data/benchmarks/hierarchical_{mcq\|gen}.jsonl` | Hierarchical tasks (6 levels) | 598 + 600 |
| `data/benchmarks/langgame_{train\|val}.jsonl` | LangGame tasks | 2,000 + 1,000 |
| `data/benchmarks/multi_digit_addition_{train\|val}.jsonl` | Math tasks | 2,000 + 1,000 |
| `data/benchmarks/stress_{mcq\|gen}.jsonl` | Stress pattern tasks | Available |

### Linguistic Resources
| Path | Description | Count |
|------|-------------|-------|
| `data/affixes/filipino_affixes.txt` | Filipino affix list | 92 affixes |
| `data/corpora/affix_annotations.jsonl` | Morpheme-annotated words | 472 words |
| `data/corpora/pacute_data/inflections.xlsx` | Inflection pairs | 80 pairs |
| `data/corpora/pacute_data/syllables.jsonl` | Syllabified words | 16,828 words |
| `data/word_frequencies.csv` | Word frequencies | 118,801 entries |

---

## 🚀 PBS Job Scripts

| Job Script | Purpose | Resources | Time |
|------------|---------|-----------|------|
| `jobs/preprocess_test_chunk1.pbs` | Test preprocessing (1 chunk) | 1 GPU | ~15 min |
| `jobs/preprocess_data.pbs` | Preprocess full dataset | 1 GPU | ~5 hours |
| `jobs/preprocess_data_parallel.pbs` | 📌 **Parallel preprocessing (20 chunks)** | 20×1 GPU | ~15 min |
| `jobs/run_cpt_test.pbs` | Test training (10 steps) | 1 GPU | ~5 min |
| `jobs/run_cpt.pbs` | 📌 **Full training (10K steps)** | 8 GPUs | ~3 days |

**Logs**: `/scratch_aisg/SPEC-SF-AISG/railey/logs/*.OU`

---

## 🔄 Workflow: File Dependencies

### Preprocessing Pipeline
```
data/corpora/seapile-v2.jsonl
    ↓ [scripts/download_seapile.py]
    ↓ [training/nemo/data/split_jsonl.py]
data/chunks/chunk_*.jsonl
    ↓ [training/nemo/data/preprocess_data.py]
    ↓ [jobs/preprocess_data_parallel.pbs]
data/processed/*.bin + *.idx
```

### Training Pipeline
```
data/processed/*.bin + *.idx
    ↓ [training/nemo/run_cpt.py]
    ↓ [jobs/run_cpt.pbs]
nemo_experiments/*/checkpoints/step_*.ckpt
```

### Evaluation Pipeline (To Be Implemented)
```
nemo_experiments/*/checkpoints/step_*.ckpt
    ↓ [scripts/evaluate_model.py]  ← TO BE CREATED
    ↓ [Uses: src/evaluation/*, src/analysis/*]
results/*/*.json
    ↓ [scripts/compare_results.py]  ← TO BE CREATED
    ↓ [scripts/generate_paper_tables.py]  ← TO BE CREATED
Comparison tables, figures, LaTeX
```

---

## 💡 Common Tasks: Quick Commands

### 1. Download Training Data
```bash
python scripts/download_seapile.py
```

### 2. Preprocess Data (Recommended: Parallel)
```bash
# Edit jobs/preprocess_data_parallel.pbs: #PBS -J 1-20
qsub jobs/preprocess_data_parallel.pbs
```

### 3. Test Training
```bash
qsub jobs/run_cpt_test.pbs
```

### 4. Full Training
```bash
qsub jobs/run_cpt.pbs
```

### 5. Check Job Status
```bash
qstat
qstat -u $USER
```

### 6. Monitor Logs
```bash
tail -f /scratch_aisg/SPEC-SF-AISG/railey/logs/<JOB_ID>.OU
```

### 7. Generate Hierarchical Benchmark
```bash
python scripts/generate_hierarchical_benchmark.py
```

### 8. Analyze Tokenizer Morphological Alignment
```bash
python scripts/analyze_tokenization_simple.py
```

### 9. Compare Tokenizers
```bash
python scripts/compare_tokenizers.py
```

---

## 🎯 Files to Create Next

These files need to be implemented for the complete pipeline:

### Preprocessing
- `training/nemo/data/preprocess_data_stochastok.py` - StochasTok preprocessing
- `training/nemo/data/preprocess_data_patok.py` - Patok preprocessing

### Training
- `jobs/run_cpt_stochastok.pbs` - PBS script for StochasTok training
- `jobs/run_cpt_patok.pbs` - PBS script for Patok training

### Evaluation
- `scripts/evaluate_model.py` - Run all benchmarks on a checkpoint
- `scripts/evaluate_pacute.py` - Evaluate on PACUTE
- `scripts/evaluate_hierarchical.py` - Evaluate on hierarchical tasks
- `scripts/evaluate_langgame.py` - Evaluate on LangGame
- `scripts/evaluate_math.py` - Evaluate on multi-digit addition
- `scripts/evaluate_morphological.py` - Compute morphological metrics

### Analysis
- `scripts/compare_results.py` - Statistical comparison of 3 models
- `scripts/generate_paper_tables.py` - LaTeX tables for paper
- `scripts/generate_paper_figures.py` - Plots and visualizations
- `scripts/ablation_studies.py` - Hyperparameter ablations

---

## 📞 Where to Look For...

**Research context**: `docs/RESEARCH_OVERVIEW.md`  
**Experimental design**: `docs/EXPERIMENTAL_FLOW.md`  
**Setup help**: `docs/SETUP.md`  
**Training help**: `docs/USAGE.md`  
**Patok details**: `docs/AFFIX_PROCESSING.md`  
**Benchmark details**: `docs/HIERARCHICAL_TASKS.md`  

**Tokenization code**: `src/tokenization/`  
**Evaluation code**: `src/evaluation/`  
**Analysis code**: `src/analysis/`  
**Training code (small)**: `training/stochastok/`  
**Training code (large)**: `training/nemo/`  

**PBS jobs**: `jobs/*.pbs`  
**Utility scripts**: `scripts/*.py`  
**Raw data**: `data/corpora/`  
**Preprocessed data**: `data/processed/`  
**Benchmarks**: `data/benchmarks/`  
**Linguistic resources**: `data/affixes/`, `data/corpora/affix_annotations.jsonl`  

**Logs**: `/scratch_aisg/SPEC-SF-AISG/railey/logs/`  
**Checkpoints**: `nemo_experiments/*/checkpoints/`  
**Results**: `results/*/` (to be created)  

---

## 🔍 Search Patterns

Looking for specific functionality? Use these grep patterns:

```bash
# Find tokenization usage
grep -r "StochastokProcessor\|PatokProcessor" src/ training/

# Find evaluation code
grep -r "mcq_evaluator\|hierarchical" src/evaluation/

# Find preprocessing scripts
find . -name "*preprocess*.py"

# Find PBS job files
find jobs/ -name "*.pbs"

# Find benchmark datasets
find data/benchmarks/ -name "*.jsonl"

# Find documentation
find docs/ -name "*.md"
```

---

## ✅ Checklist: Am I Ready to Train?

- [ ] Container setup complete (`enroot list | grep nemo_framework`)
- [ ] Environment variables configured (`.env` file exists)
- [ ] Training data downloaded (`data/corpora/seapile-v2.jsonl` exists)
- [ ] Data preprocessed (`data/processed/*.bin` and `*.idx` exist)
- [ ] Test job succeeds (`qsub jobs/run_cpt_test.pbs`)
- [ ] WandB configured (`WANDB_API_KEY` in `.env`)
- [ ] HuggingFace token set (`HF_TOKEN` in `.env`)

Once all checked, you're ready: `qsub jobs/run_cpt.pbs` 🚀

---

## 🆘 Troubleshooting

| Problem | Check This File |
|---------|----------------|
| Setup issues | `docs/SETUP.md` |
| Training fails | `docs/USAGE.md`, logs in `/scratch_aisg/.../logs/` |
| Preprocessing errors | `training/nemo/data/preprocess_data.py` |
| PBS job issues | `jobs/QUICK_REFERENCE_PBS.sh` |
| Missing dependencies | `requirements.txt`, `.env.example` |
| Container issues | `training/nemo/setup/setup_enroot.sh` |

---

Last updated: December 7, 2025
