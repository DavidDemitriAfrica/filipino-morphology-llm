# Original Source Attribution

This repository is built upon two existing repositories with proper attribution.

## Source Repositories

### StochasTok
- **Original**: https://github.com/anyasims/stochastok
- **Fork**: https://github.com/raileymontalan/stochastok
- **License**: MIT License
- **Paper**: Sims et al. (2025). "Stochastic Tokenization Improves Subword Understanding"

### PACUTE
- **Repository**: (original pacute repository)
- **License**: CC0 1.0 Universal (Public Domain)
- **Description**: Philippine Annotated Corpus for Understanding Tagalog Entities

## Components from StochasTok

### Core Tokenization
- `src/tokenization/patok_processor.py` - Affix-aware expand-contract tokenization
- `src/tokenization/stochastok_processor.py` - Original stochastic tokenization
- `docs/AFFIX_PROCESSING_README.md` - Patok documentation

### Model Architecture
- `src/models/` - Complete transformer implementation
  - `build_models.py` - Model builder
  - `model_shell.py` - Model wrapper
  - `core_models.py` - Transformer layers
  - `embedding_models.py` - Embeddings
  - `model_heads.py` - Language modeling head
  - `generator.py` - Text generation
  - `components/` - Attention, feedforward, normalization, positional encoding
  - `weight_initialization.py` - Weight initialization
  - `utils.py` - Model utilities

### Training Infrastructure
- `src/training/` - Complete training framework
  - `base_trainer.py` - Main trainer
  - `build_trainers.py` - DDP support
  - `dataset_interface_pretraining.py` - Pretraining data
  - `dataset_interface_instruction.py` - Instruction tuning data
  - `evaluator.py` - Training evaluation
  - `optimizer.py` - Optimizer configuration
  - `scheduler.py` - Learning rate scheduling
  - `loss_fn.py` - Loss functions
  - `utils.py` - Training utilities

### Data Processing
- `src/data_processing/` - Dataset preprocessing
  - `tokenize_dataset.py` - Initial tokenization
  - `stochastok_expand_dataset.py` - Apply StochasTok
  - `patok_expand_contract_dataset.py` - Apply Patok
  - `make_langgame_dataset.py` - LangGame dataset
  - `make_multi_digit_addition_dataset.py` - Math dataset
  - `utils.py` - Processing utilities

### Evaluation Framework
- `src/evaluation/benchmarks/` - Evaluation harness
  - `evaluator_interface.py` - Evaluator interface
  - `eval_wrapper.py` - Evaluation wrapper
  - `load_evaluators.py` - Evaluator loader
  - `generation_evaluator_math.py` - Math generation eval
  - `metrics.py` - Evaluation metrics
  - `mcqs/` - Multiple choice evaluations
    - `mcq_evaluator.py` - MCQ evaluator
    - `load_benchmarks.py` - Benchmark loader
    - `benchmarks/` - Individual benchmarks
      - `arc.py` - ARC benchmark
      - `blimp.py` - BLiMP benchmark
      - `hellaswag.py` - HellaSwag benchmark
      - `mmlu.py` - MMLU benchmark
      - `winogrande.py` - Winogrande benchmark
      - `langgame.py` - LangGame benchmark

### Experiments
- `experiments/train.py` - Training script
- `experiments/eval.py` - Evaluation script

### Configuration
- `configs/pretraining.yaml` - Pretraining configuration
- `configs/instruction_tuning.yaml` - Instruction tuning configuration

### Data
- `data/affixes/filipino_affixes.txt` - 93 Filipino affixes
- `data/corpora/top_1k_words` - Top 1000 words

### Tests
- `tests/test_affix_processor.py` - Patok tests

### Utilities
- `scripts/analyze_dataset_differences.py` - Dataset comparison tool

## Components from PACUTE

### Task Generation
- `src/evaluation/affixation.py` - Affixation tasks (prefix, suffix, infix, circumfix)
- `src/evaluation/composition.py` - Composition tasks (spelling, counting)
- `src/evaluation/manipulation.py` - Manipulation tasks (insertion, deletion, etc.)
- `src/evaluation/syllabification.py` - Syllabification tasks

### Core Operations
- `src/evaluation/string_operations.py` - String manipulation primitives
- `src/evaluation/syllabification_operations.py` - Filipino syllabification
- `src/evaluation/sampling.py` - Frequency-aware sampling
- `src/evaluation/constants.py` - Shared constants
- `src/evaluation/utils.py` - Utility functions

### Data
- `data/benchmarks/` - 1,040 PACUTE evaluation items
  - `mcq_affixation.jsonl` (140 items)
  - `mcq_composition.jsonl` (180 items)
  - `mcq_manipulation.jsonl` (160 items)
  - `mcq_syllabification.jsonl` (80 items)
  - `gen_affixation.jsonl` (140 items)
  - `gen_composition.jsonl` (100 items)
  - `gen_manipulation.jsonl` (160 items)
  - `gen_syllabification.jsonl` (80 items)

- `data/corpora/pacute_data/` - Source data
  - `syllables.jsonl` - 16,828 syllabified words
  - `word_frequencies.csv` - 2M+ word frequencies
  - `inflections.xlsx` - Affix examples
  - Dataset files (*.jsonl) - Pre-generated datasets

### Tests
- `tests/test_affixation.py` - Affixation tests
- `tests/test_composition.py` - Composition tests
- `tests/test_manipulation.py` - Manipulation tests
- `tests/test_syllabification.py` - Syllabification tests
- `tests/test_sampling.py` - Sampling tests
- `tests/test_word_length_filtering.py` - Filtering tests

### Documentation
- `docs/DEVELOPER_GUIDE.md` - Developer guide for PACUTE

### Notebooks
- `notebooks/create_affixation.ipynb` - Affixation dataset creation
- `notebooks/create_composition_string_manipulation_syllabification.ipynb` - Other tasks
- `notebooks/create_syllabification.ipynb` - Syllabification exploration
- `notebooks/diksiyonaryo.ipynb` - Dictionary exploration

## New Contributions

### Tokenization
- `src/tokenization/affix_decomposition.py` - Algorithm for OOV affix handling

### Evaluation Framework
- `src/evaluation/hierarchical_tasks.py` - 6-level diagnostic framework
- `src/evaluation/hierarchical_analysis.py` - Result analysis and diagnostics
- `docs/HIERARCHICAL_TASKS.md` - Framework documentation

### Morphological Metrics
- `src/analysis/morphological_metrics.py` - MorphScore, boundary F1, fragmentation, etc.
- `src/analysis/information_theory.py` - Mutual information, entropy analysis

### Utilities
- `scripts/verify_setup.py` - Installation verification
- `scripts/analyze_affix_coverage.py` - Tokenizer coverage analysis
- `scripts/demo_hierarchical_tasks.py` - Framework demonstration

### Documentation
- `README.md` - Main documentation with research positioning
- `QUICKSTART.md` - Quick start guide
- `MIGRATION_GUIDE.md` - Migration guide from separate repos
- `IMPLEMENTATION_SUMMARY.md` - Implementation overview
- `LICENSE` - Combined license file
- `ORIGINAL_SOURCES.md` - This file

## Modifications

### Integration Changes
- Reorganized file structure into monorepo layout
- Updated import paths throughout codebase
- Created unified `setup.py` for package installation
- Merged requirements files
- Created combined `.gitignore`

### Documentation Updates
- Removed LLM-ish language patterns
- Added research motivation and positioning
- Made writing more academic and terse
- Updated examples for new structure

## Verification

All original functionality has been preserved:
- ✅ All Python files from both repositories
- ✅ All configuration files
- ✅ All data files
- ✅ All tests
- ✅ All notebooks
- ✅ All documentation

## Acknowledgments

This work builds directly on:
- StochasTok by Sims et al. (2025)
- PACUTE evaluation framework
- CUTE benchmark by Edman et al. (2024)

All original licenses are respected and properly attributed.
