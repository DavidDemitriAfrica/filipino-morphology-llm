# Filipino Morphology-Aware Language Modeling

A unified framework for training and evaluating morphologically-aware tokenization methods for Filipino and other morphologically-rich languages.

## Overview

This repository combines:
1. **Patok**: An affix-aware stochastic tokenization method that extends StochasTok
2. **PACUTE**: Philippine Annotated Corpus for Understanding Tagalog Entities - a benchmark for evaluating subword understanding in Filipino

## Key Features

- **Morphologically-Aware Tokenization**: Patok processor that preferentially forms Filipino affixes during tokenization
- **Hierarchical Evaluation Framework**: Multi-level tasks that diagnose specific morphological capabilities
- **Information-Theoretic Analysis**: Quantify morphological information in tokenizations
- **Multi-Tokenizer Support**: Works with GPT-2, Llama, Gemma, and other BPE tokenizers
- **End-to-End Pipeline**: From data preprocessing through training to evaluation

## Repository Structure

```
filipino-morphology-llm/
├── src/
│   ├── tokenization/          # Patok & StochasTok processors
│   ├── models/                # Transformer architecture
│   ├── training/              # Training infrastructure (DDP support)
│   ├── evaluation/            # PACUTE benchmark + metrics
│   ├── data_processing/       # Dataset preprocessing pipelines
│   └── analysis/              # Information-theoretic analysis tools
├── data/
│   ├── affixes/               # Filipino affixes & decomposition rules
│   ├── benchmarks/            # PACUTE evaluation tasks (1,040 items)
│   ├── corpora/               # Training data
│   └── vocabularies/          # Tokenizer vocabulary analyses
├── configs/                   # Hydra configuration files
├── experiments/               # Training & evaluation scripts
├── notebooks/                 # Analysis notebooks
├── tests/                     # Unit tests
└── scripts/                   # Utility scripts
```

## Installation

```bash
# Clone repository
git clone [url]
cd filipino-morphology-llm

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Tokenize Dataset with Patok

```bash
python src/data_processing/patok_expand_contract_dataset.py \
    --input data/corpora/your_corpus \
    --output data/processed/patok_tokenized \
    --expand_prop 0.3 \
    --contract_prop 0.3 \
    --affix_preference 0.7
```

### 2. Train Model

```bash
python experiments/train.py \
    --config-name pretraining \
    trainer.dataset.name=patok_tokenized
```

### 3. Evaluate on PACUTE

```bash
python experiments/eval.py \
    --checkpoint checkpoints/model.pt \
    --benchmark pacute \
    --output results/
```

## Patok: Affix-Aware Tokenization

Patok extends StochasTok with morphological awareness:

1. **Expand Phase**: Stochastically split tokens (avoiding affix boundaries)
2. **Contract Phase**: Merge adjacent tokens with preference for forming Filipino affixes
3. **Affix Dictionary**: Uses 93 Filipino affixes with configurable preference weights

```python
from src.tokenization import PatokProcessor

processor = PatokProcessor(
    base_tokenizer="gpt2",
    affixes_file="data/affixes/filipino_affixes.txt",
    expand_prop=0.3,      # Probability of expanding a token
    contract_prop=0.3,    # Probability of contracting adjacent tokens
    affix_preference=0.7, # Preference weight for affix-forming contractions
    num_iterations=3      # Number of expand-contract cycles
)

tokens = processor.process(text)
```

## PACUTE Benchmark

### Hierarchical Task Structure

PACUTE organizes tasks into levels that diagnose specific capabilities:

- **Level 0: Character Recognition** - Basic character-level operations
- **Level 1: Character Manipulation** - String operations without morphology
- **Level 2: Morpheme Decomposition** - Identifying morphological boundaries
- **Level 3: Morpheme Manipulation** - Transforming morphological units
- **Level 4: Morpheme Composition** - Building words from morphemes
- **Level 5: Complex Morphological Reasoning** - Multi-step linguistic operations

### Task Categories

1. **Affixation** (280 items): Prefix, suffix, infix, circumfix operations
2. **Composition** (280 items): Spelling, character counting, length analysis
3. **Manipulation** (320 items): Insertion, deletion, substitution, permutation
4. **Syllabification** (160 items): Stress classification, syllable counting

```python
from src.evaluation import evaluate_pacute, generate_capability_profile

# Evaluate model on all PACUTE tasks
results = evaluate_pacute(model, tokenizer, benchmark_path="data/benchmarks/")

# Generate hierarchical capability profile
profile = generate_capability_profile(results)
# Output: Performance at each level (0-5) for each task category
```

## Information-Theoretic Analysis

```python
from src.analysis import (
    morpheme_token_mutual_information,
    morphological_perplexity,
    affix_consistency_entropy,
    compositionality_score
)

# Measure morphological information in tokenization
mi_score = morpheme_token_mutual_information(tokenizer, morpheme_annotations)

# Compare perplexity on morphologically complex words
perplexity = morphological_perplexity(model, word_list, complexity_scores)

# Measure affix tokenization consistency
consistency = affix_consistency_entropy(tokenizer, affixed_words)

# Test compositional understanding
compositionality = compositionality_score(model, root_words, affixed_words)
```

## Experiments

### Pretraining Comparison

```bash
# Baseline (standard tokenization)
python experiments/train.py --config-name pretraining \
    trainer.dataset.name=openwebtext-tokenized

# With StochasTok
python experiments/train.py --config-name pretraining \
    trainer.dataset.name=openwebtext-tokenized-stochastok0.1

# With Patok (ours)
python experiments/train.py --config-name pretraining \
    trainer.dataset.name=openwebtext-tokenized-patok0.3-0.3-0.7
```

### Evaluation Pipeline

```bash
# Run full evaluation suite
bash scripts/run_full_evaluation.sh \
    --models baseline,stochastok,patok \
    --benchmarks pacute,winogrande,hellaswag,arc
```

## Configuration

All experiments use Hydra for configuration. Key config files:

- `configs/pretraining.yaml`: Pretraining hyperparameters
- `configs/instruction_tuning.yaml`: Fine-tuning setup
- `configs/tokenization/patok.yaml`: Patok parameters
- `configs/evaluation/pacute.yaml`: PACUTE benchmark settings

Override configs from command line:
```bash
python experiments/train.py \
    model.n_layers=12 \
    trainer.batch_size=128 \
    tokenization.affix_preference=0.8
```

## Citation

If you use this code or PACUTE benchmark, please cite:

```bibtex
@inproceedings{filipino-morphology-llm-2026,
  title={Morphologically-Aware Tokenization for Low-Resource Languages},
  author={[Authors]},
  booktitle={Proceedings of [Conference]},
  year={2026}
}
```

## License

[To be determined]

## Acknowledgments

- Based on [StochasTok](https://arxiv.org/abs/2506.01687) by Sims et al.
- Inspired by [CUTE](https://aclanthology.org/2024.emnlp-main.177/) by Edman et al.
- Filipino linguistic resources from UP Diksiyonaryo

## Contact

[Contact information]
