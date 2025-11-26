# Quick Start Guide

Get up and running with Filipino Morphology-LLM in 5 minutes.

## Setup

```bash
# Navigate to repository
cd /home/ubuntu/filipino-morphology-llm

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify setup
python scripts/verify_setup.py
```

## Basic Usage Examples

### 1. Tokenize Text with Patok

```python
from src.tokenization import PatokProcessor

# Initialize processor
processor = PatokProcessor(
    base_tokenizer="gpt2",
    affixes_file="data/affixes/filipino_affixes.txt",
    expand_prop=0.3,
    contract_prop=0.3,
    affix_preference=0.7
)

# Process Filipino text
text = "Nagluto siya ng masarap na pagkain."
tokens = processor.process(text)
print(f"Tokens: {tokens}")
```

### 2. Evaluate on PACUTE Benchmark

```python
from src.evaluation import (
    create_affixation_dataset,
    create_composition_dataset,
    create_manipulation_dataset,
    create_syllabification_dataset
)

# Generate PACUTE tasks
affixation_tasks = create_affixation_dataset(
    n_inflection=50,
    n_identification=50,
    output_dir="data/benchmarks/custom/",
    format="mcq"
)

# Load and use existing benchmark
import json
with open("data/benchmarks/mcq_affixation.jsonl") as f:
    tasks = [json.loads(line) for line in f]

print(f"Loaded {len(tasks)} affixation tasks")
```

### 3. Train Model with Patok Tokenization

```bash
# First, tokenize your corpus with Patok
python src/data_processing/patok_expand_contract_dataset.py \
    --dataset_name openwebtext \
    --output_dir data/processed/ \
    --expand_prop 0.3 \
    --contract_prop 0.3 \
    --affix_preference 0.7

# Then train
python experiments/train.py \
    --config-name pretraining \
    trainer.dataset.name=processed/openwebtext-patok
```

### 4. Compare Tokenizations

```python
import tiktoken
from src.tokenization import PatokProcessor, StochastokProcessor

# Load tokenizer
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

# Initialize processors
stochastok = StochastokProcessor(gpt2_tokenizer, expand_prop=0.1)
patok = PatokProcessor(
    base_tokenizer="gpt2",
    affixes_file="data/affixes/filipino_affixes.txt",
    expand_prop=0.3,
    contract_prop=0.3,
    affix_preference=0.7
)

# Compare tokenizations
text = "Pinagmamalaki niya ang kanyang mga anak."

baseline_tokens = gpt2_tokenizer.encode(text)
stochastok_tokens = stochastok.process_tokens(baseline_tokens)
patok_tokens = patok.process(text)

print("Baseline:", gpt2_tokenizer.decode_tokens_bytes(baseline_tokens))
print("StochasTok:", [gpt2_tokenizer.decode_single_token_bytes(t) for t in stochastok_tokens])
print("Patok:", [gpt2_tokenizer.decode_single_token_bytes(t) for t in patok_tokens])
```

### 5. Analyze Morphological Alignment

```python
from src.analysis import (
    morpheme_token_mutual_information,
    affix_consistency_entropy
)

# Load morphologically annotated data
import pandas as pd
words_df = pd.read_json("data/corpora/pacute_data/syllables.jsonl", lines=True)

# Analyze tokenizations
mi_score = morpheme_token_mutual_information(patok, words_df)
print(f"Morpheme-Token MI: {mi_score:.3f}")

consistency = affix_consistency_entropy(patok, words_df["word"].tolist())
print(f"Affix Consistency: {consistency:.3f}")
```

## Common Workflows

### Full Training Pipeline

```bash
# 1. Preprocess data
python src/data_processing/tokenize_dataset.py \
    --dataset openwebtext \
    --output data/processed/openwebtext-base

# 2. Apply Patok
python src/data_processing/patok_expand_contract_dataset.py \
    --input data/processed/openwebtext-base \
    --output data/processed/openwebtext-patok

# 3. Train model
python experiments/train.py \
    --config-name pretraining \
    trainer.dataset.name=openwebtext-patok \
    trainer.save_dir=checkpoints/patok_model

# 4. Evaluate
python experiments/eval.py \
    --checkpoint checkpoints/patok_model/final.pt \
    --benchmarks pacute,winogrande,hellaswag
```

### PACUTE Benchmark Generation

```python
from src.evaluation import (
    create_affixation_dataset,
    create_composition_dataset,
    create_manipulation_dataset,
    create_syllabification_dataset,
    load_frequency_data,
    sample_by_frequency
)

# Load word data
word_freq_df = load_frequency_data("data/corpora/pacute_data/word_frequencies.csv")
syllables_df = pd.read_json("data/corpora/pacute_data/syllables.jsonl", lines=True)

# Sample words with frequency awareness
sampled_words = sample_by_frequency(
    word_freq_df,
    n_samples=100,
    freq_weight=0.5  # Balance between common and rare words
)

# Generate custom benchmark
custom_tasks = create_affixation_dataset(
    words_df=sampled_words,
    n_inflection=50,
    n_identification=50,
    output_dir="data/benchmarks/custom/",
    format="both"  # Both MCQ and generative
)
```

### Ablation Study

```bash
# Compare different Patok configurations
for expand in 0.1 0.3 0.5; do
  for contract in 0.1 0.3 0.5; do
    for affix_pref in 0.5 0.7 0.9; do
      python experiments/train.py \
        --config-name pretraining \
        tokenization.expand_prop=$expand \
        tokenization.contract_prop=$contract \
        tokenization.affix_preference=$affix_pref \
        trainer.save_dir=checkpoints/patok_${expand}_${contract}_${affix_pref}
    done
  done
done
```

## Configuration

All experiments use Hydra for configuration management. Override any parameter:

```bash
python experiments/train.py \
    model.n_layers=12 \
    model.hidden_dim=768 \
    trainer.batch_size=64 \
    trainer.learning_rate=3e-4 \
    tokenization.affix_preference=0.8
```

Or create custom config file:

```yaml
# configs/my_experiment.yaml
defaults:
  - pretraining

model:
  n_layers: 12
  hidden_dim: 768

trainer:
  batch_size: 64
  learning_rate: 3e-4

tokenization:
  expand_prop: 0.4
  contract_prop: 0.4
  affix_preference: 0.8
```

Then run:
```bash
python experiments/train.py --config-name my_experiment
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_affix_processor.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Notebooks

Explore data and results interactively:

```bash
jupyter notebook notebooks/
```

Key notebooks:
- `create_affixation.ipynb`: Generate affixation tasks
- `create_composition_string_manipulation_syllabification.ipynb`: Other task types
- `diksiyonaryo.ipynb`: Explore UP Diksiyonaryo data

## Troubleshooting

### Import Errors

```python
# If you get "No module named 'tokenization'"
# Make sure you installed the package:
pip install -e /home/ubuntu/filipino-morphology-llm

# Or use explicit imports:
from src.tokenization import PatokProcessor
```

### Path Issues

```python
# Always use absolute paths or paths relative to repo root
from pathlib import Path
repo_root = Path(__file__).parent.parent
affix_file = repo_root / "data/affixes/filipino_affixes.txt"
```

### CUDA Errors

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Train on CPU if needed
python experiments/train.py trainer.device=cpu
```

## Next Steps

- Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) if updating from old repos
- Check [README.md](README.md) for detailed documentation
- Explore example notebooks in `notebooks/`
- Run verification: `python scripts/verify_setup.py`

## Resources

- **StochasTok Paper**: https://arxiv.org/abs/2506.01687
- **CUTE Paper**: https://aclanthology.org/2024.emnlp-main.177/
- **UP Diksiyonaryo**: https://updiksiyonaryo.ph/
