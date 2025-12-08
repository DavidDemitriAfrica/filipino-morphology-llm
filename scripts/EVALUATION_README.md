# Benchmark Evaluation Guide

## Overview

This directory contains scripts for:
1. **Generating benchmarks**: Create evaluation datasets
2. **Evaluating models**: Test language models on Filipino morphology tasks
3. **Analysis**: Diagnose model capabilities and bottlenecks

## Quick Start

### Generate All Benchmarks
```bash
python scripts/generate_all_benchmarks.py
```

This creates all benchmark files in `data/benchmarks/`.

### Evaluate a Model
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute cute \
    --max-samples 100
```

## Benchmarks

### 1. PACUTE (Pilipino Affix and Character-Level Understanding of Tokens Evaluation)
- **Total**: 5,845 tasks (MCQ) + 5,380 tasks (Generative) = **11,225 tasks**
- **Affixation** (280 tasks): Filipino affix identification and application
- **Composition** (3,905 tasks): Character counting, diacritics, word formation
- **Manipulation** (5,120 tasks): Character operations (insert, delete, swap, etc.)
- **Syllabification** (1,280 tasks): Syllable counting, stress, reduplication

### 2. Hierarchical Tasks
- **Total**: 1,198 tasks (MCQ) + 600 tasks (Generative) = **1,798 tasks**
- **6 Levels**: Character recognition → Complex morphological reasoning
- Designed to diagnose where models fail in the linguistic hierarchy

### 3. LangGame (Subword Understanding)
- **Total**: 3,000 tasks (2,000 train + 1,000 val)
- 6 question types: most/contains/starts/ends/longest/shortest
- Tests understanding of token composition

### 4. Multi-Digit Addition
- **Total**: 3,000 tasks (2,000 train + 1,000 val)
- 3-digit addition problems
- Tests numerical reasoning



## Models

### GPT-2 Family
- `gpt2` (124M parameters) - PT
- `gpt2-medium` (355M) - PT
- `gpt2-large` (774M) - PT

### Qwen Family
- `qwen-2.5-0.5b` - PT
- `qwen-2.5-0.5b-it` - IT
- `qwen-2.5-1.5b` - PT
- `qwen-2.5-1.5b-it` - IT
- `qwen-2.5-3b` - PT
- `qwen-2.5-3b-it` - IT

### Llama Family
- `llama-3.2-1b` - PT
- `llama-3.2-1b-it` - IT
- `llama-3.2-3b` - PT
- `llama-3.2-3b-it` - IT

### Gemma Family
- `gemma-2b` - PT
- `gemma-2b-it` - IT
- `gemma-7b` - PT
- `gemma-7b-it` - IT

### Cerebras GPT (Open Source GPT)
- `cerebras-gpt-111m` - PT
- `cerebras-gpt-256m` - PT
- `cerebras-gpt-590m` - PT
- `cerebras-gpt-1.3b` - PT

**PT** = Pre-trained | **IT** = Instruction-tuned

## Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Path Confidence**: Average softmax probability on correct answer
- **Normalized Accuracy**: Accuracy normalized to account for random chance

## Benchmark Generation

All benchmark generation scripts are in `src/evaluation/datasets/scripts/`:

### Generate All Benchmarks (Recommended)
```bash
python scripts/generate_all_benchmarks.py
```

This runs all individual generation scripts and creates:
- PACUTE benchmarks (affixation, composition, manipulation, syllabification)
- Hierarchical benchmarks (6 levels)
- LangGame dataset
- Multi-digit addition dataset

### Generate Individual Benchmarks
```bash
# PACUTE only
python src/evaluation/datasets/scripts/generate_pacute_benchmarks.py

# Hierarchical only
python src/evaluation/datasets/scripts/generate_hierarchical_benchmark.py

# LangGame only
python src/evaluation/datasets/scripts/generate_langgame_benchmark.py

# Math only
python src/evaluation/datasets/scripts/generate_math_benchmark.py
```

## Model Evaluation

### Quick Test (10 samples)
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute \
    --max-samples 10
```

### Single Model on Multiple Benchmarks
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute hierarchical
```

### Multiple Models
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 qwen-2.5-0.5b cerebras-gpt-111m \
    --benchmarks pacute hierarchical
```

### Full Evaluation (All Models, All Benchmarks)
```bash
bash scripts/run_eval_batch.sh
```

### Custom Output Directory
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute \
    --output-dir my_results
```

## Output Format

Results are saved as JSON with this structure:

```json
{
  "model_name": {
    "hf_model_name": "gpt2",
    "model_type": "pt",
    "benchmarks": {
      "pacute": {
        "num_samples": 560,
        "accuracy": 0.4521,
        "f1_score": 0.4521,
        "precision": 0.4521,
        "recall": 0.4521,
        "path_confidence": 0.3124,
        "normalized_accuracy": 0.2695
      }
    }
  }
}
```

## Requirements

```bash
pip install torch transformers tqdm datasets
```

## Evaluation Framework Structure

```
src/evaluation/
├── datasets/
│   ├── generators/          # Benchmark task generators
│   │   ├── affixation.py
│   │   ├── composition.py
│   │   ├── manipulation.py
│   │   ├── syllabification.py
│   │   └── hierarchical.py
│   ├── scripts/             # Generation scripts
│   │   ├── generate_pacute_benchmarks.py
│   │   ├── generate_hierarchical_benchmark.py
│   │   ├── generate_langgame_benchmark.py
│   │   └── generate_math_benchmark.py
│   └── converters/          # Format converters
├── loaders/                 # Benchmark loaders
│   ├── pacute.py
│   ├── langgame.py
│   └── registry.py          # load_benchmark() function
├── evaluators/              # Model evaluators
│   ├── hierarchical.py      # HierarchicalAnalyzer
│   ├── mcq_evaluator.py
│   └── wrapper.py
├── metrics/                 # Evaluation metrics
└── utils/                   # Helper utilities
    └── sampling.py

scripts/                     # User-facing scripts
├── generate_all_benchmarks.py  # Master generation script
├── run_benchmark_evaluation.py # Model evaluation
├── demo_hierarchical_tasks.py  # Demo usage
└── evaluate_downstream.py      # Downstream task eval
```

## Implementation Details

### MCQ Evaluation
- Models are evaluated using log-probability scoring
- For each question, we compute log P(answer | question) for all options
- The option with highest log probability is selected
- Metrics are computed by comparing predictions to ground truth

### Generative Evaluation
- Models generate text given a prompt
- Answers are compared using exact match, contains match, or prefix match
- Used for hierarchical tasks in generative format

### Model Loading
- Models are loaded from HuggingFace using `AutoModelForCausalLM`
- FP16 precision is used on GPU for efficiency
- Models run in evaluation mode (no dropout)

### Benchmark Loaders
All benchmarks can be loaded using the unified interface:
```python
from evaluation.loaders import load_benchmark

# Returns generator: (prefix, ground_truth, false_options)
loader = load_benchmark("pacute")
for prefix, gt, false_opts in loader:
    # Evaluate...
```

Available benchmarks: `pacute`, `affixation`, `composition`, `manipulation`, 
`syllabification`, `hierarchical`, `langgame-train`, `langgame-val`, 
`multi_digit_addition-train`, `multi_digit_addition-val`

## Notes
1. Generate using StochasTok data generation scripts
2. Download from the StochasTok repository
3. Skip LangGame and evaluate on PACUTE and CUTE only

### GPU Memory
Larger models (7B+) require significant GPU memory:
- 7B models: ~14GB VRAM (FP16)
- 3B models: ~6GB VRAM (FP16)
- 1B models: ~2GB VRAM (FP16)

Use `--device cpu` if GPU memory is insufficient (will be slower).

### Model Access
Some models (Llama, Gemma) may require HuggingFace authentication:
```bash
huggingface-cli login
```

## Example Results

Expected performance ranges (approximate):

| Model Size | PACUTE Accuracy | CUTE Accuracy |
|------------|----------------|---------------|
| <500M | 30-45% | 25-40% |
| 500M-2B | 40-55% | 35-50% |
| 2B-7B | 50-65% | 45-60% |
| 7B+ | 60-75% | 55-70% |

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install torch transformers
```

### CUDA out of memory
```bash
# Use smaller models or CPU
python scripts/run_benchmark_evaluation.py --models gpt2 --device cpu
```

### Model not found
Some models require authentication. Log in to HuggingFace:
```bash
huggingface-cli login
```

### LangGame FileNotFoundError
LangGame data not available. Skip it:
```bash
python scripts/run_benchmark_evaluation.py \
    --models gpt2 \
    --benchmarks pacute cute  # Skip langgame
```
