# Hierarchical Task Framework for PACUTE

## Overview

The Hierarchical Task Framework organizes PACUTE evaluation tasks into 6 levels that build compositionally. This enables **precise diagnosis** of model capabilities and identification of specific failure points.

## Design Philosophy

### Key Principle: Compositional Capabilities

Each level requires capabilities from previous levels:

```
Level 0: Character Recognition (baseline)
    ↓
Level 1: Character Manipulation (requires Level 0)
    ↓
Level 2: Morpheme Decomposition (requires Level 0)
    ↓
Level 3: Morpheme Manipulation (requires Level 1 + Level 2)
    ↓
Level 4: Morpheme Composition (requires Level 2)
    ↓
Level 5: Complex Morphological Reasoning (requires Level 2-4)
```

### Diagnostic Cascade

If a model fails at Level N, we expect failures at dependent levels:

**Example:**
- ✅ Level 0 (95%): Model can recognize characters
- ✅ Level 1 (75%): Model can manipulate strings
- ❌ Level 2 (40%): **Model fails at morpheme decomposition**
- ❌ Level 3 (25%): Expected failure (needs Level 1 + Level 2)
- ❌ Level 4 (20%): Expected failure (needs Level 2)

**Diagnosis:** Tokenization doesn't align with morphological boundaries.

**Solution:** Use affix-aware tokenization (Patok) to improve Level 2.

## Level Descriptions

### Level 0: Character Recognition

**Purpose:** Test if model has access to individual characters

**Tasks:**
- Character identification: "What is the 3rd character in 'kumain'?" → 'm'
- Character counting: "How many 'a's in 'kumain'?" → 2
- Character presence: "Does 'kumain' contain 'u'?" → Yes

**Failure Mode:** Model tokenization is too coarse-grained, characters are lost

**Baseline Performance:** Usually high (85-95%) for modern tokenizers

---

### Level 1: Character Manipulation

**Purpose:** Test if model can perform character-level operations

**Tasks:**
- Character deletion: "Delete the 3rd character from 'kumain'" → "kuain"
- Character insertion: "Insert 'l' at position 3 in 'kumain'" → "kulain"
- Character substitution: "Replace 'm' with 'l' in 'kumain'" → "kulain"
- Character permutation: "Swap positions 2 and 4 in 'kumain'" → "kmauin"

**Requirements:** Level 0 (character identification)

**Failure Mode:** Model can see characters but cannot manipulate them

**Baseline Performance:** Moderate (60-75%) - depends on training data

---

### Level 2: Morpheme Decomposition

**Purpose:** Test if model understands morphological boundaries

**Tasks:**
- Affix identification: "What is the infix in 'kumain'?" → "um"
- Root extraction: "What is the root of 'nagluto'?" → "luto"
- Affix counting: "How many affixes in 'pinaglutuan'?" → 3
- Syllable boundary identification: "How many syllables in 'kumain'?" → 3

**Requirements:** Understanding of morphology (not directly from Level 0/1)

**Failure Mode:** **Critical bottleneck** - tokenization ignores morphology

**Baseline Performance:** Low (30-50%) with standard tokenizers

**Patok Improvement:** High (+20-30%) - primary benefit of affix-aware tokenization

---

### Level 3: Morpheme Manipulation

**Purpose:** Test if model can transform morphological units

**Tasks:**
- Affix removal: "Remove the infix from 'kumain'" → "kain"
- Affix substitution: "Replace 'um' with 'in' in 'kumain'" → "kinain"
- Morpheme reordering: Move affix to different position (for testing)

**Requirements:** Level 2 (identify morphemes) + Level 1 (manipulate strings)

**Failure Modes:**
- ❌ Level 2, ❌ Level 3: Cannot identify morphemes → can't manipulate them
- ✅ Level 2, ❌ Level 3: Can identify but not manipulate (rare)
- ✅ Level 1, ❌ Level 3: Lacks morphological knowledge

**Baseline Performance:** Low (20-40%)

**Patok Improvement:** Cascades from Level 2 (+15-25%)

---

### Level 4: Morpheme Composition

**Purpose:** Test if model can build words from morphemes (inverse of Level 3)

**Tasks:**
- Affix application: "Add infix 'um' to 'kain'" → "kumain"
- Multiple affixes: "Add prefix 'nag' to 'luto'" → "nagluto"
- Circumfix: "Apply 'pag-...-an' to 'luto'" → "paglutuan"

**Requirements:** Level 2 (understand morphology)

**Failure Modes:**
- ✅ Level 2, ✅ Level 3, ❌ Level 4: Asymmetric (can break but not build)
- ✅ Level 2, ✅ Level 4, ❌ Level 3: Can compose but not decompose
- ✅ Level 3, ✅ Level 4: Symmetric morphological understanding

**Baseline Performance:** Low (15-35%)

**Patok Improvement:** Cascades from Level 2 (+20-25%)

---

### Level 5: Complex Morphological Reasoning

**Purpose:** Test multi-step operations and linguistic knowledge

**Tasks:**
- Aspect transformation: "Convert 'kumain' (contemplated) to completed aspect" → proper form
- Derivational morphology: "Make 'luto' into occupation noun" → "kusinero" or "magluluto"
- Compound operations: "Extract root from X, add affix Y, apply transformation Z"

**Requirements:** All previous levels + linguistic knowledge

**Failure Mode:** Lacks compositional reasoning or linguistic training

**Baseline Performance:** Very low (10-25%)

**Patok Improvement:** Moderate (+10-15%) - requires more than just tokenization

## Usage

### 1. Generate Hierarchical Tasks

```python
from src.evaluation import HierarchicalTaskGenerator
import pandas as pd

# Load data
words_df = pd.read_json("data/corpora/pacute_data/syllables.jsonl", lines=True)
affixes_df = pd.read_csv("data/affixes/annotated_affixes.csv")

# Initialize generator
generator = HierarchicalTaskGenerator(words_df, affixes_df)

# Generate all levels
tasks_by_level = generator.generate_all_levels(
    n_per_subcategory=50,
    format="both"  # Generate both MCQ and generative
)

# Save tasks
generator.save_tasks(tasks_by_level, "data/benchmarks/hierarchical/", format="mcq")
generator.save_tasks(tasks_by_level, "data/benchmarks/hierarchical/", format="gen")
```

### 2. Evaluate Model

```python
# Evaluate your model on the tasks
# (Your evaluation code here - depends on your model)

results = []
for level, tasks in tasks_by_level.items():
    for task in tasks:
        prediction = model.predict(task.prompt_en)
        correct = (prediction == task.answer)

        results.append({
            "level": level,
            "category": task.category,
            "subcategory": task.subcategory,
            "correct": correct,
            "predicted_answer": prediction,
            "gold_answer": task.answer,
            "word": task.word
        })

# Save results
import json
with open("results/model_predictions.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
```

### 3. Analyze Results

```python
from src.evaluation import HierarchicalAnalyzer

# Load analyzer
analyzer = HierarchicalAnalyzer("results/model_predictions.jsonl")

# Generate diagnostic report
report = analyzer.generate_diagnostic_report()
print(report)

# Get capability profile
profile = analyzer.compute_capability_profile()
print(profile)

# Identify failure cascades
cascades = analyzer.identify_failure_cascades(threshold=0.6)
for level_n, level_n_plus_1, acc_n, acc_n_plus_1 in cascades:
    print(f"Cascade: Level {level_n} ({acc_n:.1%}) → Level {level_n_plus_1} ({acc_n_plus_1:.1%})")

# Get failure examples for debugging
failures = analyzer.get_failure_examples(level=2, n=10)
for failure in failures:
    print(f"Failed: {failure['prompt_en']}")
    print(f"  Gold: {failure['gold_answer']}, Predicted: {failure['predicted_answer']}")
```

### 4. Compare Models

```python
from src.evaluation import compare_multiple_models, HierarchicalAnalyzer

# Load multiple analyzers
analyzers = {
    "baseline": HierarchicalAnalyzer("results/baseline_predictions.jsonl"),
    "stochastok": HierarchicalAnalyzer("results/stochastok_predictions.jsonl"),
    "patok": HierarchicalAnalyzer("results/patok_predictions.jsonl"),
}

# Compare
comparison = compare_multiple_models(analyzers)
print(comparison)

# Find where Patok helps most
patok_vs_baseline = analyzers["patok"].compare_models(analyzers["baseline"])
significant_improvements = patok_vs_baseline[patok_vs_baseline["significant"]]
print("\nSignificant Patok Improvements:")
print(significant_improvements)
```

## Expected Performance Patterns

### Baseline Model (Standard Tokenization)

```
Level 0: 95% ████████████████████ (Character recognition - OK)
Level 1: 75% ███████████████      (Character manipulation - OK)
Level 2: 40% ████████            (Morpheme decomposition - FAIL)
Level 3: 25% █████               (Morpheme manipulation - cascade fail)
Level 4: 20% ████                (Morpheme composition - cascade fail)
Level 5: 15% ███                 (Complex reasoning - cascade fail)
```

**Diagnosis:** Bottleneck at Level 2 (morphological awareness)

### StochasTok Model

```
Level 0: 95% ████████████████████ (No change)
Level 1: 80% ████████████████     (+5% - better subword awareness)
Level 2: 50% ██████████          (+10% - some morphological help)
Level 3: 35% ███████             (+10% - cascaded improvement)
Level 4: 30% ██████              (+10% - cascaded improvement)
Level 5: 20% ████                (+5% - small improvement)
```

**Diagnosis:** Uniform improvement, but still struggles with morphology

### Patok Model (Affix-Aware)

```
Level 0: 95% ████████████████████ (No change - not character-level)
Level 1: 80% ████████████████     (Similar to StochasTok)
Level 2: 78% ███████████████████  (+38% over baseline - MAJOR)
Level 3: 70% ██████████████       (+45% - strong cascade)
Level 4: 68% █████████████        (+48% - strong cascade)
Level 5: 50% ██████████           (+35% - benefits from morphology)
```

**Diagnosis:** Targeted improvement at morphological levels (2-4), showing affix-aware tokenization provides fundamental understanding

## Key Insights from Hierarchical Analysis

### 1. Pinpoint Failure Source

Instead of: *"Model performs poorly on Filipino tasks"*

We can say: *"Model fails at Level 2 (morpheme decomposition) with 40% accuracy, while achieving 95% at Level 0 and 75% at Level 1. This indicates tokenization doesn't align with morphological boundaries."*

### 2. Distinguish Skill Types

- **Recognition vs. Manipulation:** A model might identify affixes (Level 2) but not manipulate them (Level 3)
- **Decomposition vs. Composition:** Asymmetric performance on Level 3 vs. 4 indicates one-directional understanding
- **Fundamental vs. Applied:** High Level 2 with low Level 5 shows morphological knowledge but poor application

### 3. Validate Interventions

Patok's effectiveness can be precisely quantified:
- **Primary benefit:** Level 2 (+38%)
- **Cascade effects:** Level 3 (+45%), Level 4 (+48%)
- **No regression:** Level 0-1 maintained
- **Conclusion:** Patok improves fundamental morphological understanding, not task-specific memorization

### 4. Guide Future Work

If Level 5 remains low despite good Level 2-4:
- Need more morphologically diverse training data
- Need curriculum learning emphasizing compositional operations
- Need explicit linguistic knowledge injection

## Implementation Notes

### Adding New Levels

To add a new intermediate level (e.g., Level 2.5):

```python
def generate_level2_5_custom(self, n: int, format="gen"):
    """Custom level between morpheme decomposition and manipulation."""
    tasks = []
    # Your task generation logic
    return tasks
```

### Adding New Subcategories

Within existing levels:

```python
def generate_level1_character_duplication(self, n: int, format="gen"):
    """New subcategory: duplicate a character."""
    tasks = []
    for word in self.words_df.sample(n)["word"]:
        pos = random.randint(0, len(word)-1)
        answer = word[:pos+1] + word[pos] + word[pos+1:]
        tasks.append(HierarchicalTask(
            level=1,
            category="manipulation",
            subcategory="character_duplication",
            ...
        ))
    return tasks
```

### Custom Metrics

Beyond accuracy, compute task-specific metrics:

```python
def compute_morphological_precision_recall(self, level=2):
    """For Level 2: How well does model identify exact affix boundaries?"""
    # Implementation specific to your needs
    pass
```

## References

- Original PACUTE benchmark design
- CUTE (English subword benchmark)
- StochasTok paper
- Patok (affix-aware tokenization)

## See Also

- `src/evaluation/hierarchical_tasks.py` - Task generation
- `src/evaluation/hierarchical_analysis.py` - Result analysis
- `scripts/demo_hierarchical_tasks.py` - Demo script
- `experiments/evaluate_hierarchical.py` - Full evaluation pipeline
