# Developer Quick Reference

## Module Structure

```
src/pacute/
├── __init__.py              # Package exports and API
├── constants.py             # ✨ NEW: Shared constants
├── utils.py                 # ✨ NEW: Common utilities
├── sampling.py              # ✨ ENHANCED: Frequency-aware sampling
├── string_operations.py     # ✨ ENHANCED: String manipulations
├── syllabification_operations.py  # ✨ ENHANCED: Syllabification
├── affixation.py           # Dataset: Affixation tasks
├── composition.py          # Dataset: Composition tasks
├── manipulation.py         # Dataset: Manipulation tasks
└── syllabification.py      # Dataset: Syllabification tasks
```

## Import Guide

### Core Dataset Functions
```python
from pacute import (
    create_affixation_dataset,
    create_composition_dataset,
    create_manipulation_dataset,
    create_syllabification_dataset
)
```

### Sampling Functions
```python
from pacute import (
    load_frequency_data,      # Load word frequency data
    add_frequency_ranks,       # Add ranks to DataFrame
    sample_by_frequency,       # Frequency-weighted sampling
    sample_stratified_by_length  # Length-balanced sampling
)
```

### String Operations
```python
from pacute import (
    string_to_chars,          # String → List[char]
    chars_to_string,          # List[char] → String
    normalize_diacritic,      # Remove diacritics
    diacritize,               # Add diacritics
    spell_string              # Add spaces between chars
)
```

### Syllabification
```python
from pacute import (
    syllabify,                # Syllabify Filipino words
    normalize_text            # Clean/normalize text
)
```

### Utilities (for extending the package)
```python
from pacute.utils import (
    prepare_mcq_outputs,      # Format MCQ questions
    prepare_gen_outputs,      # Format generative questions
    validate_dataframe_columns,  # Validate DataFrame
    validate_positive_integer,   # Validate int parameter
    validate_probability         # Validate probability [0,1]
)
```

### Constants (for extending the package)
```python
from pacute.constants import (
    # MCQ Configuration
    MCQ_LABEL_MAP,
    NUM_MCQ_OPTIONS,
    
    # Word Length Thresholds
    MIN_WORD_LENGTH_COMPOSITION,
    MIN_WORD_LENGTH_MANIPULATION,
    MIN_WORD_LENGTH_SYLLABIFICATION,
    
    # Character Sets
    VOWELS,
    DIACRITICS,
    ACCENTED_VOWELS,
    
    # Diacritic Mappings
    DIACRITIC_MAP,
    REVERSE_DIACRITIC_MAP,
    
    # Stress Classification
    MABILIS,
    MALUMI,
    MARAGSA,
    STRESS_PRONUNCIATION_MAP,
    
    # Default Values
    DEFAULT_FREQUENCY_FILE,
    DEFAULT_FREQ_WEIGHT,
    DEFAULT_RANDOM_STATE
)
```

## Common Workflows

### 1. Basic Dataset Creation
```python
from pacute import create_syllabification_dataset

# Create dataset
mcq_df, gen_df = create_syllabification_dataset(
    syllables_path='data/syllables.jsonl',
    word_freq_path='data/word_frequencies.csv',
    n_samples_per_task=20
)

# Save to files
mcq_df.to_json('mcq_syllabification.jsonl', orient='records', lines=True)
gen_df.to_json('gen_syllabification.jsonl', orient='records', lines=True)
```

### 2. Frequency-Aware Sampling
```python
import pandas as pd
from pacute import load_frequency_data, add_frequency_ranks, sample_by_frequency

# Load your word data
words_df = pd.read_json('data/syllables.jsonl', lines=True)

# Load and add frequency ranks
freq_df = load_frequency_data('data/word_frequencies.csv')
words_df = add_frequency_ranks(words_df, freq_df, word_column='normalized_word')

# Sample with different strategies
random_sample = sample_by_frequency(words_df, n_samples=100, freq_weight=0.0)
balanced_sample = sample_by_frequency(words_df, n_samples=100, freq_weight=0.5)
common_sample = sample_by_frequency(words_df, n_samples=100, freq_weight=1.0)
```

### 3. Length-Stratified Sampling
```python
from pacute import sample_stratified_by_length

# Ensure balanced word lengths
df['word_length'] = df['normalized_word'].str.len()
balanced_df = sample_stratified_by_length(
    df, 
    n_samples=100, 
    length_column='word_length',
    random_state=42
)
```

### 4. String Manipulations
```python
from pacute import string_to_chars, spell_string, normalize_diacritic

word = "kumakain"

# Spell out
spelled = spell_string(word)  # "k u m a k a i n"

# Convert to char list
chars = string_to_chars(word)  # ['k', 'u', 'm', 'a', 'k', 'a', 'i', 'n']

# Remove diacritics
clean = normalize_diacritic("kumakáin")  # "kumakain"
```

### 5. Syllabification
```python
from pacute import syllabify

# Syllabify Filipino words
syllables = syllabify("kumakain")  # ['ku', 'ma', 'ka', 'in']
syllables = syllabify("magandá")   # ['ma', 'gan', 'dá']

# Handles 'ng' as single unit
syllables = syllabify("mangga")    # ['mang', 'ga']
```

### 6. Creating Custom Validators
```python
from pacute.utils import (
    validate_dataframe_columns,
    validate_positive_integer,
    validate_probability
)

def my_custom_function(df, n_samples, weight):
    # Validate inputs
    validate_dataframe_columns(df, ['word', 'rank'], "Input DataFrame")
    validate_positive_integer(n_samples, "n_samples")
    validate_probability(weight, "weight")
    
    # Your logic here
    pass
```

## Parameter Defaults

### Sampling Functions
```python
load_frequency_data(
    freq_file_path='data/word_frequencies.csv'  # DEFAULT_FREQUENCY_FILE
)

add_frequency_ranks(
    df,
    freq_df,
    word_column='normalized_word',
    rank_fillna=100000  # DEFAULT_RANK_FILLNA
)

sample_by_frequency(
    df,
    n_samples,
    freq_weight=0.5,    # DEFAULT_FREQ_WEIGHT
    random_state=42     # DEFAULT_RANDOM_STATE
)
```

## Type Hints Guide

All refactored modules use type hints. Common types:

```python
from typing import List, Dict, Any, Optional, Tuple

# Function signatures
def process_word(word: str) -> List[str]: ...
def create_options(correct: str, incorrect: List[str]) -> Dict[str, str]: ...
def find_syllable(syllables: List[str]) -> Tuple[str, int]: ...
def maybe_transform(text: Optional[str] = None) -> str: ...
```

## Best Practices

### 1. Always Validate Inputs
```python
from pacute.utils import validate_dataframe_columns

def my_function(df):
    validate_dataframe_columns(df, ['required_col'], "My DataFrame")
    # Now safe to use df['required_col']
```

### 2. Use Constants for Configuration
```python
from pacute.constants import MIN_WORD_LENGTH_COMPOSITION

def filter_words(df):
    return df[df['word'].str.len() >= MIN_WORD_LENGTH_COMPOSITION]
```

### 3. Leverage Type Hints
```python
from typing import List, Optional

def process_items(items: List[str], limit: Optional[int] = None) -> List[str]:
    # Type hints help IDEs provide better autocomplete
    # and help catch errors early
    pass
```

### 4. Reuse Utility Functions
```python
from pacute.utils import prepare_mcq_outputs, prepare_gen_outputs

# Don't reimplement these - use the shared versions
mcq_result = prepare_mcq_outputs(text_en, text_tl, options)
gen_result = prepare_gen_outputs(text_en, text_tl, label)
```

## Error Handling

All validation functions raise descriptive errors:

```python
from pacute import sample_by_frequency

# Missing 'rank' column
sample_by_frequency(df, n_samples=10)
# ValueError: DataFrame must have 'rank' column. 
#             Use add_frequency_ranks() before calling sample_by_frequency().

# Invalid n_samples
sample_by_frequency(df, n_samples=-5)
# ValueError: n_samples must be a positive integer, got -5

# Invalid freq_weight
sample_by_frequency(df, n_samples=10, freq_weight=1.5)
# ValueError: freq_weight must be between 0 and 1, got 1.5
```

## Testing Your Extensions

When extending the package, follow this pattern:

```python
import pandas as pd
from pacute import load_frequency_data, add_frequency_ranks

def test_my_new_function():
    # Create test data
    test_df = pd.DataFrame({
        'normalized_word': ['ako', 'ikaw', 'siya']
    })
    
    # Load frequency data
    freq_df = load_frequency_data()
    
    # Add ranks
    ranked_df = add_frequency_ranks(test_df, freq_df)
    
    # Test your function
    result = my_new_function(ranked_df)
    
    # Assertions
    assert len(result) > 0
    assert 'rank' in result.columns
```

## IDE Setup

For best experience with type hints:

1. **VS Code**: Install Python extension
2. **PyCharm**: Type hints work out of the box
3. **Enable type checking** (optional):
   ```bash
   pip install mypy
   mypy src/pacute/
   ```
