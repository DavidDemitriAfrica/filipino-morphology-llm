"""
PACUTE: Philippine Annotated Corpus for Understanding Tagalog Entities

A Python package for creating linguistic datasets focused on Filipino/Tagalog
language processing, including affixation, composition, manipulation, and
syllabification tasks.
"""

__version__ = "0.1.0"

# Core dataset creation functions
from .affixation import create_affixation_dataset
from .composition import create_composition_dataset
from .manipulation import create_manipulation_dataset
from .syllabification import create_syllabification_dataset

# Syllabification utilities
from .syllabification_operations import syllabify, normalize_text

# Sampling utilities
from .sampling import (
    load_frequency_data,
    add_frequency_ranks,
    sample_by_frequency,
    sample_stratified_by_length
)

# String operations
from .string_operations import (
    string_to_chars,
    chars_to_string,
    normalize_diacritic,
    diacritize,
    spell_string
)

# Utility functions
from .utils import (
    prepare_mcq_outputs,
    prepare_gen_outputs,
    validate_dataframe_columns,
    validate_positive_integer,
    validate_probability
)

# Hierarchical task framework
from .hierarchical_tasks import HierarchicalTask, HierarchicalTaskGenerator
from .hierarchical_analysis import (
    HierarchicalAnalyzer,
    compare_multiple_models,
    visualize_capability_profile
)

__all__ = [
    # Dataset creation
    'create_affixation_dataset',
    'create_composition_dataset',
    'create_manipulation_dataset',
    'create_syllabification_dataset',

    # Syllabification
    'syllabify',
    'normalize_text',

    # Sampling
    'load_frequency_data',
    'add_frequency_ranks',
    'sample_by_frequency',
    'sample_stratified_by_length',

    # String operations
    'string_to_chars',
    'chars_to_string',
    'normalize_diacritic',
    'diacritize',
    'spell_string',

    # Utilities
    'prepare_mcq_outputs',
    'prepare_gen_outputs',
    'validate_dataframe_columns',
    'validate_positive_integer',
    'validate_probability',

    # Hierarchical framework
    'HierarchicalTask',
    'HierarchicalTaskGenerator',
    'HierarchicalAnalyzer',
    'compare_multiple_models',
    'visualize_capability_profile',
]
