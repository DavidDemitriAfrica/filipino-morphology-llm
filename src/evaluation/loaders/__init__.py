"""
Benchmark loaders for loading evaluation datasets.
"""

from .pacute import load_pacute
from .registry import load_benchmark, EVALS_DICT

__all__ = [
    "load_pacute",
    "load_benchmark",
    "EVALS_DICT",
]
