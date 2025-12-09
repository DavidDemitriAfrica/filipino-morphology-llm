"""
Load a benchmark loader, given the benchmark name.
"""
from functools import partial

from evaluation.loaders.arc import load_arc
from evaluation.loaders.blimp import load_blimp
from evaluation.loaders.hellaswag import load_hellaswag
from evaluation.loaders.mmlu import load_mmlu
from evaluation.loaders.winogrande import load_winogrande
from evaluation.loaders.langgame import load_langgame
from evaluation.loaders.cute import load_cute
from evaluation.loaders.pacute import load_pacute
from evaluation.loaders.multi_digit_addition import load_multi_digit_addition

EVALS_DICT = {
    "arc": partial(load_arc, split="test"),
    "winograd": partial(load_winogrande, split="test"),
    "mmlu": partial(load_mmlu, split="test"),
    "hellaswag": partial(load_hellaswag, split="test"),
    "blimp": partial(load_blimp, split="test"),
    "langgame-train": partial(load_langgame, split="train"),
    "langgame-val": partial(load_langgame, split="val"),
    "cute": partial(load_cute, split="test", max_per_task=100),  # Subsample to 100 per task (1400 total)
    "multi-digit-addition": partial(load_multi_digit_addition, split="val", max_samples=1000),
    "pacute": partial(load_pacute, split="test"),
    "pacute-affixation": partial(load_pacute, split="test", categories=["affixation"]),
    "pacute-composition": partial(load_pacute, split="test", categories=["composition"]),
    "pacute-manipulation": partial(load_pacute, split="test", categories=["manipulation"]),
    "pacute-syllabification": partial(load_pacute, split="test", categories=["syllabification"]),
}


def load_benchmark(benchmark_name):
    """
    Given the benchmark name, build the benchmark
    """
    return EVALS_DICT[benchmark_name]()
