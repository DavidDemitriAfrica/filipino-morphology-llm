"""
Multi-digit addition benchmark

Tests arithmetic capabilities through addition problems.
Format: "X+Y=" → "Z"
"""
import random
import json
import os


def load_multi_digit_addition(split="val", max_samples=1000, **kwargs):
    """
    Load multi-digit addition benchmark from JSONL file.

    Args:
        split: "train" or "val" (default: "val")
        max_samples: Maximum number of examples to load (default: 1000)

    Yields:
        For MCQ format compatibility:
        - prefix: The question (e.g., "840+425=")
        - ground_truth: The correct answer (e.g., "1265")
        - false_options: Empty list (generative task, not MCQ)

    Note: This is a generative benchmark, not MCQ. The false_options will be empty.
    """
    current_file_path = os.path.abspath(__file__)
    dir_of_file = os.path.dirname(current_file_path)
    jsonl_path = os.path.join(dir_of_file, f"../../../data/benchmarks/multi_digit_addition_{split}.jsonl")
    
    # Load all samples from JSONL
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if max_samples is not None and len(samples) >= max_samples:
                break
    
    print(f"Multi-digit Addition: Loaded {len(samples)} examples from {split} split.")
    print(f"Note: This is a generative benchmark (question → answer), not MCQ format.")
    
    # Shuffle samples
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    for i in indices:
        sample = samples[i]
        prefix = sample["question"]
        ground_truth = sample["answer"]
        false_options = []  # Generative task, no MCQ options
        
        yield prefix, ground_truth, false_options
