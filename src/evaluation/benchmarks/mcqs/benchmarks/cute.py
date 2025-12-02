"""
CUTE: Character Understanding Test Evaluation

Tests character-level understanding through composition tasks (spelling, counting).
"""
import json
import random
import os


def load_cute(split="test", **kwargs):
    """Load CUTE (Character Understanding) benchmark - composition tasks only."""
    # Find project root by looking for data/benchmarks directory
    current_file_path = os.path.abspath(__file__)
    search_dir = current_file_path

    for _ in range(10):  # Search up to 10 levels
        search_dir = os.path.dirname(search_dir)
        potential_file = os.path.join(search_dir, "data/benchmarks/mcq_composition.jsonl")
        if os.path.exists(potential_file):
            mcq_file = potential_file
            break
    else:
        raise FileNotFoundError("Could not find data/benchmarks/mcq_composition.jsonl")


    tasks = []
    with open(mcq_file) as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)

    print(f"CUTE: Loaded {len(tasks)} composition (character understanding) tasks.")

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    for i in indices:
        task = tasks[i]
        # Get the English prompt
        prompt_data = task["prompts"][0]
        prefix = prompt_data["text_en"]

        # Extract options
        mcq_options = prompt_data["mcq_options"]
        ground_truth = mcq_options["correct"]
        false_options = [
            mcq_options["incorrect1"],
            mcq_options["incorrect2"],
            mcq_options["incorrect3"]
        ]

        yield prefix, ground_truth, false_options
