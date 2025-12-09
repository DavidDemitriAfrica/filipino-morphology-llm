import random
import json
import os

def load_langgame(split, **kwargs):
    """Load and process the benchmark from JSONL file"""
    current_file_path = os.path.abspath(__file__)
    dir_of_file = os.path.dirname(current_file_path)
    jsonl_path = os.path.join(dir_of_file, f"../../../data/benchmarks/langgame_{split}.jsonl")
    
    # Load all samples from JSONL
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"LangGame: Loaded {len(samples)} examples from {split} split.")
    
    # Shuffle indices
    index = list(range(len(samples)))
    random.shuffle(index)
    
    for i in index:
        sample = samples[i]
        prefix = sample["question"]
        options = sample["options"]
        ground_truth = options[0]
        false_options = options[1:]
        yield prefix, ground_truth, false_options
