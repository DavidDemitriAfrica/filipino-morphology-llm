import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import pandas as pd
from pacute.affixation import create_affixation_dataset

os.makedirs("tasks", exist_ok=True)
inflections = pd.read_excel("exploration/inflections.xlsx", sheet_name="data")

print("Generating MCQ affixation dataset...")
mcq_dataset = create_affixation_dataset(inflections, mode='mcq', random_seed=1859)
mcq_dataset.to_json("tasks/mcq_affixation.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_dataset)} samples")

print("\nGenerating GEN affixation dataset...")
gen_reverse_dataset = create_affixation_dataset(inflections, mode='gen', random_seed=1859)
gen_reverse_dataset.to_json("tasks/gen_affixation.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_reverse_dataset)} samples")

print("\nAll affixation datasets generated successfully!")
