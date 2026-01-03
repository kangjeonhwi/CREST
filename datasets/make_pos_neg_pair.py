import json
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def process_prm_to_pairs(output_path="prm800k_pairwise_correction.json"):
    print(">>> 1. Loading trl-lib/prm800k dataset...")
    ds = load_dataset("trl-lib/prm800k", split="train")
    df = ds.to_pandas()
    processed_samples = []

    print(">>> 2. Grouping by prompts and extracting pairs...")
    grouped = df.groupby('prompt')

    for prompt, group in tqdm(grouped, total=len(grouped), desc="Processing Prompts"):
        valid_next_steps = {}

        for _, row in group.iterrows():
            # Ensure Python list types for safe iteration
            steps = list(row['completions']) if not isinstance(row['completions'], list) else row['completions']
            labels = list(row['labels']) if not isinstance(row['labels'], list) else row['labels']

            # Skip malformed rows
            if len(steps) != len(labels):
                continue

            current_prefix = []
            for step, label in zip(steps, labels):
                # Normalize label to boolean correctness
                is_correct = bool(label == 1.0 or label == True or label == 1)

                # Collect valid next steps for each prefix
                if is_correct:
                    prefix_tuple = tuple(current_prefix)
                    if prefix_tuple not in valid_next_steps:
                        valid_next_steps[prefix_tuple] = set()
                    valid_next_steps[prefix_tuple].add(step)

                current_prefix.append(step)

        for _, row in group.iterrows():
            steps = list(row['completions']) if not isinstance(row['completions'], list) else row['completions']
            labels = list(row['labels']) if not isinstance(row['labels'], list) else row['labels']

            if len(steps) != len(labels):
                continue

            for i, (step, label) in enumerate(zip(steps, labels)):
                is_correct = bool(label == 1.0 or label == True or label == 1)

                # For an incorrect step, find possible correct replacements under the same prefix
                if not is_correct:
                    prefix = steps[:i]
                    bad_step = step

                    neg_seq = prefix + [bad_step]
                    prefix_tuple = tuple(prefix)
                    possible_fixes = valid_next_steps.get(prefix_tuple, [])

                    pos_list = []
                    for fix_step in possible_fixes:
                        pos_seq = prefix + [fix_step]
                        pos_list.append(pos_seq)

                    sample = {
                        "prompt": str(prompt),
                        "completions_neg": neg_seq,
                        "completions_pos": pos_list,
                        "has_pos": int(len(pos_list) > 0)
                    }
                    processed_samples.append(sample)

    print(f">>> 3. Calculating statistics...")

    total_samples = len(processed_samples)
    has_pos_count = sum(s['has_pos'] for s in processed_samples)
    avg_len = float(np.mean([len(s['completions_neg']) for s in processed_samples])) if total_samples > 0 else 0.0

    print(f"==========================================")
    print(f"Total Generated Samples    : {total_samples}")
    print(f"Samples with Correction(s) : {has_pos_count} ({has_pos_count/total_samples*100:.2f}%)")
    print(f"Average Step Length        : {avg_len:.2f}")
    print(f"==========================================")

    print(f">>> 4. Saving to {output_path}...")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    process_prm_to_pairs(output_path="./datasets/pos_neg_dataset/prm800k_pairwise_correction.json")
