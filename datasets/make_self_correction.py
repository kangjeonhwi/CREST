import json
import os
import random
from tqdm import tqdm


def generate_self_correction_dataset(
    input_path="./datasets/pos_neg_dataset/prm800k_pairwise_correction.json",
    output_path="./datasets/pos_neg_dataset/prm800k_self_correction.json"
):
    # Fixed set of self-correction trigger phrases (English prompts)
    correction_triggers = [
        "Wait, I might be wrong.",
        "Hold on, let me double check that.",
        "Actually, that calculation seems incorrect.",
        "Let me re-evaluate the previous step.",
        "Wait, that's not right. Let me think again.",
        "Upon reviewing, I found an error.",
        "Correction:",
        "Let me try that step again.",
        "Wait, I made a mistake in the logic.",
        "Let's backtrack and correct this.",
        "The previous deduction implies a contradiction. Let me fix it.",
        "Hold on, I missed a detail.",
        "Let's verify this calculation. No, that's wrong.",
        "Wait, the result should be different.",
        "Let me fix the last step.",
        "Looking back, I made an error.",
        "That step is invalid. Let me correct it.",
        "Oops, let me correct that.",
        "Let's reconsider the last step.",
        "Wait, let me take a closer look."
    ]

    print(f">>> 1. Loading input dataset from {input_path}...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f">>> 2. Generating contrastive self-correction pairs...")

    new_dataset = []

    # Stats counters
    cnt_total_input = len(raw_data)
    cnt_generated_pairs = 0

    for sample in tqdm(raw_data, desc="Processing samples"):
        # Skip samples without any valid positive correction candidates
        if sample['has_pos'] == 0:
            continue

        neg_seq = sample['completions_neg']
        context_steps = neg_seq

        pos_candidates = sample['completions_pos']
        valid_corrections = [p[-1] for p in pos_candidates]

        if not valid_corrections:
            continue

        chosen_correction = random.choice(valid_corrections)
        chosen_trigger = random.choice(correction_triggers)

        new_sample = {
            "prompt": sample['prompt'],
            "context_steps": context_steps,
            "continuation_neg": "",
            "continuation_pos": f" {chosen_trigger} {chosen_correction}",
            "meta": {
                "bad_step": context_steps[-1],
                "trigger": chosen_trigger,
                "correction_step": chosen_correction
            }
        }

        new_dataset.append(new_sample)
        cnt_generated_pairs += 1

    print(f"==========================================")
    print(f"Total Input Samples     : {cnt_total_input}")
    print(f"Generated Pairs         : {cnt_generated_pairs}")
    print(f"==========================================")

    print(f">>> 3. Saving to {output_path}...")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)

    generate_self_correction_dataset(
        input_path="./datasets/pos_neg_dataset/prm800k_pairwise_correction.json",
        output_path="./datasets/pos_neg_dataset/prm800k_self_correction.json"
    )
