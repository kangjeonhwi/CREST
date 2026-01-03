import os
import sys
import json
import torch
import itertools
import warnings
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root for local imports
PROJECT_ROOT = "/home/kangjh/CREST/benchmark"
sys.path.append(PROJECT_ROOT)

from configs import RANDOM_SEED
from src.data_loader import DataLoader
from src.utils import MathVerifier

# Suppress noisy SymPy/Antlr warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sympy")

# =========================
# Config
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
AVAILABLE_GPUS = [4, 5, 6, 7]

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
VECTOR_PATH = "./steering_vector/error_vectors_norm.pt"
OUTPUT_DIR = './src/analysis/results/error_grid'
TARGET_LAYERS = [8, 12, 16, 20, 24]
TARGET_VECTORS = [0, 1, 2, 3]
ALPHAS = [-15, -10, -8, -5, -2, -1, 0, 1, 2, 5, 8, 10, 15]

GREEDY_PARAMS = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "pad_token_id": 151643,
    "eos_token_id": [151645, 151643],
}

PROMPT_TEMPLATE = {
    "system_message": (
        "You are a helpful and logical AI assistant. "
        "Solve the mathematical problem step by step to ensure accuracy. "
        "You must enclose your final answer in \\boxed{}."
    )
}

# =========================
# Steering
# =========================
class SteeringController:
    def __init__(self, model, vector_path, device):
        self.model = model
        self.vectors = torch.load(vector_path, map_location="cpu")
        self.hooks = []
        self.device = device

    def apply_steering(self, layer_idx, vector_idx, alpha):
        # Ensure only one active hook at a time
        self.reset_steering()

        layer_key = str(layer_idx)
        if layer_key not in self.vectors:
            return False

        module = self.model.model.layers[layer_idx]
        target_vector = self.vectors[layer_key][vector_idx].to(self.device).to(self.model.dtype)

        def hook_fn(module, input, output):
            # Add steering vector to the layer output hidden state
            if isinstance(output, tuple):
                return (output[0] + (alpha * target_vector),) + output[1:]
            return output + (alpha * target_vector)

        self.hooks.append(module.register_forward_hook(hook_fn))
        return True

    def reset_steering(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

# =========================
# Worker
# =========================
def worker_process(cuda_id, original_gpu_id, config_subset, raw_data, output_file):
    torch.cuda.set_device(cuda_id)
    device = torch.device(f"cuda:{cuda_id}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Decoder-only batched generation requires left padding
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = GREEDY_PARAMS["pad_token_id"]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    steering = SteeringController(model, VECTOR_PATH, device)

    # Prebuild chat-formatted prompts
    prompts = []
    for item in raw_data:
        messages = [
            {"role": "system", "content": PROMPT_TEMPLATE["system_message"]},
            {"role": "user", "content": item["question"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    batch_size = 24

    with open(output_file, "w", encoding="utf-8") as f:
        for config in tqdm(config_subset, desc=f"GPU {original_gpu_id}", position=cuda_id):
            layer, vec_idx, alpha = config

            results = []
            correct_count = 0

            try:
                if not steering.apply_steering(layer, vec_idx, alpha):
                    continue

                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i : i + batch_size]
                    batch_data = raw_data[i : i + batch_size]

                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                    input_len = inputs.input_ids.shape[1]

                    with torch.inference_mode():
                        outputs = model.generate(**inputs, **GREEDY_PARAMS)

                    generated_tokens = outputs[:, input_len:]
                    decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                    for j, response in enumerate(decoded_responses):
                        gold_answer = MathVerifier.extract_answer(str(batch_data[j]["gold"]))
                        extracted_pred = MathVerifier.extract_answer(response)
                        is_correct = MathVerifier.is_equivalent(extracted_pred, gold_answer)

                        if is_correct:
                            correct_count += 1

                        results.append(
                            {
                                "id": i + j,
                                "question": batch_data[j]["question"],
                                "gold_raw": gold_answer,
                                "pred_raw": extracted_pred,
                                "full_response": response,
                                "is_correct": float(is_correct),
                            }
                        )

                    # Reduce fragmentation pressure
                    del inputs, outputs, generated_tokens, decoded_responses

            except Exception as e:
                print(f"[GPU {original_gpu_id}] Error (L{layer}, V{vec_idx}, A{alpha}): {e}")
                torch.cuda.empty_cache()

            finally:
                steering.reset_steering()

            log_entry = {
                "config": {"layer": layer, "vector": vec_idx, "alpha": alpha},
                "metrics": {"accuracy": correct_count / len(raw_data) if raw_data else 0},
                "details": results,
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            f.flush()

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"steering_experiment_MATH_Robust_{timestamp}"

    loader = DataLoader(seed=RANDOM_SEED)
    raw_data = loader.load_math(level_filter="Level 4", limit=50) + loader.load_math(level_filter="Level 5", limit=50)
    print(f"Loaded {len(raw_data)} samples.")

    all_configs = list(itertools.product(TARGET_LAYERS, TARGET_VECTORS, ALPHAS))
    chunks = [all_configs[i :: len(AVAILABLE_GPUS)] for i in range(len(AVAILABLE_GPUS))]

    mp.set_start_method("spawn", force=True)
    processes = []

    for rank, original_gpu_id in enumerate(AVAILABLE_GPUS):
        output_file = os.path.join(OUTPUT_DIR, f"{base_filename}_part_{original_gpu_id}.jsonl")
        p = mp.Process(
            target=worker_process,
            args=(rank, original_gpu_id, chunks[rank], raw_data, output_file),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_output = os.path.join(OUTPUT_DIR, f"{base_filename}_merged.jsonl")
    with open(final_output, "w", encoding="utf-8") as outfile:
        for gpu_id in AVAILABLE_GPUS:
            part_file = os.path.join(OUTPUT_DIR, f"{base_filename}_part_{gpu_id}.jsonl")
            if not os.path.exists(part_file):
                continue
            with open(part_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

    print(f"Saved: {final_output}")

if __name__ == "__main__":
    main()