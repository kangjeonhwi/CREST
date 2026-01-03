# main.py
import os
import json
import gc
import shutil
import logging
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.extract_vector.configs as cfg

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

class SteeringExtractor:
    def __init__(self, model_name, system_prompt, device="cuda", debug_mode=False):
        self.device = device
        self.system_prompt = system_prompt
        self.debug_mode = debug_mode

        logger.info(f"Loading tokenizer/model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # offsets_mapping requires a fast tokenizer
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Tokenizer must be a fast tokenizer to use return_offsets_mapping=True.")  # [web:7]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Capture per-layer hidden states from each transformer block via forward hooks
        def get_activation(name):
            def hook(_module, _input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self.activations[name] = hidden.detach()
            return hook

        for name, layer in self.model.model.layers.named_children():
            self.hooks.append(layer.register_forward_hook(get_activation(name)))

        logger.info(f"Registered {len(self.hooks)} forward hooks.")

    def clear_activations(self):
        self.activations = {}

    def get_last_step_activations(self, prompt, previous_steps, target_step, debug_label=""):
        # Build a chat-formatted text and locate the token span for target_step
        self.clear_activations()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if previous_steps:
            assistant_prefix = " ".join(previous_steps)
            assistant_full = assistant_prefix + " " + target_step
        else:
            assistant_prefix = ""
            assistant_full = target_step

        full_messages = messages + [{"role": "assistant", "content": assistant_full}]
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        full_encoding = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        full_tokens = full_encoding["input_ids"]
        offset_mapping = full_encoding["offset_mapping"][0]

        search_string = (" " + target_step) if previous_steps else target_step
        target_start_char = full_text.rfind(search_string)
        if target_start_char == -1:
            target_start_char = full_text.rfind(target_step)
            search_string = target_step

        if target_start_char == -1:
            raise ValueError("Failed to locate target_step span in the formatted chat text.")

        target_end_char = target_start_char + len(search_string)

        start_idx = None
        end_idx = None
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start_idx is None and start >= target_start_char:
                start_idx = i
            if start_idx is not None and end >= target_end_char:
                end_idx = i + 1
                break

        if start_idx is None or end_idx is None:
            # Fallback to last token span if alignment fails
            start_idx = full_tokens.shape[1] - 1
            end_idx = full_tokens.shape[1]

        if self.debug_mode:
            target_token_ids = full_tokens[0, start_idx:end_idx]
            decoded_target = self.tokenizer.decode(target_token_ids)
            logger.info(f"{debug_label} | token_span=[{start_idx}:{end_idx}] | decoded='{decoded_target[:120]}'")

        inputs = {"input_ids": full_tokens.to(self.device)}
        if "attention_mask" in full_encoding:
            inputs["attention_mask"] = full_encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        # Mean pooling over the token span for each layer
        layer_means = {}
        for layer_name, hidden_states in self.activations.items():
            if hidden_states.dim() == 3:
                target_states = hidden_states[0, start_idx:end_idx, :]
            else:
                target_states = hidden_states[start_idx:end_idx, :]

            if target_states.shape[0] == 0:
                target_states = hidden_states[0, -1:, :] if hidden_states.dim() == 3 else hidden_states[-1:, :]

            layer_means[layer_name] = torch.mean(target_states, dim=0).cpu()

        return layer_means


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    temp_dir = os.path.join(cfg.OUTPUT_DIR, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    output_filename = "error_vectors_qwen.pt"
    full_output_path = os.path.join(cfg.OUTPUT_DIR, output_filename)

    logger.info(f"Loading dataset: {cfg.DATASET_PATH}")
    with open(cfg.DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    extractor = SteeringExtractor(cfg.MODEL_NAME, cfg.SYSTEM_MESSAGE, debug_mode=False)

    logger.info("Start extraction loop.")
    buffer_data = []
    chunk_paths = []

    for idx, sample in tqdm(enumerate(data), total=len(data)):
        try:
            if sample.get("has_pos", 0) == 0:
                continue

            prompt = sample["prompt"]
            neg_chain = sample["completions_neg"]
            pos_chains = sample["completions_pos"]

            prefix_steps = neg_chain[:-1]
            bad_step = neg_chain[-1]

            neg_acts = extractor.get_last_step_activations(prompt, prefix_steps, bad_step)

            pos_acts_list = []
            for pos_chain in pos_chains:
                good_step = pos_chain[-1]
                pos_acts_list.append(extractor.get_last_step_activations(prompt, prefix_steps, good_step))

            steering_vectors = {}
            for layer_name in pos_acts_list[0].keys():
                neg_vec = neg_acts[layer_name]
                pos_stack = torch.stack([d[layer_name] for d in pos_acts_list])
                mean_pos = torch.mean(pos_stack, dim=0)
                steering_vectors[layer_name] = (neg_vec - mean_pos)

            buffer_data.append(
                {
                    "sample_idx": idx,
                    "prompt": prompt,
                    "error_step": bad_step,
                    "steering_vectors": steering_vectors,
                }
            )

            del neg_acts, pos_acts_list, steering_vectors, pos_stack, mean_pos
            if len(buffer_data) >= cfg.SAVE_INTERVAL:
                chunk_idx = len(chunk_paths)
                chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.pt")
                os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
                torch.save(buffer_data, chunk_path)
                chunk_paths.append(chunk_path)
                buffer_data = []
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"Checkpoint saved: chunk_{chunk_idx}.pt")

        except Exception as e:
            logger.warning(f"Skip sample {idx}: {e}")
            continue

    if buffer_data:
        chunk_idx = len(chunk_paths)
        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.pt")
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        torch.save(buffer_data, chunk_path)
        chunk_paths.append(chunk_path)
        logger.info(f"Final checkpoint saved: chunk_{chunk_idx}.pt")

    logger.info(f"Merging {len(chunk_paths)} chunks.")
    final_extracted = []
    for cp in chunk_paths:
        chunk = torch.load(cp)
        final_extracted.extend(chunk)
        del chunk
        gc.collect()

    logger.info(f"Saving final output: {full_output_path}")
    torch.save(final_extracted, full_output_path)

    logger.info("Cleaning up temp files.")
    shutil.rmtree(temp_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
