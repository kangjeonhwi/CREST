import os
import random
import torch
from tqdm import tqdm
from typing import List, Dict, Optional
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer

import configs as cfg

class DatasetLoader:
    @staticmethod
    def load_gsm8k(split: str = "train", sample_size: int = None) -> List[Dict]:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        problems = [{"question": item["question"], "dataset": "GSM8K"} for item in dataset]
        return DatasetLoader._sample(problems, sample_size)

    @staticmethod
    def load_math_dataset(
        levels: List[int] = [4, 5],
        split: str = "train",
        sample_size: int = None,
    ) -> List[Dict]:
        try:
            configs = get_dataset_config_names("EleutherAI/hendrycks_math")
        except Exception:
            configs = [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ]

        problems = []
        for config_name in configs:
            try:
                subset = load_dataset("EleutherAI/hendrycks_math", config_name, split=split)
                filtered = [
                    {"question": item["problem"], "dataset": "MATH"}
                    for item in subset
                    if item["level"] in [f"Level {l}" for l in levels]
                ]
                problems.extend(filtered)
            except Exception as e:
                print(f"[Warning] Failed to load MATH subset '{config_name}': {e}")

        return DatasetLoader._sample(problems, sample_size)

    @staticmethod
    def load_aime_dataset(sample_size: int = None) -> List[Dict]:
        try:
            dataset = load_dataset("gneubig/aime-1983-2024", split="train")
            problems = [{"question": item["Question"], "dataset": "AIME"} for item in dataset]
            return DatasetLoader._sample(problems, sample_size)
        except Exception as e:
            print(f"[Error] Failed to load AIME dataset: {e}")
            return []

    @staticmethod
    def _sample(data: List[Dict], size: Optional[int]) -> List[Dict]:
        if size and size < len(data):
            random.seed(42)
            return random.sample(data, size)
        return data


class SteeringVectorExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing Model: {cfg.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()

        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()

            return hook

        for i, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))

    def _clear_activations(self):
        self.activations = {}

    def format_input(self, problem: str, system_msg: str) -> str:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": problem}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def process_and_save(self, problems: List[Dict]):
        naive_style = cfg.REASONING_STYLES["Naive"]
        target_styles = [k for k in cfg.REASONING_STYLES.keys() if k != "Naive"]

        style_diff_sums = {style: {} for style in target_styles}
        sample_counts = {style: 0 for style in target_styles}

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        for style_name in target_styles:
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, f"{style_name}_vectors"), exist_ok=True)

        buffers = {style: [] for style in target_styles}
        chunk_idxs = {style: 0 for style in target_styles}

        limit = cfg.DEBUG_SAMPLES if cfg.DEBUG_SAMPLES is not None else len(problems)
        print(f"Processing {limit} problems for {len(target_styles)} styles...")

        for sample_idx, item in enumerate(tqdm(problems[:limit], desc="Extracting Vectors")):
            question = item["question"]

            try:
                naive_text = self.format_input(question, naive_style["system_message"])
                for style_name in target_styles:
                    style_text = self.format_input(
                        question, cfg.REASONING_STYLES[style_name]["system_message"]
                    )

                    batch_texts = [naive_text, style_text]
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=cfg.MAX_MODEL_LEN,
                    ).to(self.model.device)

                    self._clear_activations()
                    with torch.no_grad():
                        self.model(**inputs)

                    sorted_layers = sorted(
                        self.activations.keys(), key=lambda x: int(x.split("_")[1])
                    )

                    stacked_vecs = []
                    for layer_name in sorted_layers:
                        act_tensor = self.activations[layer_name]
                        last_token_acts = act_tensor[:, -1, :].cpu()
                        diff = (last_token_acts[1] - last_token_acts[0]).float()
                        stacked_vecs.append(diff)

                        if layer_name not in style_diff_sums[style_name]:
                            style_diff_sums[style_name][layer_name] = diff
                        else:
                            style_diff_sums[style_name][layer_name] += diff

                    per_sample_tensor = torch.stack(stacked_vecs)

                    buffers[style_name].append(
                        {
                            "sample_idx": sample_idx,
                            "dataset": item.get("dataset", None),
                            "vector": per_sample_tensor,
                        }
                    )

                    sample_counts[style_name] += 1

                    if cfg.SAVE_INTERVAL and (sample_counts[style_name] % cfg.SAVE_INTERVAL == 0):
                        self._flush_style_buffer(style_name, buffers, chunk_idxs)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM detected. Skipping problem.")
                    torch.cuda.empty_cache()
                    continue
                raise e

        for style_name in target_styles:
            self._flush_style_buffer(style_name, buffers, chunk_idxs, force=True)

        self._save_vectors(style_diff_sums, sample_counts, target_styles)

    def _flush_style_buffer(self, style_name, buffers, chunk_idxs, force: bool = False):
        if (not force) and (len(buffers[style_name]) == 0):
            return
        if force and (len(buffers[style_name]) == 0):
            return

        out_dir = os.path.join(cfg.OUTPUT_DIR, f"{style_name}_vectors")
        chunk_path = os.path.join(out_dir, f"chunk_{chunk_idxs[style_name]:06d}.pt")
        torch.save(buffers[style_name], chunk_path)
        buffers[style_name].clear()
        chunk_idxs[style_name] += 1

    def _save_vectors(self, diff_sums, counts, styles):
        print("Saving steering vectors...")

        for style in styles:
            count = counts[style]
            if count == 0:
                continue

            sorted_layers = sorted(diff_sums[style].keys(), key=lambda x: int(x.split("_")[1]))

            stacked_vecs = []
            for layer in sorted_layers:
                mean_vec = diff_sums[style][layer] / count
                stacked_vecs.append(mean_vec)

            final_tensor = torch.stack(stacked_vecs)
            save_path = os.path.join(cfg.OUTPUT_DIR, f"steering_vec_{style}.pt")
            torch.save(final_tensor, save_path)
            print(f"Saved: {save_path} | Shape: {final_tensor.shape}")

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    random.seed(cfg.SEED)

    print("Loading datasets...")
    problems = (
        DatasetLoader.load_gsm8k(sample_size=cfg.SAMPLE_SIZE_GSM8K)
        + DatasetLoader.load_math_dataset(sample_size=cfg.SAMPLE_SIZE_MATH)
        + DatasetLoader.load_aime_dataset(sample_size=cfg.SAMPLE_SIZE_AIME)
    )

    random.seed(cfg.SEED)
    random.shuffle(problems)

    extractor = SteeringVectorExtractor()
    try:
        extractor.process_and_save(problems)
    finally:
        extractor.cleanup()