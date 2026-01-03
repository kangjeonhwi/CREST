import os
import re
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# --- Configuration ---
@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_model_len: int = 4096
    output_dir: str = "./math_reasoning_v3_robust"
    seed: int = 42
    sample_size_gsm8k: int = 1000
    sample_size_math: int = 1000
    sample_size_aime: int = 50

# --- Prompt Templates ---
REASONING_STYLES = {
    "Naive": {
        "system_message": "You are a helpful and logical AI assistant. Think step by step and solve this problem carefully. Show all your work. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Standard": {
        "system_message": "You are a helpful and logical AI assistant. Solve this problem using the most standard, textbook academic method. Be formal and concise. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Visual": {
        "system_message": "You are a helpful and logical AI assistant. Solve this problem by visualizing it. Use geometric interpretations, number lines, or draw mental diagrams. Avoid abstract algebra if possible and rely on spatial reasoning. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Programmatic": {
        "system_message": "You are a helpful and logical AI assistant. Think like a programmer. Break the problem down into an algorithm or pseudocode. Use logical steps like 'Initialize', 'Loop', 'Condition' to explain the solution process. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Reverse": {
        "system_message": "You are a helpful and logical AI assistant. Use 'Working Backward' strategy. Start from what you want to find or assume the answer, and work your way back to the given conditions to verify logically. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Analogical": {
        "system_message": "You are a helpful and logical AI assistant. Use analogies or real-world metaphors. Translate the abstract numbers into concrete objects (e.g., apples, money, distance) to make the logic intuitive. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "First_Principles": {
        "system_message": "You are a helpful and logical AI assistant. Reason from first principles. Do not use any memorized formulas. Derive every step logically from the basic definitions of the numbers and operations involved. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Pattern": {
        "system_message": "You are a helpful and logical AI assistant. Use inductive reasoning. Test the logic with smaller, simpler numbers to find a pattern, then generalize that pattern to solve the specific problem. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Didactic": {
        "system_message": "You are a helpful and logical AI assistant. Explain it as if you are a kind tutor teaching a confused student. Use conversational language, ask rhetorical questions, and guide the thought process gently. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Intuitive": {
        "system_message": "You are a helpful and logical AI assistant. Start with a strong intuitive guess or estimation (Fermi method) to bound the answer, then refine it with precise calculation. Focus on the 'number sense'. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    },
    "Decomposition": {
        "system_message": "You are a helpful and logical AI assistant. Break the problem into a numbered list of independent sub-problems. Solve each sub-problem explicitly and combine them for the final result. Solve the mathematical problem step by step to ensure accuracy. You must enclose your final answer in \\boxed{}."
    }
}

# --- Data Loading ---
class DatasetLoader:
    @staticmethod
    def load_gsm8k(split: str = "train", sample_size: int = None) -> List[Dict]:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        problems = [{'question': item['question'], 'dataset': 'GSM8K'} for item in dataset]
        return DatasetLoader._sample(problems, sample_size)
    
    @staticmethod
    def load_math_dataset(levels: List[int] = [4, 5], split: str = "train", sample_size: int = None) -> List[Dict]:
        try:
            configs = get_dataset_config_names("EleutherAI/hendrycks_math")
        except Exception:
            configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
            
        problems = []
        for config_name in configs:
            try:
                subset = load_dataset("EleutherAI/hendrycks_math", config_name, split=split)
                filtered = [
                    {'question': item['problem'], 'dataset': 'MATH'} 
                    for item in subset if item['level'] in [f"Level {l}" for l in levels]
                ]
                problems.extend(filtered)
            except Exception as e:
                print(f"[Warning] Failed to load MATH subset '{config_name}': {e}")
        
        return DatasetLoader._sample(problems, sample_size)
    
    @staticmethod
    def load_aime_dataset(sample_size: int = None) -> List[Dict]:
        try:
            dataset = load_dataset("gneubig/aime-1983-2024", split="train")
            problems = [{'question': item['Question'], 'dataset': 'AIME'} for item in dataset]
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

# --- Core Logic: Steering Vector Extraction ---
class SteeringVectorExtractor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left' 
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        ).eval()
        
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Detach to save memory and avoid graph computation
                self.activations[name] = output.detach()
            return hook

        for i, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))

    def _clear_activations(self):
        self.activations = {}

    def format_input(self, problem: str, system_msg: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": problem}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def process_and_save(self, problems: List[Dict]):
        naive_style = REASONING_STYLES["Naive"]
        target_styles = [k for k in REASONING_STYLES.keys() if k != "Naive"]
        
        style_diff_sums = {style: {} for style in target_styles}
        sample_counts = {style: 0 for style in target_styles}
        
        print(f"Processing {len(problems)} problems for {len(target_styles)} styles...")

        for item in tqdm(problems, desc="Extracting Vectors"):
            question = item['question']
            
            try:
                naive_text = self.format_input(question, naive_style["system_message"])
                
                for style_name in target_styles:
                    style_text = self.format_input(question, REASONING_STYLES[style_name]["system_message"])
                    
                    # Batch: [Naive, Style]
                    batch_texts = [naive_text, style_text]
                    
                    inputs = self.tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True,       
                        truncation=True,  
                        max_length=self.config.max_model_len
                    ).to(self.model.device)

                    self._clear_activations()
                    
                    with torch.no_grad():
                        self.model(**inputs)
                    
                    # Compute Difference Vector
                    # With left-padding, -1 index aligns to the last token of the prompt for both sequences
                    for layer_name, act_tensor in self.activations.items():
                        last_token_acts = act_tensor[:, -1, :].cpu() # Shape: [2, Hidden_Dim]
                        diff = (last_token_acts[1] - last_token_acts[0]).float()
                        
                        if layer_name not in style_diff_sums[style_name]:
                            style_diff_sums[style_name][layer_name] = diff
                        else:
                            style_diff_sums[style_name][layer_name] += diff
                    
                    sample_counts[style_name] += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM detected. Skipping problem.")
                    torch.cuda.empty_cache()
                    continue
                raise e

        self._save_vectors(style_diff_sums, sample_counts, target_styles)

    def _save_vectors(self, diff_sums, counts, styles):
        os.makedirs(self.config.output_dir, exist_ok=True)
        print("Saving steering vectors...")
        
        for style in styles:
            count = counts[style]
            if count == 0:
                continue
                
            # Sort layers numerically (layer_0, layer_1...)
            sorted_layers = sorted(diff_sums[style].keys(), key=lambda x: int(x.split('_')[1]))
            
            stacked_vecs = []
            for layer in sorted_layers:
                mean_vec = diff_sums[style][layer] / count
                stacked_vecs.append(mean_vec)
            
            # Tensor Shape: [Num_Layers, Hidden_Dim]
            final_tensor = torch.stack(stacked_vecs)
            save_path = os.path.join(self.config.output_dir, f"steering_vec_{style}.pt")
            torch.save(final_tensor, save_path)
            print(f"Saved: {save_path} | Shape: {final_tensor.shape}")

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure CUDA devices are set before anything else if needed
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    config = ExperimentConfig()
    
    # Load and Mix Datasets
    print("Loading datasets...")
    problems = (
        DatasetLoader.load_gsm8k(sample_size=config.sample_size_gsm8k) + 
        DatasetLoader.load_math_dataset(sample_size=config.sample_size_math) +
        DatasetLoader.load_aime_dataset(sample_size=config.sample_size_aime)
    )
    
    random.seed(config.seed)
    random.shuffle(problems)
    
    extractor = SteeringVectorExtractor(config)
    try:
        extractor.process_and_save(problems)
    finally:
        extractor.cleanup()