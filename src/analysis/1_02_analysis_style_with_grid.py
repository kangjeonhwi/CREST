import json
import os
import re
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    num_trials_per_problem: int = 1
    temperature: float = 0.0
    max_tokens: int = 2048
    output_dir: str = "./results/01_style_generated/prompt"
    seed: int = 42

REASONING_STYLES = {
    "Naive": {
        "name": "Naive",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Think step by step and solve this problem carefully. Show all your work. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Standard": {
        "name": "Standard",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Solve this problem using the most standard, textbook academic method. Be formal and concise. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Visual": {
        "name": "Visual",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Solve this problem by visualizing it. Use geometric interpretations, number lines, or draw mental diagrams. "
            "Avoid abstract algebra if possible and rely on spatial reasoning. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Programmatic": {
        "name": "Programmatic",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Think like a programmer. Break the problem down into an algorithm or pseudocode. "
            "Use logical steps like 'Initialize', 'Loop', 'Condition' to explain the solution process. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Reverse": {
        "name": "Reverse",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Use 'Working Backward' strategy. Start from what you want to find or assume the answer, "
            "and work your way back to the given conditions to verify logically. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Analogical": {
        "name": "Analogical",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Use analogies or real-world metaphors. Translate the abstract numbers into concrete objects "
            "(e.g., apples, money, distance) to make the logic intuitive. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "First_Principles": {
        "name": "First_Principles",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Reason from first principles. Do not use any memorized formulas. "
            "Derive every step logically from the basic definitions of the numbers and operations involved. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Pattern": {
        "name": "Pattern",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Use inductive reasoning. Test the logic with smaller, simpler numbers to find a pattern, "
            "then generalize that pattern to solve the specific problem. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Didactic": {
        "name": "Didactic",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Explain it as if you are a kind tutor teaching a confused student. "
            "Use conversational language, ask rhetorical questions, and guide the thought process gently. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Intuitive": {
        "name": "Intuitive",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Start with a strong intuitive guess or estimation (Fermi method) to bound the answer, "
            "then refine it with precise calculation. Focus on the 'number sense'. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    "Decomposition": {
        "name": "Decomposition",
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Break the problem into a numbered list of independent sub-problems. "
            "Solve each sub-problem explicitly and combine them for the final result. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    }
}

class DatasetLoader:
    @staticmethod
    def load_gsm8k(split: str = "test", sample_size: int = None) -> List[Dict]:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        problems = []
        for item in dataset:
            answer_match = re.search(r'####\s*(.+)', item['answer'])
            answer = answer_match.group(1).strip() if answer_match else item['answer']
            problems.append({
                'question': item['question'],
                'answer': answer,
                'dataset': 'GSM8K',
                'difficulty': 'Elementary'
            })
        
        if sample_size:
            import random
            random.seed(42)
            problems = random.sample(problems, min(sample_size, len(problems)))
        return problems
    
    @staticmethod
    def load_math_dataset(levels: List[int] = [4, 5], split: str = "test", sample_size: int = None) -> List[Dict]:
        try:
            configs = get_dataset_config_names("EleutherAI/hendrycks_math")
        except Exception:
            configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
            
        problems = []
        print(f"Loading MATH dataset subsets: {configs}...")
        
        for config_name in configs:
            try:
                subset = load_dataset("EleutherAI/hendrycks_math", config_name, split=split)
                for item in subset:
                    if item['level'] in [f"Level {l}" for l in levels]:
                        answer = RobustAnswerEvaluator.extract_boxed_content(item['solution'])
                        if not answer:
                            answer = item['answer']
                        
                        problems.append({
                            'question': item['problem'],
                            'answer': answer,
                            'dataset': 'MATH',
                            'difficulty': item['level'],
                            'subject': item['type']
                        })
            except Exception as e:
                print(f"Warning: Failed to load MATH subset '{config_name}': {e}")
                continue

        if sample_size:
            import random
            random.seed(42)
            problems = random.sample(problems, min(sample_size, len(problems)))
        return problems
    
    @staticmethod
    def load_aime_dataset(sample_size: int = None) -> List[Dict]:
        try:
            dataset = load_dataset("gneubig/aime-1983-2024", split="train")
        except Exception as e:
            print(f"Error loading AIME dataset: {e}")
            return []

        problems = []
        for item in dataset:
            problems.append({
                'question': item['Question'],
                'answer': str(item['Answer']),
                'dataset': 'AIME',
                'difficulty': 'Competition',
                'year': item.get('Year', 'Unknown')
            })
        
        if sample_size:
            import random
            random.seed(42)
            problems = random.sample(problems, min(sample_size, len(problems)))
        return problems

class RobustAnswerEvaluator:
    @staticmethod
    def extract_boxed_content(text: str) -> str:
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def extract_answer_from_response(response: str) -> str:
        if '####' in response:
            return response.split('####')[1].strip()
        
        boxed = RobustAnswerEvaluator.extract_boxed_content(response)
        if boxed:
            return boxed
            
        patterns = [
            r'(?:The|Final)\s+answer\s+is\s+([^\.]+)',
            r'=\s+([^\.]+?)$'
        ]
        for p in patterns:
            match = re.search(p, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        last_number_match = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', response)
        if last_number_match:
            return last_number_match[-1]
            
        return ""

    @staticmethod
    def parse_number(text: str) -> Optional[float]:
        text = str(text).strip().lower()
        text = re.sub(r'[^\d\.\-/]', '', text)
        
        try:
            if '/' in text:
                num, den = text.split('/')
                return float(num) / float(den)
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def normalize_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[\{\}\$\s,]', '', text)
        return text

    @staticmethod
    def is_correct(predicted: str, ground_truth: str) -> bool:
        if not predicted:
            return False

        pred_val = RobustAnswerEvaluator.parse_number(predicted)
        gt_val = RobustAnswerEvaluator.parse_number(ground_truth)

        if pred_val is not None and gt_val is not None:
            return math.isclose(pred_val, gt_val, rel_tol=1e-5)

        pred_norm = RobustAnswerEvaluator.normalize_text(predicted)
        gt_norm = RobustAnswerEvaluator.normalize_text(ground_truth)
        
        return pred_norm == gt_norm

def construct_prompt(question: str, style_config: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": style_config["system_message"]},
        {"role": "user", "content": f"Problem:\n{question}"}
    ]

class MathReasoningExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluator = RobustAnswerEvaluator()
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        print(f"Loading model {config.model_name}...")
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
            seed=config.seed
        )
    
    def run_experiment(self, problems: List[Dict], experiment_name: str):
        print(f"\nStarting Experiment (v3 - Robust Regex): {experiment_name}")
        
        results = []
        all_prompts = []
        metadata = []
        
        for prob_idx, problem in enumerate(problems):
            for style_name, style_config in REASONING_STYLES.items():
                prompt = construct_prompt(problem['question'], style_config)
                all_prompts.append(prompt)
                metadata.append({
                    'problem_idx': prob_idx,
                    'problem': problem,
                    'style_name': style_name
                })
        
        print(f"Running inference on {len(all_prompts)} prompts...")
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        outputs = self.llm.chat(messages=all_prompts, sampling_params=sampling_params, use_tqdm=True)
        
        print("Evaluating results with Robust Regex Parsing...")
        for output, meta in tqdm(zip(outputs, metadata), total=len(outputs)):
            response_text = output.outputs[0].text if output.outputs else ""
            
            predicted_answer = self.evaluator.extract_answer_from_response(response_text)
            is_correct = self.evaluator.is_correct(predicted_answer, meta['problem']['answer'])
            
            results.append({
                'problem_idx': meta['problem_idx'],
                'style': meta['style_name'],
                'dataset': meta['problem']['dataset'],
                'is_correct': is_correct,
                'predicted_answer': predicted_answer,
                'ground_truth': meta['problem']['answer'],
                'full_response': response_text
            })
        
        self.save_and_report(results, experiment_name)

    def save_and_report(self, results: List[Dict], experiment_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = os.path.join(self.config.output_dir, f"{experiment_name}_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        df = pd.DataFrame(results)
        print("\n### [v3 Robust] Final Accuracy Report")
        print(f"Overall Accuracy: {df['is_correct'].mean():.2%}")
        
        pivot = df.pivot_table(index='style', columns='dataset', values='is_correct', aggfunc='mean')
        print("\n### Breakdown by Style & Dataset")
        print(pivot.applymap(lambda x: f"{x:.2%}"))
        
        csv_path = os.path.join(self.config.output_dir, f"{experiment_name}_summary_{timestamp}.csv")
        pivot.to_csv(csv_path)
        print(f"\nSaved detailed results to {json_path}")

def main():
    config = ExperimentConfig()
    experiment = MathReasoningExperiment(config)
    
    print("Loading data...")
    problems = (
        DatasetLoader.load_gsm8k(sample_size=1000) + 
        DatasetLoader.load_math_dataset(sample_size=1000) +
        DatasetLoader.load_aime_dataset(sample_size=50)
    )
    
    experiment.run_experiment(problems, "robust_regex_evaluation")

if __name__ == "__main__":
    main()