import torch

RANDOM_SEED = 42
OUTPUT_DIR = "./benchmark/results"

GENERATION_PARAMS = {
    "n": 100,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 2048,
    "stop": ["<|im_end|>", "<|endoftext|>", "User:", "Observation:"]
}

GREEDY_PARAMS = {
    "n": 1,
    "temperature": 0.0,  # Greedy
    "top_p": 1.0,
    "max_tokens": 2048,
    "stop": ["<|im_end|>", "<|endoftext|>", "User:", "Observation:"]
}

PROMPT_TEMPLATES = {
    "qwen_instruct": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        ),
        "use_chat_template": True
    },
    
    "qwen_math": {
        "system_message": (
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
        "use_chat_template": True
    },
    
    "deepseek_math": {
        "system_message": (
            "The user will ask a math problem. You are a math expert. "
            "Solve the problem using a chain of thought process. "
            "Finally, present the answer in LaTeX format enclosed in \\boxed{}."
        ),
        "use_chat_template": True
    },
}

MODEL_REGISTRY = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen_instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct": "qwen_math",
    "Qwen/Qwen2.5-14B-Instruct": "qwen_instruct",
    "Qwen/Qwen2.5-32B-Instruct": "qwen_instruct",  # Large Scale Check
    "deepseek-ai/deepseek-math-7b-rl": "deepseek_math", # Reference
}

DATA_CONFIG = {
    "gsm8k": 500,   
    "math_level4": 500,
    "math_level5": 500,
    "aime": 50, 
    "olympiad": 50
}