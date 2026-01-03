# configs.py
import os

CUDA_VISIBLE_DEVICES = "5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DATASET_PATH = "./datasets/pos_neg_dataset/prm800k_self_correction.json"
OUTPUT_DIR = "./steering_vector"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MAX_MODEL_LEN = 4096
SEED = 42

SAMPLE_SIZE_GSM8K = 1000
SAMPLE_SIZE_MATH = 1000
SAMPLE_SIZE_AIME = 50

SAVE_INTERVAL = 200
DEBUG_SAMPLES = None

SYSTEM_MESSAGE = (
    "You are a helpful and logical AI assistant. "
    "Solve the mathematical problem step by step to ensure accuracy. "
    "You must enclose your final answer in \\boxed{}."
)

REASONING_STYLES = {
    "Naive": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Think step by step and solve this problem carefully. Show all your work. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Standard": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Solve this problem using the most standard, textbook academic method. "
            "Be formal and concise. Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Visual": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Solve this problem by visualizing it. Use geometric interpretations, number lines, or draw mental diagrams. "
            "Avoid abstract algebra if possible and rely on spatial reasoning. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Programmatic": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Think like a programmer. Break the problem down into an algorithm or pseudocode. "
            "Use logical steps like 'Initialize', 'Loop', 'Condition' to explain the solution process. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Reverse": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Use 'Working Backward' strategy. Start from what you want to find or assume the answer, "
            "and work your way back to the given conditions to verify logically. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Analogical": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Use analogies or real-world metaphors. Translate the abstract numbers into concrete objects "
            "(e.g., apples, money, distance) to make the logic intuitive. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "First_Principles": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Reason from first principles. Do not use any memorized formulas. "
            "Derive every step logically from the basic definitions of the numbers and operations involved. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Pattern": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Use inductive reasoning. Test the logic with smaller, simpler numbers to find a pattern, "
            "then generalize that pattern to solve the specific problem. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Didactic": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Explain it as if you are a kind tutor teaching a confused student. "
            "Use conversational language, ask rhetorical questions, and guide the thought process gently. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Intuitive": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Start with a strong intuitive guess or estimation (Fermi method) to bound the answer, "
            "then refine it with precise calculation. Focus on the 'number sense'. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
    "Decomposition": {
        "system_message": (
            "You are a helpful and logical AI assistant. "
            "Break the problem into a numbered list of independent sub-problems. "
            "Solve each sub-problem explicitly and combine them for the final result. "
            "Solve the mathematical problem step by step to ensure accuracy. "
            "You must enclose your final answer in \\boxed{}."
        )
    },
}
