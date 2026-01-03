# configs.py
import os

CUDA_VISIBLE_DEVICES = "6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DATASET_PATH = "./datasets/pos_neg_dataset/prm800k_pairwise_correction.json"
OUTPUT_DIR = "./steering_vector"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_MESSAGE = (
    "You are a helpful and logical AI assistant. "
    "Solve the mathematical problem step by step to ensure accuracy. "
    "You must enclose your final answer in \\boxed{}."
)

SAVE_INTERVAL = 200  # Periodically save chunks to avoid RAM/GPU memory blow-up
DEBUG_SAMPLES = None  # e.g., 2 for quick debug, or None to run all