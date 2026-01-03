import os
import glob
import torch
import numpy as np
from tqdm import tqdm

TARGET_DIRS = [
    "./steering_vector/Analogical_vectors",
    "./steering_vector/Decomposition_vectors",
    "./steering_vector/Didactic_vectors",
    "./steering_vector/First_Principles_vectors",
    "./steering_vector/Intuitive_vectors",
    "./steering_vector/Pattern_vectors",
    "./steering_vector/Programmatic_vectors",
    "./steering_vector/Reverse_vectors",
    "./steering_vector/Standard_vectors",
    "./steering_vector/Visual_vectors",
]
SAVE_DIR = "./steering_vector/styles/processed_vectors"
os.makedirs(SAVE_DIR, exist_ok=True)


def preprocess_and_save():
    print("[*] Converting chunk .pt files into .npy arrays...")
    style_metadata = []

    for dir_path in TARGET_DIRS:
        style_name = os.path.basename(dir_path).replace("_vectors", "")
        chunk_files = sorted(glob.glob(os.path.join(dir_path, "chunk_*.pt")))
        if not chunk_files:
            continue

        print(f" -> Processing {style_name} ({len(chunk_files)} chunks)...")

        buffer = []
        for fpath in tqdm(chunk_files, leave=False):
            try:
                data = torch.load(fpath, map_location="cpu")  # load on CPU to save GPU memory
                for item in data:
                    buffer.append(item["vector"].float().numpy())
            except Exception as e:
                print(f"    [Error] {fpath}: {e}")

        if not buffer:
            continue

        full_arr = np.stack(buffer)  # [N, Layers, Hidden]
        save_path = os.path.join(SAVE_DIR, f"{style_name}.npy")
        np.save(save_path, full_arr)

        style_metadata.append(
            {"name": style_name, "path": save_path, "shape": full_arr.shape, "count": full_arr.shape[0]}
        )
        print(f"    Saved to {save_path} | Shape: {full_arr.shape}")

        del buffer, full_arr

    np.save(os.path.join(SAVE_DIR, "metadata.npy"), style_metadata)  # simple index for later loading
    print("[*] Done.")


if __name__ == "__main__":
    preprocess_and_save()
