import os
import torch
import gguf

# Paths
INPUT_PATH = "./steering_vector/error_vectors_norm.pt"
OUTPUT_DIR = "./steering_vector/gguf_versions"

# ControlVector metadata
MODEL_HINT = "qwen2.5"
NUM_LAYERS = 28  # Qwen2.5-7B has 28 transformer layers

# Export targets
TARGET_CONFIGS = [
    {"vec_idx": 3, "alpha": 5, "filename": "error_3_alpha_5"},
    {"vec_idx": 2, "alpha": -10, "filename": "error_2_alpha_-10"},
]


def save_scaled_vectors_to_gguf():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    data = torch.load(INPUT_PATH, map_location="cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cfg in TARGET_CONFIGS:
        vec_idx, alpha, fname = cfg["vec_idx"], cfg["alpha"], cfg["filename"]
        output_path = os.path.join(OUTPUT_DIR, f"{fname}.gguf")
        print(f"Writing: {output_path} (vec_idx={vec_idx}, alpha={alpha})")

        writer = gguf.GGUFWriter(output_path, "controlvector")

        # Minimal metadata required by common controlvector loaders
        writer.add_string("controlvector.model_hint", MODEL_HINT)
        writer.add_uint32("controlvector.layer_count", NUM_LAYERS)

        # We apply alpha here; do not let the loader renormalize the vectors.
        writer.add_bool("controlvector.normalize", False)

        saved = 0
        for layer_idx in range(NUM_LAYERS):
            layer_key = str(layer_idx)
            if layer_key not in data:
                continue

            vec = data[layer_key][vec_idx].detach().float() * alpha
            writer.add_tensor(f"direction.{layer_idx}", vec.numpy().reshape(-1))
            saved += 1

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print(f"Done: {fname}.gguf (layers={saved}/{NUM_LAYERS})")


if __name__ == "__main__":
    save_scaled_vectors_to_gguf()
