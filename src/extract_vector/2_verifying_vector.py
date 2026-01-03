import os
import json
import gc
import shutil
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import configs as cfg

class SelfCorrectionSteeringExtractor:
    def __init__(self, model_name, system_prompt, device="cuda", debug_mode=True):
        print(f">>> Loading Model: {model_name}...")
        self.device = device
        self.system_prompt = system_prompt
        self.debug_mode = debug_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self.activations[name] = hidden.detach()
            return hook

        for name, layer in self.model.model.layers.named_children():
            self.hooks.append(layer.register_forward_hook(get_activation(name)))

        print(f">>> Registered hooks for {len(self.hooks)} layers.")

    def clear_activations(self):
        self.activations = {}

    def get_targeted_activations(
        self,
        prompt,
        context_steps,
        target_step,
        intermediate_steps=None,
        debug_label=""
    ):
        self.clear_activations()

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": prompt})

        assistant_content_parts = context_steps[:]
        if intermediate_steps:
            if isinstance(intermediate_steps, list):
                assistant_content_parts.extend(intermediate_steps)
            else:
                assistant_content_parts.append(intermediate_steps)

        assistant_content_parts.append(target_step)
        assistant_full_content = " ".join(assistant_content_parts)
        full_messages = messages + [{"role": "assistant", "content": assistant_full_content}]

        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        full_encoding = self.tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
        full_tokens = full_encoding["input_ids"]
        offset_mapping = full_encoding["offset_mapping"][0]

        search_target = target_step.strip()
        target_start_char = full_text.rfind(search_target)
        if target_start_char == -1:
            search_target = target_step
            target_start_char = full_text.rfind(search_target)
        target_end_char = target_start_char + len(search_target)

        start_idx = 0
        end_idx = len(offset_mapping)
        found_start = False

        for i, (start, end) in enumerate(offset_mapping):
            if not found_start and (start <= target_start_char < end):
                start_idx = i
                found_start = True
            if end >= target_end_char:
                end_idx = i + 1
                break

        if self.debug_mode:
            print(f"\n{'-'*40}")
            print(f"[{debug_label}]")
            extracted_tokens = full_tokens[0, start_idx:end_idx]
            decoded_extracted = self.tokenizer.decode(extracted_tokens)
            is_match = search_target.startswith(decoded_extracted.strip()[:5])
            status = "MATCH" if is_match else "MISMATCH (Possible Cut-off)"
            print(f"Mapped Tokens: {start_idx} ~ {end_idx}")
            print(f"Decoded Snippet: '{decoded_extracted}'")
            print(f"Status: {status}")
            print(f"{'-'*40}")

        inputs = {"input_ids": full_tokens.to(self.device)}
        if "attention_mask" in full_encoding:
            inputs["attention_mask"] = full_encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        layer_means = {}
        for layer_name, hidden_states in self.activations.items():
            if hidden_states.dim() == 3:
                target_states = hidden_states[0, start_idx:end_idx, :]
            else:
                target_states = hidden_states[start_idx:end_idx, :]

            if target_states.shape[0] == 0:
                if self.debug_mode:
                    print(f"WARNING: Empty state slice for {debug_label}. Using last token.")
                target_states = hidden_states[0, -1:, :] if hidden_states.dim() == 3 else hidden_states[-1:, :]

            mean_state = torch.mean(target_states, dim=0)
            layer_means[layer_name] = mean_state.cpu()

        return layer_means


def main():
    model_id = cfg.MODEL_NAME
    system_prompt = cfg.SYSTEM_MESSAGE
    output_filename = "self_correction_vectors_qwen.pt"
    full_output_path = os.path.join(cfg.OUTPUT_DIR, output_filename)

    temp_dir = os.path.join(cfg.OUTPUT_DIR, "temp_chunks")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print(f">>> Loading dataset from {cfg.DATASET_PATH}...")
    with open(cfg.DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_data = data[:cfg.DEBUG_SAMPLES] if cfg.DEBUG_SAMPLES else data
    extractor = SelfCorrectionSteeringExtractor(model_id, system_prompt, debug_mode=False)

    print(">>> Starting 3-Way Extraction Loop...")

    buffer_data = []
    chunk_paths = []

    for idx, sample in tqdm(enumerate(target_data), total=len(target_data)):
        try:
            prompt = sample["prompt"]
            context_steps = sample["context_steps"]
            meta = sample["meta"]

            bad_step = meta["bad_step"]
            trigger = meta["trigger"]
            correction_step = meta["correction_step"]

            bad_acts = extractor.get_targeted_activations(
                prompt,
                context_steps,
                target_step=bad_step,
                debug_label=f"SAMPLE {idx} - BASELINE (BAD)"
            )

            trigger_acts = extractor.get_targeted_activations(
                prompt,
                context_steps,
                target_step=trigger,
                debug_lab
