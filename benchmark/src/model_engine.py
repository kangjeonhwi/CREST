import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from configs import PROMPT_TEMPLATES, GENERATION_PARAMS, RANDOM_SEED


class ModelEngine:
    def __init__(self, model_path, prompt_config_key):
        self.model_path = model_path
        self.prompt_key = prompt_config_key
        self.llm = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Initialize tokenizer + vLLM engine."""
        print(f"[ModelEngine] Initializing model: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            seed=RANDOM_SEED,
        )

    def format_prompts(self, questions):
        """Apply model-specific system prompt and tokenizer chat template."""
        config = PROMPT_TEMPLATES.get(self.prompt_key, PROMPT_TEMPLATES["qwen_instruct"])
        system_msg = config["system_message"]

        formatted_prompts = []
        for q in questions:
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})

            messages.append({"role": "user", "content": q})

            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(full_prompt)

        return formatted_prompts

    def generate(self, questions, sampling_config=None):
        """
        Generate with optional sampling_config override.
        If None, uses configs.GENERATION_PARAMS.
        """
        config = sampling_config if sampling_config else GENERATION_PARAMS

        prompts = self.format_prompts(questions)

        sampling_params = SamplingParams(
            n=config["n"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            stop=config["stop"],
            seed=RANDOM_SEED,
        )

        print(
            f"[ModelEngine] Generating {len(questions)} items × {config['n']} paths "
            f"(temp={config['temperature']})"
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)

        final_results = []
        for output in outputs:
            generated_texts = [o.text for o in output.outputs]
            final_results.append(generated_texts)

        return final_results

    def unload(self):
        """Release vLLM engine resources to avoid OOM on subsequent loads."""
        print(f"[ModelEngine] Unloading model: {self.model_path}")

        if self.llm:
            from vllm.distributed.parallel_state import destroy_model_parallel

            try:
                destroy_model_parallel()
            except:
                pass

            del self.llm

        if self.tokenizer:
            del self.tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[ModelEngine] GPU memory released.")