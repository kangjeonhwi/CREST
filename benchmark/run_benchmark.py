import os
import json
import argparse
import traceback
import numpy as np
from datetime import datetime
from collections import defaultdict

from configs import MODEL_REGISTRY, RANDOM_SEED, OUTPUT_DIR
from src.data_loader import DataLoader
from src.model_engine import ModelEngine
from src.metrics import Evaluator
from src.utils import MathVerifier


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_output_dir():
    """Create output directory if missing."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[Setup] Created output directory: {OUTPUT_DIR}")


def aggregate_results(per_problem_results):
    """Aggregate metrics by dataset source."""
    stats = defaultdict(lambda: defaultdict(list))

    for res in per_problem_results:
        cat = res.get("source", "unknown")

        stats[cat]["pass@1"].append(res["pass@1"])
        stats[cat]["pass@100"].append(res["pass@100"])
        stats[cat]["major@100"].append(res.get("major@100", 0.0))

        stats[cat]["diversity_all"].append(res["diversity_all"])
        stats[cat]["diversity_correct"].append(res["diversity_correct"])
        stats[cat]["unique_path"].append(res["unique_path_ratio"])

    final_agg = {}
    for cat, metrics in stats.items():
        final_agg[cat] = {k: float(np.mean(v)) for k, v in metrics.items()}
        final_agg[cat]["count"] = len(metrics["pass@1"])

    return final_agg


def print_summary_table(model_name, agg_stats):
    """Print a compact summary table."""
    print(f"\n[Summary] Model: {model_name}")
    print("=" * 100)
    print(
        f"{'Source':<15} | {'N':<5} | {'P@1':<8} | {'P@100':<8} | {'Maj@100':<8} | {'Div(All)':<8} | {'UniqPath':<8}"
    )
    print("-" * 100)

    for cat, metrics in sorted(agg_stats.items()):
        print(
            f"{cat:<15} | {metrics['count']:<5} | "
            f"{metrics['pass@1']:.4f}   | {metrics['pass@100']:.4f}   | {metrics['major@100']:.4f}   | "
            f"{metrics['diversity_all']:.4f}   | {metrics['unique_path']:.4f}"
        )
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to run (defaults to all in config)",
    )
    args = parser.parse_args()

    setup_output_dir()

    # Load once and reuse across models for fair comparison.
    data_loader = DataLoader(seed=RANDOM_SEED)
    raw_data = data_loader.load_all_datasets()

    if not raw_data:
        print("[Error] No data loaded. Check data_loader.py or configs.")
        return

    target_models = args.models if args.models else MODEL_REGISTRY.keys()

    print(f"[Run] Starting benchmark on {len(target_models)} models")
    print(f"[Run] Total problems: {len(raw_data)}")

    for model_name in target_models:
        if model_name not in MODEL_REGISTRY:
            print(f"[Warn] {model_name} not in registry. Skipping.")
            continue

        prompt_key = MODEL_REGISTRY[model_name]
        safe_name = model_name.split("/")[-1]
        result_file = os.path.join(OUTPUT_DIR, f"results_{safe_name}.json")

        # Resume: skip if results exist.
        if os.path.exists(result_file):
            print(f"[Run] Results exist for {safe_name}. Skipping.")
            continue

        print("\n" + "#" * 80)
        print(f"[Run] Model: {model_name} | Prompt: {prompt_key}")
        print("#" * 80)

        try:
            # A) Inference (GPU-heavy)
            engine = ModelEngine(model_name, prompt_key)
            questions = [d["question"] for d in raw_data]
            outputs_list = engine.generate(questions)

            # Release GPU memory before evaluation (embedding model may use GPU).
            engine.unload()
            del engine

            # B) Evaluation
            print("[Eval] Evaluating responses...")
            evaluator = Evaluator()

            per_problem_results = []
            jsonl_file = result_file.replace(".json", "_generations.jsonl")

            with open(jsonl_file, "w", encoding="utf-8") as f_jsonl:
                for i, (data_item, outputs) in enumerate(zip(raw_data, outputs_list)):
                    metrics = evaluator.evaluate_batch(
                        question=data_item["question"],
                        outputs=outputs,
                        gold_answer=data_item["gold"],
                    )

                    metrics["source"] = data_item["source"]
                    metrics["model"] = model_name

                    per_problem_results.append(metrics)

                    log_entry = {
                        "id": i,
                        "source": data_item["source"],
                        "question": data_item["question"],
                        "gold": data_item["gold"],
                        "outputs": outputs,
                        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, list)},
                    }
                    f_jsonl.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                    if (i + 1) % 50 == 0:
                        print(f"[Eval] {i + 1}/{len(raw_data)} done")

            # C) Aggregate + save
            agg_stats = aggregate_results(per_problem_results)

            final_output = {
                "config": {
                    "model": model_name,
                    "prompt_key": prompt_key,
                    "seed": RANDOM_SEED,
                    "timestamp": datetime.now().isoformat(),
                },
                "overall_stats": agg_stats,
                "per_problem_details": per_problem_results,
            }

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

            print(f"[Save] Wrote results: {result_file}")
            print_summary_table(safe_name, agg_stats)

        except Exception as e:
            print(f"[Error] Failed on {model_name}: {e}")
            traceback.print_exc()
            try:
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
            continue

    print("\n[Done] All benchmarks completed.")

if __name__ == "__main__":
    main()
