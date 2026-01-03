import os
import json
import argparse
import traceback
import numpy as np
from datetime import datetime
from collections import defaultdict

# Project modules
from configs import MODEL_REGISTRY, RANDOM_SEED, OUTPUT_DIR, GREEDY_PARAMS
from src.data_loader import DataLoader
from src.model_engine import ModelEngine
from src.metrics import Evaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def aggregate_greedy_results(per_problem_results):
    """Aggregate greedy-mode results (Pass@1 == Accuracy)."""
    stats = defaultdict(list)

    for res in per_problem_results:
        cat = res.get("source", "unknown")
        stats[cat].append(res["pass@1"])

    final_agg = {}
    for cat, values in stats.items():
        final_agg[cat] = {
            "accuracy": float(np.mean(values)),
            "count": len(values)
        }

    return final_agg


def print_greedy_summary(model_name, agg_stats):
    """Print a concise summary for greedy evaluation."""
    print(f"\nGreedy evaluation summary: {model_name}")
    print("=" * 60)
    print(f"{'Source':<20} | {'N':<6} | {'Accuracy':<10}")
    print("-" * 60)

    total_correct = 0
    total_count = 0

    for cat, metrics in sorted(agg_stats.items()):
        acc = metrics["accuracy"]
        count = metrics["count"]
        print(f"{cat:<20} | {count:<6} | {acc:.2%}")

        total_correct += acc * count
        total_count += count

    print("-" * 60)
    if total_count > 0:
        avg_acc = total_correct / total_count
        print(f"{'AVERAGE':<20} | {total_count:<6} | {avg_acc:.2%}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to run")
    args = parser.parse_args()

    setup_output_dir()

    # Load datasets with a fixed seed for reproducibility.
    data_loader = DataLoader(seed=RANDOM_SEED)
    raw_data = data_loader.load_all_datasets()

    if not raw_data:
        print("No data loaded.")
        return

    target_models = args.models if args.models else MODEL_REGISTRY.keys()
    print(f"Starting greedy benchmark: {len(list(target_models))} models")

    for model_name in target_models:
        if model_name not in MODEL_REGISTRY:
            continue

        prompt_key = MODEL_REGISTRY[model_name]
        safe_name = model_name.split("/")[-1]

        result_file = os.path.join(OUTPUT_DIR, f"results_greedy_{safe_name}.json")

        if os.path.exists(result_file):
            print(f"Skipping (already exists): {safe_name}")
            continue

        print("\n" + "#" * 80)
        print(f"Model: {model_name} (greedy)")
        print("#" * 80)

        try:
            # Load model and generate with greedy parameters (N=1, temp=0).
            engine = ModelEngine(model_name, prompt_key)
            questions = [d["question"] for d in raw_data]
            outputs_list = engine.generate(questions, sampling_config=GREEDY_PARAMS)

            engine.unload()
            del engine

            print("Evaluating outputs...")
            evaluator = Evaluator()

            per_problem_results = []
            jsonl_file = result_file.replace(".json", "_generations.jsonl")

            with open(jsonl_file, "w", encoding="utf-8") as f_jsonl:
                for i, (data_item, outputs) in enumerate(zip(raw_data, outputs_list)):
                    metrics = evaluator.evaluate_batch(
                        question=data_item["question"],
                        outputs=outputs,
                        gold_answer=data_item["gold"]
                    )

                    metrics["source"] = data_item["source"]
                    metrics["model"] = model_name

                    per_problem_results.append(metrics)

                    log_entry = {
                        "id": i,
                        "source": data_item["source"],
                        "question": data_item["question"],
                        "gold": data_item["gold"],
                        "prediction": outputs[0],
                        "is_correct": metrics["pass@1"] > 0.0
                    }
                    f_jsonl.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            agg_stats = aggregate_greedy_results(per_problem_results)

            final_output = {
                "config": {
                    "model": model_name,
                    "mode": "greedy",
                    "params": GREEDY_PARAMS,
                    "seed": RANDOM_SEED,
                    "timestamp": datetime.now().isoformat()
                },
                "overall_stats": agg_stats,
            }

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

            print(f"Saved results: {result_file}")
            print_greedy_summary(safe_name, agg_stats)

        except Exception as e:
            print(f"Error processing model: {model_name}: {e}")
            traceback.print_exc()
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
            continue

    print("All greedy benchmarks completed.")


if __name__ == "__main__":
    main()