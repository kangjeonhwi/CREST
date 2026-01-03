import numpy as np
from datasets import load_dataset
from configs import DATA_CONFIG, RANDOM_SEED

class DataLoader:
    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed

    def load_gsm8k(self, limit=None):
        """Load GSM8K test split."""
        print(f"📦 Loading GSM8K (Limit: {limit})...")
        try:
            ds = load_dataset("gsm8k", "main", split="test")
            ds = ds.shuffle(seed=self.seed)
            if limit:
                ds = ds.select(range(limit))

            return [
                {
                    "source": "gsm8k",
                    "question": item["question"],
                    "gold": item["answer"],
                }
                for item in ds
            ]
        except Exception as e:
            print(f"⚠️ Failed to load GSM8K: {e}")
            return []

    def load_math(self, level_filter=None, limit=None):
        """Load Hendrycks MATH test split (all subjects), optionally filter by level."""
        print(f"📦 Loading MATH ({level_filter}, Limit: {limit})...")
        try:
            math_subjects = [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ]

            ds_list = []
            for subject in math_subjects:
                d = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
                ds_list.append(d)

            from datasets import concatenate_datasets

            ds = concatenate_datasets(ds_list)

            if level_filter:
                ds = ds.filter(lambda x: level_filter in str(x["level"]))

            ds = ds.shuffle(seed=self.seed)
            if limit:
                ds = ds.select(range(limit))

            return [
                {
                    "source": f"math_{level_filter.replace(' ', '').lower() if level_filter else 'all'}",
                    "question": item["problem"],
                    "gold": item["solution"],
                }
                for item in ds
            ]
        except Exception as e:
            print(f"⚠️ Failed to load MATH: {e}")
            return []

    def load_aime(self, limit=None):
        """Load AIME split."""
        print(f"📦 Loading AIME (Limit: {limit})...")
        try:
            ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
            ds = ds.shuffle(seed=self.seed)
            if limit:
                ds = ds.select(range(limit))

            return [
                {
                    "source": "aime",
                    "question": item["problem"],
                    "gold": str(item["answer"]),
                }
                for item in ds
            ]
        except Exception as e:
            print(f"⚠️ Failed to load AIME: {e}")
            return []

    def load_olympiad_bench(self, limit=None):
        """Load OlympiadBench split; prefer final_answer, fallback to solution."""
        print(f"📦 Loading OlympiadBench (Limit: {limit})...")
        try:
            ds = load_dataset(
                "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train"
            )

            ds = ds.shuffle(seed=self.seed)
            if limit:
                ds = ds.select(range(limit))

            data_list = []
            for item in ds:
                gold = item.get("final_answer", "")
                if not gold or str(gold).strip() == "":
                    gold = item.get("solution", "")

                data_list.append(
                    {
                        "source": "olympiad_bench",
                        "question": item["question"],
                        "gold": str(gold),
                    }
                )
            return data_list

        except Exception as e:
            print(f"⚠️ Failed to load OlympiadBench: {e}")
            return []

    def load_all_datasets(self):
        """Load and merge all datasets defined in DATA_CONFIG."""
        all_data = []

        if DATA_CONFIG.get("gsm8k", 0) > 0:
            all_data.extend(self.load_gsm8k(DATA_CONFIG["gsm8k"]))

        if DATA_CONFIG.get("math_level4", 0) > 0:
            all_data.extend(self.load_math("Level 4", DATA_CONFIG["math_level4"]))

        if DATA_CONFIG.get("math_level5", 0) > 0:
            all_data.extend(self.load_math("Level 5", DATA_CONFIG["math_level5"]))

        if DATA_CONFIG.get("aime", 0) > 0:
            all_data.extend(self.load_aime(DATA_CONFIG["aime"]))

        if DATA_CONFIG.get("olympiad", 0) > 0:
            all_data.extend(self.load_olympiad_bench(DATA_CONFIG["olympiad"]))

        print(f"✅ Total Data Loaded: {len(all_data)} samples")
        return all_data
