import numpy as np
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
from .utils import MathVerifier


def compute_pass_at_k(n_correct, n_total, k):
    """Unbiased pass@k estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n_total < k:
        return float(n_correct > 0)
    if n_correct == 0:
        return 0.0
    if n_correct == n_total:
        return 1.0

    from scipy.special import comb

    val = 1.0 - (comb(n_total - n_correct, k) / comb(n_total, k))
    return float(val)


class DiversityCalculator:
    _instance = None
    _model = None

    def __new__(cls, model_name="all-MiniLM-L6-v2", device=None):
        if cls._instance is None:
            cls._instance = super(DiversityCalculator, cls).__new__(cls)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔄 Loading Embedding Model ({model_name}) on {device}...")
            cls._model = SentenceTransformer(model_name, device=device)
        return cls._instance

    def compute_metrics(
        self, samples: List[str], correct_indices: List[int]
    ) -> Tuple[float, float, float]:
        """Return (diversity_all, diversity_correct, unique_cluster_ratio)."""
        if len(samples) < 2:
            return 0.0, 0.0, 0.0

        embeddings = self._model.encode(
            samples, convert_to_tensor=True, show_progress_bar=False
        )

        # Mean cosine distance over all non-diagonal pairs.
        cos_sim_all = util.cos_sim(embeddings, embeddings)
        mask = ~torch.eye(len(samples), dtype=bool, device=cos_sim_all.device)
        diversity_all = 1.0 - cos_sim_all[mask].mean().item()

        diversity_correct = 0.0
        unique_ratio = 0.0

        if len(correct_indices) >= 2:
            correct_embs = embeddings[correct_indices]
            cos_sim_corr = util.cos_sim(correct_embs, correct_embs)
            mask_corr = ~torch.eye(
                len(correct_indices), dtype=bool, device=cos_sim_corr.device
            )
            diversity_correct = 1.0 - cos_sim_corr[mask_corr].mean().item()

            # Greedy clustering with cosine similarity threshold.
            clusters = []
            threshold = 0.90
            for i in range(len(correct_indices)):
                is_new = True
                for rep_idx in clusters:
                    if cos_sim_corr[i][rep_idx] > threshold:
                        is_new = False
                        break
                if is_new:
                    clusters.append(i)

            unique_ratio = len(clusters) / len(correct_indices)

        elif len(correct_indices) == 1:
            unique_ratio = 1.0

        return diversity_all, diversity_correct, unique_ratio


class Evaluator:
    def __init__(self):
        self.verifier = MathVerifier()
        self.div_calc = DiversityCalculator()

    def evaluate_batch(self, question: str, outputs: List[str], gold_answer: str) -> Dict:
        """Evaluate N outputs for a single question."""
        extracted_gold = self.verifier.extract_answer(gold_answer)
        extracted_preds = [self.verifier.extract_answer(o) for o in outputs]

        correct_indices = []
        is_correct_list = []
        for i, pred in enumerate(extracted_preds):
            is_corr = self.verifier.is_equivalent(pred, extracted_gold)
            is_correct_list.append(is_corr)
            if is_corr:
                correct_indices.append(i)

        n_correct = len(correct_indices)
        n_total = len(outputs)

        pass_metrics = {}
        for k in [1, 10, 50, 100]:
            pass_metrics[f"pass@{k}"] = compute_pass_at_k(n_correct, n_total, k)

        # Majority vote over normalized extracted predictions.
        valid_preds = [p for p in extracted_preds if p]
        if valid_preds:
            norm_preds = [self.verifier.normalize_answer(p) for p in valid_preds]
            most_common = Counter(norm_preds).most_common(1)
            if most_common:
                major_ans = most_common[0][0]
                is_major_correct = self.verifier.is_equivalent(major_ans, extracted_gold)
                pass_metrics["major@100"] = 1.0 if is_major_correct else 0.0
            else:
                pass_metrics["major@100"] = 0.0
        else:
            pass_metrics["major@100"] = 0.0

        div_all, div_corr, uniq_ratio = self.div_calc.compute_metrics(
            outputs, correct_indices
        )

        return {
            "question": question,
            "gold": extracted_gold,
            "preds_sample": extracted_preds[:5],
            "n_correct": n_correct,
            "n_total": n_total,
            **pass_metrics,
            "diversity_all": div_all,
            "diversity_correct": div_corr,
            "unique_path_ratio": uniq_ratio,
        }