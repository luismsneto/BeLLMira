import time
from typing import List, Dict, Optional, Tuple

import numpy as np

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


class ModelEmbeddingQualityEvaluator(ModelEvaluatorInterface):
    """
    Evaluates embedding model quality using relevance triplets.

    Each triplet is (query, relevant_document, irrelevant_document).
    For each triplet the evaluator:
      - Embeds all three texts
      - Computes cosine similarity between (query, relevant) and (query, irrelevant)
      - Records the similarity gap and whether the relevant doc ranked higher

    Aggregate metrics:
      - Accuracy: % of triplets where sim(q, relevant) > sim(q, irrelevant)
      - Mean similarity gap: mean(sim_relevant - sim_irrelevant)
      - Mean relevant similarity
      - Mean irrelevant similarity
    """

    def __init__(
        self,
        url: str,
        triplets: List[Tuple[str, str, str]],
        embedding_model_name: str = "/app/model/embedding",
        temperature: float = 0.0,
    ):
        if not triplets:
            raise ValueError("At least one (query, relevant, irrelevant) triplet is required.")
        self.model_url = url
        self.triplets = triplets
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.model_client = ModelClient(base_url=self.model_url)

    def _embed(self, text: str) -> Tuple[Optional[List[float]], float, Optional[int]]:
        """Returns (embedding_vector, latency_seconds, token_count)."""
        req = self.model_client.build_embedding_request(
            input_text=text,
            model_name=self.embedding_model_name,
        )
        start = time.time()
        response = self.model_client.send_request(req)
        latency = time.time() - start

        if not response.ok:
            print(f"Embedding request failed: {response.status_code} {response.reason}")
            return None, latency, None

        try:
            body = response.json()
            vector = body["data"][0]["embedding"]
            tokens = body.get("usage", {}).get("total_tokens")
            return vector, latency, tokens
        except (KeyError, IndexError, ValueError) as e:
            print(f"Failed to parse embedding response: {e}")
            return None, latency, None

    def evaluate(self, max_prompts: int = None) -> List[Dict]:
        """
        Evaluate each triplet. Returns a list of per-triplet result dicts.
        max_prompts limits the number of triplets evaluated (None = all).
        """
        print("ModelEmbeddingQualityEvaluator: warming up model...")
        self.warm_up_model()
        print("ModelEmbeddingQualityEvaluator: warm-up finished.")
        results = []
        triplets = self.triplets if max_prompts is None else self.triplets[:max_prompts]
        for i, (query, relevant, irrelevant) in enumerate(triplets):
            print(f"  Triplet {i + 1}/{len(triplets)}: query='{query[:40]}...'")
            q_vec, q_lat, q_tok = self._embed(query)
            r_vec, r_lat, r_tok = self._embed(relevant)
            ir_vec, ir_lat, ir_tok = self._embed(irrelevant)

            if q_vec is None or r_vec is None or ir_vec is None:
                results.append({
                    "Query": query[:80],
                    "Error": "One or more embeddings failed",
                    "Sim_relevant": None,
                    "Sim_irrelevant": None,
                    "Sim_gap": None,
                    "Correct_rank": None,
                    "Avg_latency": round((q_lat + r_lat + ir_lat) / 3, 4),
                })
                continue

            sim_rel = cosine_similarity(q_vec, r_vec)
            sim_irr = cosine_similarity(q_vec, ir_vec)
            gap = sim_rel - sim_irr

            result = {
                "Query": query[:80],
                "Sim_relevant": round(sim_rel, 6),
                "Sim_irrelevant": round(sim_irr, 6),
                "Sim_gap": round(gap, 6),
                "Correct_rank": sim_rel > sim_irr,
                "Avg_latency": round((q_lat + r_lat + ir_lat) / 3, 4),
                "Query_tokens": q_tok,
            }
            print(f"    sim_rel={sim_rel:.4f}  sim_irr={sim_irr:.4f}  gap={gap:.4f}  correct={result['Correct_rank']}")
            results.append(result)
        return results

    def warm_up_model(self, warmup_count: int = 5, warmup_prompt: str = "Hello world."):
        print(f"Warming up with {warmup_count} embedding requests...")
        for i in range(warmup_count):
            try:
                req = self.model_client.build_embedding_request(
                    input_text=warmup_prompt,
                    model_name=self.embedding_model_name,
                )
                response = self.model_client.send_request(req)
                if response.ok:
                    print(f"  Warmup {i + 1} succeeded.")
                else:
                    print(f"  Warmup {i + 1} failed: {response.status_code}")
            except Exception as e:
                print(f"  Warmup {i + 1} raised: {e}")

    def extract_threshold_metrics(self, results: List[Dict]) -> Dict:
        valid = [r for r in results if r.get("Sim_gap") is not None]
        if not valid:
            return {"Error": "No valid results to aggregate"}

        n = len(valid)
        accuracy = round(sum(1 for r in valid if r["Correct_rank"]) / n, 4)
        mean_gap = round(sum(r["Sim_gap"] for r in valid) / n, 6)
        mean_sim_rel = round(sum(r["Sim_relevant"] for r in valid) / n, 6)
        mean_sim_irr = round(sum(r["Sim_irrelevant"] for r in valid) / n, 6)
        mean_latency = round(sum(r["Avg_latency"] for r in valid) / n, 4)

        errors = [r.get("Error") for r in results if r.get("Error")]

        return {
            "Triplets_evaluated": n,
            "Accuracy": accuracy,
            "Mean_sim_gap": mean_gap,
            "Mean_sim_relevant": mean_sim_rel,
            "Mean_sim_irrelevant": mean_sim_irr,
            "Avg_latency": mean_latency,
            "Error": "; ".join(errors) if errors else None,
        }
