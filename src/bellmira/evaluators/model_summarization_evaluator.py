import time
from collections import Counter
from typing import List, Dict, Optional, Tuple

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient


# ---------------------------------------------------------------------------
# ROUGE implementation (no external dependencies)
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    hyp_counts = Counter(_ngrams(hyp_tokens, n))
    ref_counts = Counter(_ngrams(ref_tokens, n))
    overlap = sum((hyp_counts & ref_counts).values())
    precision = overlap / sum(hyp_counts.values()) if hyp_counts else 0.0
    recall = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Space-efficient LCS using two rows of DP."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compression_ratio(document: str, summary: str) -> float:
    doc_words = len(document.split())
    sum_words = len(summary.split())
    return round(sum_words / doc_words, 4) if doc_words > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelSummarizationEvaluator(ModelEvaluatorInterface):
    """
    Evaluates summarization quality using ROUGE-1, ROUGE-2, and ROUGE-L.

    Accepts a list of (document, reference_summary) pairs.  For each pair the
    evaluator prompts the model to produce a summary, then scores it against
    the reference using ROUGE metrics computed without external dependencies.

    Additional metric:
      - Compression ratio: summary word count / document word count
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a concise summarization assistant. "
        "When given a document, output only the summary — no preamble, no commentary."
    )
    DEFAULT_USER_TEMPLATE = "Summarize the following document in a few sentences:\n\n{document}"

    def __init__(
        self,
        url: str,
        pairs: List[Tuple[str, str]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        """
        Args:
            url: Base URL of the model server.
            pairs: List of (document, reference_summary) tuples.
            system_prompt: Override the default summarization system prompt.
            user_template: Override the default user message template.
                           Must contain a ``{document}`` placeholder.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for the generated summary.
        """
        if not pairs:
            raise ValueError("At least one (document, reference_summary) pair is required.")
        self.model_url = url
        self.pairs = pairs
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or self.DEFAULT_USER_TEMPLATE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name()

    def _summarize(self, document: str) -> Tuple[Optional[str], float, Optional[int], Optional[int]]:
        """
        Calls the model to summarize document.
        Returns (summary_text, latency, prompt_tokens, completion_tokens).
        """
        user_prompt = self.user_template.format(document=document)
        req = self.model_client.build_chat_request(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        start = time.time()
        response = self.model_client.send_request(req)
        latency = time.time() - start

        if not response.ok:
            return None, latency, None, None

        try:
            body = response.json()
            summary = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return summary, latency, usage.get("prompt_tokens"), usage.get("completion_tokens")
        except (KeyError, IndexError, ValueError) as e:
            print(f"Failed to parse summarization response: {e}")
            return None, latency, None, None

    def evaluate(self, max_prompts: int = None) -> List[Dict]:
        """
        Summarize each document and compute ROUGE scores against the reference.
        max_prompts limits the number of pairs evaluated (None = all).
        Returns a list of per-pair result dicts.
        """
        print("ModelSummarizationEvaluator: warming up model...")
        self.warm_up_model()
        print("ModelSummarizationEvaluator: warm-up finished.")
        results = []
        pairs = self.pairs if max_prompts is None else self.pairs[:max_prompts]

        for i, (document, reference) in enumerate(pairs):
            print(f"  Pair {i + 1}/{len(pairs)}: doc='{document[:60]}...'")
            summary, latency, prompt_tokens, completion_tokens = self._summarize(document)

            if summary is None:
                results.append({
                    "Error": "Summarization request failed",
                    "Latency": round(latency, 4),
                    "ROUGE1_F1": None,
                    "ROUGE2_F1": None,
                    "ROUGEL_F1": None,
                    "Compression_ratio": None,
                })
                continue

            r1 = rouge_n(summary, reference, n=1)
            r2 = rouge_n(summary, reference, n=2)
            rl = rouge_l(summary, reference)
            comp = compression_ratio(document, summary)

            result = {
                "Latency": round(latency, 4),
                "Prompt_tokens": prompt_tokens,
                "Completion_tokens": completion_tokens,
                "Compression_ratio": comp,
                "ROUGE1_precision": r1["precision"],
                "ROUGE1_recall": r1["recall"],
                "ROUGE1_F1": r1["f1"],
                "ROUGE2_precision": r2["precision"],
                "ROUGE2_recall": r2["recall"],
                "ROUGE2_F1": r2["f1"],
                "ROUGEL_precision": rl["precision"],
                "ROUGEL_recall": rl["recall"],
                "ROUGEL_F1": rl["f1"],
                "Generated_summary": summary,
                "Reference_summary": reference,
            }
            print(
                f"    R1={r1['f1']:.4f}  R2={r2['f1']:.4f}  RL={rl['f1']:.4f}  "
                f"compression={comp:.3f}  latency={latency:.2f}s"
            )
            results.append(result)
        return results

    def warm_up_model(self, warmup_count: int = 3, warmup_prompt: str = "Summarize: The sky is blue."):
        print(f"Warming up with {warmup_count} requests...")
        for i in range(warmup_count):
            try:
                req = self.model_client.build_chat_request(
                    user_prompt=warmup_prompt,
                    system_prompt=self.system_prompt,
                    model_name=self.model_name,
                    temperature=0,
                    max_tokens=32,
                )
                response = self.model_client.send_request(req)
                if response.ok:
                    print(f"  Warmup {i + 1} succeeded.")
                else:
                    print(f"  Warmup {i + 1} failed: {response.status_code}")
            except Exception as e:
                print(f"  Warmup {i + 1} raised: {e}")

    def extract_threshold_metrics(self, results: List[Dict]) -> Dict:
        """
        Aggregate per-pair results into overall means.
        """
        valid = [r for r in results if r.get("ROUGE1_F1") is not None]
        if not valid:
            return {"Error": "No valid results to aggregate"}

        n = len(valid)

        def mean(key):
            vals = [r[key] for r in valid if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        errors = [r.get("Error") for r in results if r.get("Error")]

        return {
            "Pairs_evaluated": n,
            "Avg_ROUGE1_F1": mean("ROUGE1_F1"),
            "Avg_ROUGE2_F1": mean("ROUGE2_F1"),
            "Avg_ROUGEL_F1": mean("ROUGEL_F1"),
            "Avg_ROUGE1_precision": mean("ROUGE1_precision"),
            "Avg_ROUGE1_recall": mean("ROUGE1_recall"),
            "Avg_compression_ratio": mean("Compression_ratio"),
            "Avg_latency": mean("Latency"),
            "Avg_prompt_tokens": mean("Prompt_tokens"),
            "Avg_completion_tokens": mean("Completion_tokens"),
            "Error": "; ".join(errors) if errors else None,
        }
