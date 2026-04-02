import json
import time
from typing import Dict, List, Optional, Tuple

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.evaluators.model_summarization_evaluator import rouge_l
from bellmira.llm_model.llm_model_client import ModelClient

# Keywords used to detect refusal responses heuristically
_REFUSAL_PHRASES = (
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i'm not able", "i am not able", "i won't", "i will not",
    "sorry, i", "i apologize",
)


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


class ModelRegressionEvaluator(ModelEvaluatorInterface):
    """
    Compares a candidate model endpoint against a baseline on the same prompt set.

    For each prompt both models are queried and their outputs are compared across:
      - Exact match
      - Output drift (ROUGE-L of candidate vs baseline output as reference)
      - Token count delta
      - Latency delta
      - Refusal rate
      - JSON validity (when json_schema is provided)

    The baseline and candidate can be different model sizes, fine-tunes, or the
    same model on different environments (e.g. dv vs qa).
    """

    def __init__(
        self,
        baseline_url: str,
        candidate_url: str,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_schema: Optional[dict] = None,
        max_tokens: int = 1000,
    ):
        """
        Args:
            baseline_url: Base URL of the reference model server.
            candidate_url: Base URL of the model being evaluated.
            prompts: List of user prompts to run against both models.
            system_prompt: Optional system prompt applied to both models.
            temperature: Sampling temperature (use 0.0 for deterministic comparison).
            json_schema: If set, responses are checked for JSON validity.
            max_tokens: Maximum tokens per response.
        """
        if not prompts:
            raise ValueError("At least one prompt is required.")
        self.prompts = prompts
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.json_schema = json_schema
        self.max_tokens = max_tokens

        self.baseline_client = ModelClient(base_url=baseline_url)
        self.candidate_client = ModelClient(base_url=candidate_url)
        self.baseline_model_name = self.baseline_client.get_model_name()
        self.candidate_model_name = self.candidate_client.get_model_name()

        print(f"Baseline:  {baseline_url}  →  {self.baseline_model_name}")
        print(f"Candidate: {candidate_url}  →  {self.candidate_model_name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(
        self, client: ModelClient, model_name: str, prompt: str
    ) -> Tuple[Optional[str], float, Optional[int], Optional[int], int]:
        """
        Query one model.  Returns (output_text, latency, prompt_tokens, completion_tokens, status_code).
        """
        req = client.build_chat_request(
            user_prompt=prompt,
            system_prompt=self.system_prompt,
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_schema=self.json_schema,
        )
        start = time.time()
        response = client.send_request(req)
        latency = time.time() - start

        if not response.ok:
            return None, latency, None, None, response.status_code

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return (
                text,
                latency,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                response.status_code,
            )
        except (KeyError, IndexError, ValueError) as e:
            print(f"Failed to parse response: {e}")
            return None, latency, None, None, response.status_code

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def warm_up_model(self, warmup_count: int = 3, warmup_prompt: str = "Hello! Please respond quickly."):
        print(f"Warming up baseline and candidate with {warmup_count} requests each...")
        for client, name in (
            (self.baseline_client, "baseline"),
            (self.candidate_client, "candidate"),
        ):
            model_name = (
                self.baseline_model_name if name == "baseline" else self.candidate_model_name
            )
            for i in range(warmup_count):
                try:
                    req = client.build_chat_request(
                        user_prompt=warmup_prompt,
                        system_prompt=None,
                        model_name=model_name,
                        temperature=0,
                        max_tokens=32,
                    )
                    response = client.send_request(req)
                    status = "ok" if response.ok else response.status_code
                    print(f"  [{name}] warmup {i + 1}: {status}")
                except Exception as e:
                    print(f"  [{name}] warmup {i + 1} raised: {e}")

    def evaluate(self, max_prompts: int = None) -> List[Dict]:
        """
        Run all prompts against both models and return per-prompt comparison dicts.
        max_prompts limits the number of prompts evaluated (None = all).
        """
        print("ModelRegressionEvaluator: warming up models...")
        self.warm_up_model()
        print("ModelRegressionEvaluator: warm-up finished.")

        prompts = self.prompts if max_prompts is None else self.prompts[:max_prompts]
        results = []

        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i + 1}/{len(prompts)}: '{prompt[:60]}...'")

            b_text, b_lat, b_ptok, b_ctok, b_code = self._query(
                self.baseline_client, self.baseline_model_name, prompt
            )
            c_text, c_lat, c_ptok, c_ctok, c_code = self._query(
                self.candidate_client, self.candidate_model_name, prompt
            )

            # Output drift: ROUGE-L of candidate vs baseline (baseline = reference)
            rougel_drift = None
            if b_text and c_text:
                rougel_drift = rouge_l(hypothesis=c_text, reference=b_text)["f1"]

            exact_match = (b_text is not None and c_text is not None and b_text == c_text)

            # Token deltas (positive = candidate uses more tokens)
            ctok_delta = (
                (c_ctok - b_ctok) if (c_ctok is not None and b_ctok is not None) else None
            )
            latency_delta = round(c_lat - b_lat, 4)

            # Refusal detection
            b_refusal = _is_refusal(b_text) if b_text else None
            c_refusal = _is_refusal(c_text) if c_text else None

            # JSON validity (only meaningful when json_schema is set)
            b_json_valid = _is_valid_json(b_text) if (b_text and self.json_schema) else None
            c_json_valid = _is_valid_json(c_text) if (c_text and self.json_schema) else None

            result = {
                "Prompt": prompt[:120],
                "Baseline_output": b_text,
                "Candidate_output": c_text,
                "Exact_match": exact_match,
                "ROUGEL_drift": rougel_drift,
                "Baseline_latency": round(b_lat, 4),
                "Candidate_latency": round(c_lat, 4),
                "Latency_delta": latency_delta,
                "Baseline_completion_tokens": b_ctok,
                "Candidate_completion_tokens": c_ctok,
                "Completion_tokens_delta": ctok_delta,
                "Baseline_refusal": b_refusal,
                "Candidate_refusal": c_refusal,
                "Baseline_status": b_code,
                "Candidate_status": c_code,
            }
            if self.json_schema:
                result["Baseline_json_valid"] = b_json_valid
                result["Candidate_json_valid"] = c_json_valid

            print(
                f"    exact={exact_match}  rouge_drift={rougel_drift}  "
                f"lat_delta={latency_delta:+.3f}s  ctok_delta={ctok_delta}"
            )
            results.append(result)

        return results

    def extract_threshold_metrics(self, results: List[Dict]) -> Dict:
        """
        Aggregate per-prompt results into a flat summary dict ready for MLflow logging.

        Sign convention: positive latency/token deltas mean the candidate is slower/larger.
        ROUGE-L drift closer to 1.0 means outputs are near-identical; closer to 0.0 means
        significant output change.
        """
        if not results:
            return {"Error": "No results to aggregate"}

        n = len(results)
        valid = [r for r in results if r["Baseline_output"] and r["Candidate_output"]]
        n_valid = len(valid)

        def mean(key):
            vals = [r[key] for r in valid if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def rate(key, value=True):
            vals = [r[key] for r in valid if r.get(key) is not None]
            return round(sum(1 for v in vals if v == value) / len(vals), 4) if vals else None

        b_errors = sum(1 for r in results if r["Baseline_status"] != 200)
        c_errors = sum(1 for r in results if r["Candidate_status"] != 200)

        metrics = {
            "Baseline_model": self.baseline_model_name,
            "Candidate_model": self.candidate_model_name,
            "Prompts_evaluated": n,
            "Valid_comparisons": n_valid,
            "Baseline_error_count": b_errors,
            "Candidate_error_count": c_errors,
            # Quality
            "Exact_match_rate": rate("Exact_match", True),
            "Avg_ROUGEL_drift": mean("ROUGEL_drift"),
            # Latency
            "Avg_baseline_latency": mean("Baseline_latency"),
            "Avg_candidate_latency": mean("Candidate_latency"),
            "Avg_latency_delta": mean("Latency_delta"),
            "Avg_latency_delta_pct": (
                round(mean("Latency_delta") / mean("Baseline_latency") * 100, 2)
                if mean("Baseline_latency") else None
            ),
            # Tokens
            "Avg_baseline_completion_tokens": mean("Baseline_completion_tokens"),
            "Avg_candidate_completion_tokens": mean("Candidate_completion_tokens"),
            "Avg_completion_tokens_delta": mean("Completion_tokens_delta"),
            # Behaviour
            "Baseline_refusal_rate": rate("Baseline_refusal", True),
            "Candidate_refusal_rate": rate("Candidate_refusal", True),
        }

        if self.json_schema:
            metrics["Baseline_json_valid_rate"] = rate("Baseline_json_valid", True)
            metrics["Candidate_json_valid_rate"] = rate("Candidate_json_valid", True)

        return metrics
