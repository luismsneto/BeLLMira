import logging
import time
from typing import List, Dict, Optional, Tuple

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient
from bellmira.utils.text_metrics import rouge_n, rouge_l, compression_ratio
from bellmira.utils.metrics_utils import mean_of_key

logger = logging.getLogger(__name__)

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
            logger.warning("Failed to parse summarization response: %s", e)
            return None, latency, None, None

    def evaluate(self, max_prompts: int = None) -> List[Dict]:
        """
        Summarize each document and compute ROUGE scores against the reference.
        max_prompts limits the number of pairs evaluated (None = all).
        Returns a list of per-pair result dicts.
        """
        logger.info("ModelSummarizationEvaluator: warming up model...")
        self.warm_up_model()
        logger.info("ModelSummarizationEvaluator: warm-up finished.")
        results = []
        pairs = self.pairs if max_prompts is None else self.pairs[:max_prompts]

        for i, (document, reference) in enumerate(pairs):
            logger.info("Pair %d/%d: doc='%s...'", i + 1, len(pairs), document[:60])
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
            logger.debug(
                "R1=%.4f  R2=%.4f  RL=%.4f  compression=%.3f  latency=%.2fs",
                r1["f1"], r2["f1"], rl["f1"], comp, latency,
            )
            results.append(result)
        return results

    def warm_up_model(self, warmup_count: int = 3, warmup_prompt: str = "Summarize: The sky is blue."):
        logger.debug("Warming up with %d requests...", warmup_count)
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
                    logger.debug("Warmup %d succeeded.", i + 1)
                else:
                    logger.warning("Warmup %d failed: %s", i + 1, response.status_code)
            except Exception as e:
                logger.warning("Warmup %d raised: %s", i + 1, e)

    def extract_threshold_metrics(self, results: List[Dict]) -> Dict:
        """
        Aggregate per-pair results into overall means.
        """
        valid = [r for r in results if r.get("ROUGE1_F1") is not None]
        if not valid:
            return {"Error": "No valid results to aggregate"}

        errors = [r.get("Error") for r in results if r.get("Error")]

        return {
            "Pairs_evaluated": len(valid),
            "Avg_ROUGE1_F1": mean_of_key(valid, "ROUGE1_F1"),
            "Avg_ROUGE2_F1": mean_of_key(valid, "ROUGE2_F1"),
            "Avg_ROUGEL_F1": mean_of_key(valid, "ROUGEL_F1"),
            "Avg_ROUGE1_precision": mean_of_key(valid, "ROUGE1_precision"),
            "Avg_ROUGE1_recall": mean_of_key(valid, "ROUGE1_recall"),
            "Avg_compression_ratio": mean_of_key(valid, "Compression_ratio"),
            "Avg_latency": mean_of_key(valid, "Latency"),
            "Avg_prompt_tokens": mean_of_key(valid, "Prompt_tokens"),
            "Avg_completion_tokens": mean_of_key(valid, "Completion_tokens"),
            "Error": "; ".join(errors) if errors else None,
        }
