import time
from typing import Dict, List, Optional

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient


class ModelTTFTEvaluator(ModelEvaluatorInterface):
    """
    Measures Time-To-First-Token (TTFT) and streaming throughput across contexts
    of increasing length.

    Accepts a ``contexts`` dict (``{label: text}``) so it works with any corpus.
    Use :mod:`bellmira.utils.context_utils` to build the dict from a file:

    .. code-block:: python

        from bellmira.utils.context_utils import (
            contexts_from_word_counts,   # generic — any text file
            contexts_from_files,         # one file per context
            contexts_from_bible,         # convenience wrapper for the Bible corpus
        )

        contexts = contexts_from_word_counts(my_text, word_counts=[500, 1000, 2000, 4000])
        evaluator = ModelTTFTEvaluator(url=..., prompts=..., contexts=contexts)
    """

    def __init__(
        self,
        url: str,
        prompts: List[str],
        contexts: Dict[str, str],
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            url:           Model server base URL.
            prompts:       Questions / instructions appended after each context.
            contexts:      Dict mapping a label to a context string.
                           Build with ``context_utils.*``.
            temperature:   Sampling temperature.
            system_prompt: Optional system prompt.
        """
        self.model_url = url
        self.prompts = prompts
        self.prompt_context = contexts
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name()

    def _measure_stream(self, full_prompt: str) -> Dict:
        """Send one streaming request and return TTFT, total time, and output char count."""
        req = self.model_client.build_chat_request(
            full_prompt,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        ttft = None
        total_chars = 0
        start = time.time()
        try:
            for chunk in self.model_client.stream_chat_response(req):
                if ttft is None:
                    ttft = time.time() - start
                total_chars += len(chunk)
        except Exception as e:
            return {"Error": str(e), "TTFT": None, "Total_time": None, "Output_chars": 0}
        total_time = time.time() - start
        return {
            "TTFT": round(ttft, 4) if ttft is not None else None,
            "Total_time": round(total_time, 4),
            "Output_chars": total_chars,
        }

    def evaluate(self, max_prompts: int = 1) -> Dict[str, List[Dict]]:
        print("ModelTTFTEvaluator: warming up model...")
        self.warm_up_model()
        print("ModelTTFTEvaluator: warm-up finished.")
        results_dict = {}
        for ref_key, context in self.prompt_context.items():
            results_dict[ref_key] = []
            for prompt in self.prompts[:max_prompts]:
                full_prompt = context + "\n" + prompt
                print(f"  [{ref_key}] streaming prompt: {prompt[:40]}...")
                stats = self._measure_stream(full_prompt)
                results_dict[ref_key].append(stats)
                print(f"  [{ref_key}] TTFT={stats.get('TTFT')}s  Total={stats.get('Total_time')}s  Chars={stats.get('Output_chars')}")
        return results_dict

    def warm_up_model(self, warmup_count: int = 5, warmup_prompt: str = "Hello! Please respond quickly."):
        print(f"Warming up with {warmup_count} streaming requests...")
        for i in range(warmup_count):
            try:
                req = self.model_client.build_chat_request(
                    warmup_prompt,
                    system_prompt=None,
                    model_name=self.model_name,
                    temperature=0,
                )
                for _ in self.model_client.stream_chat_response(req):
                    pass
                print(f"  Warmup {i + 1} done.")
            except Exception as e:
                print(f"  Warmup {i + 1} failed: {e}")

    def compute_averages(self, results_dict: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        averages = {}
        for key, entries in results_dict.items():
            ttft_values = [e["TTFT"] for e in entries if e.get("TTFT") is not None]
            total_values = [e["Total_time"] for e in entries if e.get("Total_time") is not None]
            char_values = [e["Output_chars"] for e in entries if e.get("Output_chars") is not None]
            errors = [e["Error"] for e in entries if "Error" in e]
            averages[key] = {
                "Avg_TTFT": round(sum(ttft_values) / len(ttft_values), 4) if ttft_values else None,
                "Avg_total_time": round(sum(total_values) / len(total_values), 4) if total_values else None,
                "Avg_output_chars": round(sum(char_values) / len(char_values), 1) if char_values else None,
                "Errors": errors if errors else None,
            }
        return averages

    def extract_threshold_metrics(self, results: Dict[str, List[Dict]]) -> Dict:
        averages = self.compute_averages(results)
        flat = {}
        errors = []
        for key, metrics in averages.items():
            if metrics["Avg_TTFT"] is not None:
                flat[f"{key}_TTFT(s)"] = metrics["Avg_TTFT"]
            if metrics["Avg_total_time"] is not None:
                flat[f"{key}_Total_time(s)"] = metrics["Avg_total_time"]
            if metrics["Errors"]:
                errors.extend(metrics["Errors"])
        if errors:
            flat["Error"] = "; ".join(errors)
        return flat
