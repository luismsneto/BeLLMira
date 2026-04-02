import time
import threading
import itertools
from typing import Dict, List, Optional

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient


class ModelParallelLoadEvaluator(ModelEvaluatorInterface):
    """
    Measures throughput and latency under parallel request load.

    Accepts a ``contexts`` dict (``{label: text}``) so it works with any corpus.
    Use :mod:`bellmira.utils.context_utils` to build the dict from a file:

    .. code-block:: python

        from bellmira.utils.context_utils import (
            contexts_from_word_counts,   # generic — any text file
            contexts_from_files,         # one file per context
            contexts_from_bible,         # convenience wrapper for the Bible corpus
        )

        contexts = contexts_from_word_counts(my_text, word_counts=[500, 1000, 2000, 4000])
        evaluator = ModelParallelLoadEvaluator(url=..., prompts=...,
                                               contexts=contexts,
                                               concurrency_levels=[1, 4, 8, 16])
    """

    def __init__(
        self,
        url: str,
        prompts: List[str],
        contexts: Dict[str, str],
        concurrency_levels: List[int],
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ):
        """
        Args:
            url:                Model server base URL.
            prompts:            Pool of prompts cycled across concurrent threads.
            contexts:           Dict mapping a label to a context string.
                                Build with ``context_utils.*``.
            concurrency_levels: List of concurrency values to sweep (e.g. ``[1, 2, 4, 8]``).
            temperature:        Sampling temperature.
            system_prompt:      Optional system prompt.
            json_schema:        Optional guided-JSON schema.
        """
        self.model_url = url
        self.prompts = prompts
        self.prompt_context = contexts
        self.concurrency_levels = concurrency_levels
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name()

    def _run_concurrent_requests(self, context: str, concurrency: int) -> Dict:
        latencies = []
        status_counts = {}
        threads = []
        prompt_tokens_list = []
        completion_tokens_list = []
        lock = threading.Lock()
        prompt_cycle = itertools.cycle(self.prompts)

        def worker(prompt):
            req = self.model_client.build_chat_request(
                context + "\n" + prompt,
                system_prompt=self.system_prompt,
                model_name=self.model_name,
                temperature=0,
            )
            start = time.time()
            response = self.model_client.send_request(req)
            latency = time.time() - start
            prompt_tokens = response.json().get("usage", {}).get("prompt_tokens") if response.status_code == 200 else None
            completion_tokens = response.json().get("usage", {}).get("completion_tokens") if response.status_code == 200 else None
            with lock:
                latencies.append(latency)
                if completion_tokens:
                    completion_tokens_list.append(completion_tokens)
                if prompt_tokens:
                    prompt_tokens_list.append(prompt_tokens)
                status_counts[response.status_code] = status_counts.get(response.status_code, 0) + 1

        start_time = time.time()
        for i in range(concurrency):
            prompt = next(prompt_cycle)
            print(f"Concurrency {concurrency}: Running thread for prompt: {prompt[:20]}")
            t = threading.Thread(target=worker, args=(prompt,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print(f"Concurrency {concurrency}: Joined all threads!")
        duration = time.time() - start_time

        throughput = concurrency / duration if duration > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else None
        avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list) if completion_tokens_list else None
        result = {
            "Avg_latency": round(avg_latency, 3) if avg_latency else None,
            "Avg_prompt_tokens": round(avg_prompt_tokens, 1) if avg_prompt_tokens else None,
            "Avg_completion_tokens": round(avg_completion_tokens, 1) if avg_completion_tokens else None,
            "Throughput": round(throughput, 2),
            "Requests": len(latencies),
            "Status_codes": status_counts,
        }
        print(f"Concurrency {concurrency}: Result: {result}")
        return result

    def evaluate(self, max_prompts: int = 1) -> Dict[str, Dict[str, Dict]]:
        results_dict = {}
        self.warm_up_model()
        for ref_key, context in self.prompt_context.items():
            for concurrency in self.concurrency_levels:
                key = f"{concurrency}_par_req"
                print(f"Running {key} with {ref_key} context...")
                stats = self._run_concurrent_requests(context, concurrency)
                error_codes = [code for code, count in stats["Status_codes"].items() if code >= 400 and count > 0]
                if error_codes:
                    print(f"Error {error_codes} detected for {key} with {ref_key} context. Skipping...")
                    return results_dict
                avg_tokens = stats["Avg_prompt_tokens"]
                label = f"{avg_tokens // 1000}k_tokens"
                if key not in results_dict:
                    results_dict[key] = {}
                results_dict[key][label] = stats
        return results_dict

    def warm_up_model(self, warmup_count: int = 10, warmup_prompt: str = "Hello! Please respond quickly."):
        print(f"Warming up the model with {warmup_count} requests...")
        for i in range(warmup_count):
            try:
                req = self.model_client.build_chat_request(
                    warmup_prompt,
                    system_prompt=None,
                    model_name=self.model_name,
                    temperature=0,
                )
                response = self.model_client.send_request(req)
                if response.ok:
                    print(f"Warmup request {i+1} succeeded.")
                else:
                    print(f"Warmup request {i+1} failed with code {response.status_code}.")
            except Exception as e:
                print(f"Warmup request {i+1} raised an error: {e}")

    def extract_threshold_metrics(self, results: dict) -> dict:
        extracted = {}
        metrics_to_extract = ["Avg_latency", "Throughput", "Avg_prompt_tokens", "Avg_completion_tokens"]
        for concurrency_key, token_data in results.items():
            extracted[concurrency_key] = []
            for token_label, metrics in token_data.items():
                extracted[concurrency_key].append({k: metrics[k] for k in metrics if k in metrics_to_extract})
        return extracted
