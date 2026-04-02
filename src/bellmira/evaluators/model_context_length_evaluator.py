import time
from typing import Dict, List, Optional

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient


class ModelContextLengthEvaluator(ModelEvaluatorInterface):
    """
    Measures response latency and token counts across contexts of increasing length.

    Accepts a ``contexts`` dict (``{label: text}``) so it works with any corpus.
    Use :mod:`bellmira.utils.context_utils` to build the dict from a file:

    .. code-block:: python

        from bellmira.utils.context_utils import (
            contexts_from_word_counts,   # generic — any text file
            contexts_from_files,         # one file per context
            contexts_from_bible,         # convenience wrapper for the Bible corpus
        )

        contexts = contexts_from_word_counts(my_text, word_counts=[500, 1000, 2000, 4000])
        evaluator = ModelContextLengthEvaluator(url=..., prompts=..., contexts=contexts)
    """

    def __init__(
        self,
        url: str,
        prompts: List[str],
        contexts: Dict[str, str],
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
        type: str = "chat",
    ):
        """
        Args:
            url:           Model server base URL.
            prompts:       Questions / instructions appended after each context.
            contexts:      Dict mapping a label to a context string.
                           Build with ``context_utils.*``.
            temperature:   Sampling temperature.
            system_prompt: Optional system prompt.
            json_schema:   Optional guided-JSON schema.
            type:          ``"chat"`` or ``"embedding"``.
        """
        self.model_url = url
        self.prompts = prompts
        self.prompt_context = contexts
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        self.type = type
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name()
        print("ModelContextLengthEvaluator initialized.")

    def evaluate(self, max_prompts: int = 2) -> Dict[str, List[Dict]]:
        print("ModelContextLengthEvaluator warming up model...")
        self.warm_up_model()
        print("ModelContextLengthEvaluator warm up model finished.")
        results_dict = {}
        for ref_key, context in self.prompt_context.items():
            results_dict[ref_key] = []
            for prompt in self.prompts[:max_prompts]:
                start_time = time.time()
                if self.type == "chat":
                    req = self.model_client.build_chat_request(
                        context + "\n" + prompt,
                        system_prompt=self.system_prompt,
                        model_name=self.model_name,
                        temperature=self.temperature,
                    )
                elif self.type == "embedding":
                    req = self.model_client.build_embedding_request(
                        input_text=context + "\n" + prompt,
                        model_name=self.model_name,
                    )
                result = self.model_client.send_request(req)
                end_time = time.time()
                if not result.ok:
                    results_dict[ref_key].append({
                        "Code": result.status_code,
                        "Message": result.reason,
                    })
                    return results_dict
                try:
                    result_json = result.json()
                    message = result_json.get("message") if result.status_code != 200 else None
                    usage = result_json.get("usage", {})
                except ValueError:
                    message = f"Non-JSON response: {result.text[:200]}"
                    usage = {}
                results_dict[ref_key].append({
                    "Code": result.status_code,
                    "Message": message,
                    "Execution_time": end_time - start_time,
                    "Total_tokens": usage.get("total_tokens"),
                    "Prompt_tokens": usage.get("prompt_tokens"),
                    "Completion_tokens": usage.get("completion_tokens"),
                })
        return results_dict

    def warm_up_model(self, warmup_count: int = 10, warmup_prompt: str = "Hello! Please respond quickly."):
        print(f"Warming up the model with {warmup_count} requests...")
        for i in range(warmup_count):
            try:
                if self.type == "chat":
                    req = self.model_client.build_chat_request(
                        warmup_prompt,
                        system_prompt=None,
                        model_name=self.model_name,
                        temperature=0,
                    )
                elif self.type == "embedding":
                    req = self.model_client.build_embedding_request(
                        input_text=warmup_prompt,
                        model_name=self.model_name,
                    )
                response = self.model_client.send_request(req)
                if response.ok:
                    print(f"Warmup request {i+1} succeeded.")
                else:
                    print(f"Warmup request {i+1} failed with code {response.status_code}.")
            except Exception as e:
                print(f"Warmup request {i+1} raised an error: {e}")

    def compute_averages(self, results_dict: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        avg_results = {}
        for key, results in results_dict.items():
            total_exec, total_tokens, prompt_tokens, comp_tokens = 0, 0, 0, 0
            n, token_entries = 0, 0
            error_msg = None
            for result in results:
                if "Execution_time" in result:
                    n += 1
                    total_exec += result["Execution_time"]
                    if result["Total_tokens"] is not None:
                        token_entries += 1
                        total_tokens += result["Total_tokens"]
                        prompt_tokens += result["Prompt_tokens"]
                        comp_tokens += result["Completion_tokens"]
                if result["Code"] != 200:
                    error_msg = f"{result['Code']} {result['Message']}"
            avg_result = {"Avg_execution_time": round(total_exec / n) if n else None}
            if token_entries:
                avg_result.update({
                    "Avg_total_tokens": round(total_tokens / token_entries),
                    "Avg_prompt_tokens": round(prompt_tokens / token_entries),
                    "Avg_completion_tokens": round(comp_tokens / token_entries),
                })
            if error_msg:
                avg_result["Errors"] = error_msg
            avg_results[key] = avg_result
        return avg_results

    def extract_threshold_metrics(
        self,
        data: dict,
        token_thresholds: List[int] = [1000, 2000, 4000, 7500, 12000, 16000, 20000, 24000, 32000],
    ) -> dict:
        averages = self.compute_averages(data)
        threshold_columns = [f"{t // 1000}k Tok Avg Time" for t in token_thresholds]
        row = {}
        entries = list(averages.values())
        for threshold, col_name in zip(token_thresholds, threshold_columns):
            min_exec_time = None
            min_completion_tokens = None
            min_prompt_tokens = float("inf")
            for entry in entries:
                avg_prompt_tokens = entry.get("Avg_prompt_tokens")
                exec_time = entry.get("Avg_execution_time")
                if avg_prompt_tokens is None or exec_time is None:
                    continue
                if avg_prompt_tokens >= threshold and avg_prompt_tokens < min_prompt_tokens:
                    min_prompt_tokens = avg_prompt_tokens
                    min_exec_time = exec_time
                    min_completion_tokens = entry.get("Avg_completion_tokens")
            if min_exec_time is not None:
                row[col_name] = {
                    "Avg_Exec_Time": min_exec_time,
                    "Avg_prompt_tokens": min_prompt_tokens,
                    "Avg_completion_tokens": min_completion_tokens,
                }
        row["Error"] = averages.get("Errors", None)
        return row
