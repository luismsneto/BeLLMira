import logging
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelEvaluatorInterface(ABC):

    @abstractmethod
    def evaluate(self, max_prompts: int = 1) -> Dict:
        """Run evaluation using the model and return results."""
        pass

    @abstractmethod
    def extract_threshold_metrics(self, results: dict) -> dict:
        """Collapse raw results into a flat metrics dict ready for logging."""
        pass

    def warm_up_model(self, warmup_count: int = 10, warmup_prompt: str = "Hello! Please respond quickly.") -> None:
        """Send warmup chat requests to prime the model server before evaluation.

        Subclasses that use a different request type (embedding, streaming, multiple
        clients) must override this method.
        """
        logger.debug("Warming up the model with %d requests...", warmup_count)
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
                    logger.debug("Warmup request %d succeeded.", i + 1)
                else:
                    logger.warning("Warmup request %d failed with code %s.", i + 1, response.status_code)
            except Exception as e:
                logger.warning("Warmup request %d raised an error: %s", i + 1, e)

    def compute_averages(self, results_dict: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Average Execution_time and token counts across per-request result dicts.

        Subclasses that measure different fields (e.g. TTFT) must override this method.
        """
        avg_results = {}
        for key, results in results_dict.items():
            total_exec, total_tokens, prompt_tokens, comp_tokens = 0.0, 0, 0, 0
            n, token_entries = 0, 0
            error_msg: Optional[str] = None
            for result in results:
                code = result.get("Code")
                if code is not None and code != 200:
                    error_msg = f"{code} {result.get('Message')}"
                if "Execution_time" in result:
                    n += 1
                    total_exec += result["Execution_time"]
                    if result.get("Total_tokens") is not None:
                        token_entries += 1
                        total_tokens += result["Total_tokens"]
                        prompt_tokens += result.get("Prompt_tokens", 0)
                        comp_tokens += result.get("Completion_tokens", 0)
            avg_result: Dict = {"Avg_execution_time": round(total_exec / n, 2) if n else None}
            if token_entries:
                avg_result["Avg_total_tokens"] = round(total_tokens / token_entries, 2)
                avg_result["Avg_prompt_tokens"] = round(prompt_tokens / token_entries, 2)
                avg_result["Avg_completion_tokens"] = round(comp_tokens / token_entries, 2)
            if error_msg:
                avg_result["Errors"] = error_msg
            avg_results[key] = avg_result
        return avg_results
