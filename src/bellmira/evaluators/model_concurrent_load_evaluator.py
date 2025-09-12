import time
import requests
import threading
from typing import List, Dict, Optional
import json
import random
import itertools

from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient

def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='latin-1') as file:
        return file.read()

def extract_bible_chapters(text,  chapter_numbers: List[int], book: str = "Genesis") -> Dict[str, str]:
    references = {}
    for num in chapter_numbers:
        key = f"bible_old_testament_U{book}{num}"
        verse_ref = f"{book} {num}:1\t"
        references[key] = text.split(verse_ref)[0]
    return references

class ModelParallelLoadEvaluator(ModelEvaluatorInterface):
    def __init__(
        self,
        bible_path: str,
        url: str,
        prompts: List[str],
        chapter_numbers: List[int],
        concurrency_levels: List[int],
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None
    ):
        self.bible_path = bible_path
        self.model_url = url
        self.prompts = prompts
        self.concurrency_levels = concurrency_levels
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name( )
        self.prompt_context = self._build_prompt_context(file_path=bible_path, chapter_numbers=chapter_numbers)
    
    def _build_prompt_context(self, file_path: str, chapter_numbers = List[int]) -> str:
        bible_text = read_text_file(file_path)
        return extract_bible_chapters(text=bible_text, chapter_numbers=chapter_numbers)
    
    def _run_concurrent_requests(self, context: str, concurrency: int) -> Dict:
        latencies = []
        status_counts = {}
        threads = []
        prompt_tokens_list = []
        completion_tokens_list = []
        prompt_cycle = itertools.cycle(self.prompts)
        def worker(prompt):
            req = self.model_client.build_chat_request(
                context + "\nYou analyse the following Bible book and Answer the next question:\n" + prompt,
                system_prompt=None,
                model_name=self.model_name,
                temperature=0
            )
            start = time.time()
            response = self.model_client.send_request(req)
            latency = time.time() - start
            prompt_tokens = response.json().get("usage", {}).get('prompt_tokens') if response.status_code == 200 else None
            completion_tokens = response.json().get("usage", {}).get('completion_tokens') if response.status_code == 200 else None
            latencies.append(latency)
            completion_tokens_list.append(completion_tokens) if completion_tokens else None
            prompt_tokens_list.append(prompt_tokens) if prompt_tokens else None
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
        avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list)  if len(prompt_tokens_list) > 0 else None
        avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list)  if len(completion_tokens_list) > 0 else None
        result = {
            "Avg_latency": round(avg_latency, 3) if avg_latency else None,
            "Avg_prompt_tokens": round(avg_prompt_tokens, 1) if avg_prompt_tokens else None,
            "Avg_completion_tokens": round(avg_completion_tokens, 1) if avg_completion_tokens else None,
            "Throughput": round(throughput, 2),
            "Requests": len(latencies),
            "Status_codes": status_counts
        }
        print(f"Concurrency {concurrency}: Result: {result}")
        return result

    
    #Go through concurrency levels first, then context lengths
    def evaluate(self, max_prompts: int = 1) -> Dict[str, Dict[str, Dict]]:
        results_dict = {}
        self.warm_up_model()
        for ref_key, context in self.prompt_context.items():
            for concurrency in self.concurrency_levels:
                key = f"{concurrency}_par_req"
                print(f"Running {key} with {ref_key} context...")
                stats = self._run_concurrent_requests(context, concurrency)
                error_codes = [code for code, count in stats["Status_codes"].items() if code >= 400 and count > 0]
                if len(error_codes) > 0:
                    print(f"Error {error_codes} detected for {key} with {ref_key} context. Skipping...")
                    return results_dict
                else:
                    avg_tokens = stats["Avg_prompt_tokens"]
                    label = f"{avg_tokens // 1000}k_tokens"
                    if key not in results_dict:
                        results_dict[key] = {}
                    results_dict[key][label] = stats
        return results_dict
    
    def warm_up_model(self, warmup_count: int = 10, warmup_prompt: str = "Hello! Please respond quickly."):
        """
        Send a few dummy requests to warm up the model.
        """
        print(f"Warming up the model with {warmup_count} requests...")
        for i in range(warmup_count):
            try:
                req = self.model_client.build_chat_request(
                    warmup_prompt,
                    system_prompt=None,
                    model_name=self.model_name,
                    temperature=0
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
            extracted[concurrency_key] =[]
            for token_label, metrics in token_data.items():
                extracted[concurrency_key].append({k: metrics[k] for k in metrics if k in metrics_to_extract})
        return extracted