# bible_model_test.py

import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import json

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
    
class ModelContextLengthEvaluator(ModelEvaluatorInterface):
    def __init__(
        self,
        bible_path: str,
        url: str,
        prompts: List[str],
        chapter_numbers: List[int],
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
        type: str = "chat"
    ):
        print(bible_path, url, prompts, chapter_numbers, temperature, system_prompt, json_schema)
        print("ModelContextLengthEvaluator initialized.")
        self.bible_path = bible_path
        self.model_url = url
        self.prompts = prompts
        self.chapter_numbers = chapter_numbers
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name( )
        self.prompt_context = self._build_prompt_context(file_path=bible_path, chapter_numbers =chapter_numbers)
        self.type = type
    
    def _build_prompt_context(self, file_path: str, chapter_numbers = List[int]) -> str:
        bible_text = read_text_file(file_path)
        return extract_bible_chapters(bible_text, chapter_numbers)
    
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
                        context + "\nYou analyse the following Bible book and Answer the next question:\n" + prompt,
                        system_prompt=None,
                        model_name=self.model_name,
                        temperature=0
                    )
                elif self.type == "embedding":
                    req = self.model_client.build_embedding_request(
                        input_text=context + "\nYou analyse the following Bible book and Answer the next question:\n" + prompt,
                        model_name=self.model_name
                    )
                result = self.model_client.send_request(req)
                end_time = time.time()
                if not result.ok:
                    error_stats = {
                        "Code": result.status_code,
                        "Message": result.reason
                    }
                    results_dict[ref_key].append(error_stats)
                    return results_dict
                try:
                    result_json = result.json()
                    message = result_json.get("message") if result.status_code != 200 else None
                    usage = result_json.get("usage", {})
                except ValueError:
                    message = f"Non-JSON response: {result.text[:200]}"
                    usage = {}
                request_stats = {
                    "Code": result.status_code,
                    "Message": message,
                    "Execution_time": end_time - start_time,
                    "Total_tokens": usage.get('total_tokens'),
                    "Prompt_tokens": usage.get('prompt_tokens'),
                    "Completion_tokens": usage.get('completion_tokens')
                }
                results_dict[ref_key].append(request_stats)
        return results_dict
    
    def warm_up_model(self, warmup_count: int = 10, warmup_prompt: str = "Hello! Please respond quickly."):
        """
        Send a few dummy requests to warm up the model.
        """
        print(f"Warming up the model with {warmup_count} requests...")
        for i in range(warmup_count):
            try:
                req = None
                if self.type == "chat":
                    req = self.model_client.build_chat_request(
                        warmup_prompt,
                        system_prompt=None,
                        model_name=self.model_name,
                        temperature=0
                    )
                elif self.type == "embedding":
                    req = self.model_client.build_embedding_request(
                        input_text=warmup_prompt,
                        model_name=self.model_name
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
                    "Avg_completion_tokens": round(comp_tokens / token_entries)
                })
            if error_msg:
                avg_result["Errors"] = error_msg

            avg_results[key] = avg_result

        return avg_results
        
    def extract_threshold_metrics(self, data: dict, token_thresholds=[1000, 2000, 4000, 7500, 12000, 16000, 20000, 24000, 32000]) -> dict:
        averages = self.compute_averages(data)
        threshold_columns = [f"{t // 1000}k Tok Avg Time" for t in token_thresholds]
        row = {}

        entries = list(averages.values())

        for threshold, col_name in zip(token_thresholds, threshold_columns):
            min_exec_time = None
            min_completion_tokens = None
            min_prompt_tokens = float('inf')
            
            for entry in entries:
                avg_prompt_tokens = entry.get("Avg_prompt_tokens") if "Avg_prompt_tokens" in entry else None
                exec_time = entry.get("Avg_execution_time") if "Avg_execution_time" in entry else None

                if avg_prompt_tokens is None or exec_time is None:
                    continue

                if avg_prompt_tokens >= threshold and avg_prompt_tokens < min_prompt_tokens:
                    min_prompt_tokens = avg_prompt_tokens
                    min_exec_time = exec_time
                    min_completion_tokens = entry.get("Avg_completion_tokens") if "Avg_completion_tokens" in entry else None

            if min_exec_time is not None:
                row[col_name] = {
                    "Avg_Exec_Time": min_exec_time,
                    "Avg_prompt_tokens": min_prompt_tokens,
                    "Avg_completion_tokens": min_completion_tokens,
                }
        row["Error"] = averages.get("Errors", None)
        return row