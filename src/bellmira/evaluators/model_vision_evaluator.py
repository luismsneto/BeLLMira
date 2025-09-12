import os
import base64
import requests
import uuid
from typing import List
from PIL import Image
from io import BytesIO
from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.llm_model.llm_model_client import ModelClient
from huggingface_hub import login
import json
import time
from typing import List, Optional, Dict, Tuple, Literal
  
def is_valid_image_prompt_list(image_prompt_list) -> bool:
    if not isinstance(image_prompt_list, list):
        return False

    for item in image_prompt_list:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        prompt, url = item
        if not isinstance(prompt, str) or not isinstance(url, str):
            return False
    return True

def download_images_from_huggingface(
    image_urls: list,
    local_path: str,
    hf_token: str = None
) -> list:
    """
    Downloads images from provided URLs, saves them to local_path with unique names,
    returns list of saved file paths.
    """
    if hf_token:
        login(token=hf_token)
    os.makedirs(local_path, exist_ok=True)

    saved_files = []

    for url in image_urls:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"image_{unique_id}.png"
            save_path = os.path.join(local_path, filename)
            image.save(save_path, format="PNG")
            saved_files.append(save_path)

            print(f"[{url}] ✅ Saved: {save_path}")

        except Exception as e:
            print(f"[{url}] ❌ Failed to process {url}: {e}")

    return saved_files

class ModelVisionEvaluator(ModelEvaluatorInterface):
    def __init__(self, 
                 url: str,
                 image_folder_path: str = "./images",
                 prompts: List[str] = ["Identify the elements in the image and describe them in a list."],
                 temperature: float = 0.0,
                 system_prompt: Optional[str] = None,
                 json_schema: Optional[dict] = None,
    ):
        """
        Logs in to Hugging Face, downloads 12 images, saves them locally.
        """
        self.model_url = url
        self.prompts = prompts
        self.context_path = image_folder_path
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name( )
        self.images = self._load_images()

    def _load_images(self) -> Dict[str, str]:
        """
        Loads all images from self.context_path and returns a dict:
        { "Size:{size}_Dim:{width}×{height}": image_path }
        """
        
        def format_bytes(size: int) -> str:
            if size < 1024:
                return f"{size} B"
            elif size < 1024 ** 2:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 ** 2):.2f} MB"

        image_info = []
        for filename in sorted(os.listdir(self.context_path)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(self.context_path, filename)
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                    size_bytes = os.path.getsize(path)
                    size_human = format_bytes(size_bytes)
                    image_info.append((size_bytes, size_human, width, height, path) )
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
        image_info.sort(key=lambda x: x[0])
        sorted_images = {
            f"Size:{size_human}_Dim:{width}×{height}": path
            for size, size_human, width, height, path in image_info
        }
        return sorted_images

    def evaluate(self, max_prompts: int = 1) -> Dict[str, List[Dict]]:
        results_dict = {}
        self.warm_up_model()
        for image_key, image_path in self.images.items():
            results_dict[image_key] = []
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Combine with each prompt (up to limit)
            for prompt in self.prompts[:max_prompts]:
                start_time = time.time()
                req = self.model_client.build_chat_request(
                    prompt,
                    system_prompt="Analyse the next image and tell what it is about.",
                    model_name=self.model_name,
                    temperature=0,
                    image_prompt=image_data
                )
                result = self.model_client.send_request(req)
                end_time = time.time()
                if not result.ok:
                    error_stats = {
                        "Code": result.status_code,
                        "Message": result.reason
                    }
                    results_dict[image_key].append(error_stats)
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
                results_dict[image_key].append(request_stats)
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

    def compute_averages(self, results_dict: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        avg_results = {}

        for key, results in results_dict.items():
            total_exec = 0.0
            total_tokens = 0
            prompt_tokens = 0
            comp_tokens = 0
            exec_count = 0
            token_count = 0
            errors = []
            for result in results:
                code = result.get("Code")
                if code != 200:
                    errors.append(f"{code} {result.get('Message')}")
                exec_time = result.get("Execution_time")
                if exec_time is not None:
                    total_exec += exec_time
                    exec_count += 1
                total_t = result.get("Total_tokens")
                if total_t is not None:
                    total_tokens += total_t
                    prompt_tokens += result.get("Prompt_tokens", 0)
                    comp_tokens += result.get("Completion_tokens", 0)
                    token_count += 1
            avg_result = {}
            if exec_count:
                avg_result["Avg_execution_time"] = round(total_exec / exec_count, 2)
            else:
                avg_result["Avg_execution_time"] = None
            if token_count:
                avg_result["Avg_total_tokens"] = round(total_tokens / token_count, 2)
                avg_result["Avg_prompt_tokens"] = round(prompt_tokens / token_count, 2)
                avg_result["Avg_completion_tokens"] = round(comp_tokens / token_count, 2)
            if errors:
                avg_result["Errors"] = errors
            avg_results[key] = avg_result
        return avg_results

    def extract_threshold_metrics(self,
                                avg_results: Dict[str, Dict], 
                                metrics: List[str] = ["Avg_execution_time"],
                                key_mode: Literal["full", "size", "dim"] = "dim",
                                suffix_map: Dict[str, str] = {"Avg_execution_time": "Avg_Exec_T(s)"}) -> dict:
        """
        Flattens the averaged metrics into a dict.
        - Only includes specified metrics.
        - Allows key formatting by 'full' (Size+Dim), 'size' only, or 'dim' only.
        - Allows suffix abbreviations via suffix_map.
        - Errors are grouped under 'Error' key.
        """
        averages = self.compute_averages(avg_results)
        result = {}
        errors = []
        for key, key_metrics in averages.items():
            # Extract parts
            size_part = ""
            dim_part = ""

            if "Size:" in key and "_Dim:" in key:
                size_part, dim_part = key.split("_")
            elif "Size:" in key:
                size_part = key
            elif "Dim:" in key:
                dim_part = key

            if key_mode == "full":
                base_key = key
            elif key_mode == "size":
                base_key = size_part
            elif key_mode == "dim":
                base_key = dim_part
            else:
                base_key = key

            for metric in metrics:
                short_metric = suffix_map.get(metric, metric) if suffix_map else metric
                value = key_metrics.get(metric)
                if value is not None:
                    result[f"{base_key}_{short_metric}"] = value

            if "Errors" in key_metrics:
                for err in key_metrics["Errors"]:
                    errors.append(f"{base_key}_Error:{err}")
        if errors:
            result["Error"] = ";".join(errors)

        return result
        #def extract_threshold_metrics(self, data: dict) -> dict:
            
        #    return row