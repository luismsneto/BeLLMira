import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import json
from collections import Counter
import pandas as pd

from bellmira.llm_model.llm_model_client import ModelClient

class ModelClassificationEvaluator():
    def __init__(
        self,
        input_col: str,
        output_col: str,
        url: str,
        data_path: str,
        data_format: str = "parquet",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None
    ):
        prefix, path = data_path.split(":", 1)
        path = path.lstrip("/").strip()
        match prefix:
            case "s3":
                # Handle S3 path
                return "Handling S3 path. Not implemented"
            case "dbfs":
                # Handle DBFS path
                print("Handling DBFS path")
            case "file":
                # Handle local file path
                print("Handling local file path")
            case "adls":
                # Handle Azure Data Lake Storage path
                print("Handling ADLS path")
                try:
                    client_id = dbutils.secrets.get(scope="DataBricksKVScopeAIP", key="DTBK002-SPNClientID-AI")
                    client_secret = dbutils.secrets.get(scope="DataBricksKVScopeAIP", key="DTBK002-SPNClientSecret-AI")
                    tenant_id_endpoint = dbutils.secrets.get(scope="DataBricksKVScopeAIP", key="DataBricksAccessToken")
                    datalake_url = dbutils.secrets.get(scope="DataBricksKVScopeAIP", key="DataLakeSAUri")

                    spark.conf.set("fs.azure.account.auth.type", "OAuth")
                    spark.conf.set("fs.azure.account.oauth.provider.type", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
                    spark.conf.set("fs.azure.account.oauth2.client.id", client_id)
                    spark.conf.set("fs.azure.account.oauth2.client.secret", client_secret)
                    spark.conf.set("fs.azure.account.oauth2.client.endpoint", tenant_id_endpoint)

                    print("ADLS access configured successfully.")
                except Exception as e:
                    print(f"Error configuring ADLS access: {e}")
                    print("Please ensure Databricks secrets scope 'DataBricksKVScopeAIP' and the required keys exist.")
                data_path = datalake_url + path
            case _:
                # Handle unknown prefix
                return "Unknown prefix"
        try:
            data_df = spark.read.format(data_format).load(data_path)
            data_df.createOrReplaceTempView("categories_raw")
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            raise e
        
        #self.bible_path = bible_path
        #self.prompts = prompts
        #self.chapter_numbers = chapter_numbers
        self.model_url = url
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.json_schema = json_schema
        self.model_client = ModelClient(base_url=self.model_url)
        self.model_name = self.model_client.get_model_name( )

        self.input_col = input_col
        self.output_col = output_col
        self.data = data_df

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

    def evaluate(self, max_prompts: int = 2) -> Dict[str, List[Dict]]:
        print("ClassificationEvaluator warming up model...")
        self.warm_up_model(warmup_count=3)
        print("ClassificationEvaluator warm up model finished.")
        results_dict = {}

        batch=self.data.limit(max_prompts).toPandas()
        texts = []

        if self.input_col not in batch or self.output_col not in batch:
            raise ValueError(f"Batch is missing required columns: {self.input_col} or {self.output_col}")

        results_list = []
        for user_input, assistant_output in zip(batch[self.input_col], batch[self.output_col]):
            start_time = time.time()
            req = self.model_client.build_chat_request(
                user_prompt=user_input,
                system_prompt=self.system_prompt,
                model_name=self.model_name,
                enable_thinking=False
            )
            result = self.model_client.send_request(req)
            end_time = time.time()
            if not result.ok:
                error_stats = {
                    "Code": result.status_code,
                    "Message": result.reason
                }
                results_list.append(error_stats)
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
                "Execution_time": end_time - start_time,
                "Total_tokens": usage.get('total_tokens'),
                "Prompt_tokens": usage.get('prompt_tokens'),
                "Completion_tokens": usage.get('completion_tokens'),
                "Prediction": result_json.get("choices", {})[0].get("message", {}).get("content"),
                "Label":assistant_output
            }
            print(request_stats)
            results_list.append(request_stats)
        return results_list
        
    def extract_threshold_metrics(self, results: List[Dict]) -> Dict[str, Dict]:

        n = len(results)
        if n == 0: 
            return {}
        errors = []
        avg_execution_time = sum(r["Execution_time"] for r in results) / n
        avg_prompt_tokens = sum(r["Prompt_tokens"] for r in results) / n
        avg_completion_tokens = sum(r["Completion_tokens"] for r in results) / n
        [errors.append(r["Message"]) for r in results if r["Code"] != 200]
        labels = set(r["Label"] for r in results)
        preds = set(r["Prediction"] for r in results)
        classes = sorted(labels | preds)

        # Initialize counts
        TP = Counter()
        FP = Counter()
        FN = Counter()
        TN = Counter()
        
        # Count TP, FP, FN per class
        for cls in classes:
            for r in results:
                pred = r["Prediction"]
                label = r["Label"]

                if pred == cls and label == cls:
                    TP[cls] += 1
                elif pred == cls and label != cls:
                    FP[cls] += 1
                elif pred != cls and label == cls:
                    FN[cls] += 1
                elif pred != cls and label != cls:
                    TN[cls] += 1

        # Compute per-class metrics
        metrics = {}
        for cls in classes:
            tp, fp, fn, tn = TP[cls], FP[cls], FN[cls], TN[cls]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics[cls] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }

        # Macro average
        macro_precision = round(sum(m["Precision"] for m in metrics.values()) / len(classes), 2)
        macro_recall = round(sum(m["Recall"] for m in metrics.values()) / len(classes), 2)
        macro_f1 = round(sum(m["F1"] for m in metrics.values()) / len(classes), 2)

        # Micro average
        total_TP = sum(TP.values())
        total_FP = sum(FP.values())
        total_FN = sum(FN.values())

        micro_precision = round(total_TP / (total_TP + total_FP), 2) if (total_TP + total_FP) > 0 else 0.0
        micro_recall = round(total_TP / (total_TP + total_FN), 2) if (total_TP + total_FN) > 0 else 0.0
        micro_f1 = round((2 * micro_precision * micro_recall) / (micro_precision + micro_recall), 2) if (micro_precision + micro_recall) > 0 else 0.0

        # Accuracy
        correct = sum(1 for r in results if r["Prediction"] == r["Label"])
        accuracy = round(correct / len(results), 2) if results else 0.0

        # Confusion matrix
        conf_mat = pd.DataFrame(0, index=classes, columns=classes)
        for r in results:
            conf_mat.loc[r["Label"], r["Prediction"]] += 1

        return {
            "Avg_Execution_Time": round(avg_execution_time, 1),
            "Avg_Prompt_Tokens": round(avg_prompt_tokens, 1),
            "Avg_Completion_Tokens": round(avg_completion_tokens, 1),
            #"Per_Class": metrics,
            "Macro-Precision": macro_precision,
            "Macro-Recall": macro_recall,
            "Macro-F1": macro_f1,
            "Micro-Precision": micro_precision,
            "Micro-Recall": micro_recall,
            "Micro-F1": micro_f1,
            "Accuracy": accuracy,
            "Error": "\n".join(errors)
        }