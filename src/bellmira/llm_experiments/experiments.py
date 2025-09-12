import threading
import time
import traceback
import pandas as pd
import requests
from bellmira.evaluators import ModelClassificationEvaluator, ModelParallelLoadEvaluator, ModelContextLengthEvaluator, ModelVisionEvaluator
from bellmira.utils.dict_logger import ExperimentLogger
from bellmira.llm_model.llm_model import LLMModel

# -------------- Thread 1: Start the model -----------------
def start_model(repo_id, server_options, stop_event, server_finished_event):
  try:
    llm_model = LLMModel(
        repo_id=repo_id,
        hf_filename = server_options.get("hf_filename", None),
        hf_file_repo = server_options.get("hf_file_repo", None),
    )
    llm_model.serve_model(
      max_model_len=server_options["max_model_len"], 
      reasoning_parser=server_options.get("reasoning_parser", None), 
      dtype=server_options.get("dtype", None),
      quantization=server_options.get("quantization", None),
      gpu_memory_utilization=server_options.get("gpu_memory_utilization", None),
      cpu_offload=server_options.get("cpu_offload", None),
      enable_prefix_caching=server_options.get("enable_prefix_caching", False),
      stop_event=stop_event,
      server_finished_event=server_finished_event,
      disable_logs=True
    )
  except Exception as e:
      print(f"Experiment for {repo_id}:Exception in start_model: {e}")
      traceback.print_exc()

# -------------- Thread 2: Poll health & evaluate ------------
def wait_for_model_and_evaluate(
    model,
    max_model_len,
    quantization,
    precision,
    dtype,
    size,
    prefix_caching,
    env,
    vllm_version,
    evaluator_class,
    evaluator_args,
    columns,
    output_filename,
    server_finished_event=None,
    stop_event=None,
    url=None,
    timeout=600,
    interval=30,
):
    logger = ExperimentLogger(
        file_path=output_filename,
        columns=columns,
    )
    log = {
        "Model": model,
        "Max Model Len": max_model_len,
        "Quantization": quantization,
        "Precision": precision,
        "Dtype": dtype,
        "Size": size,
        "Prefix Caching": prefix_caching,
        "Env": env,
        "VLLM Version": vllm_version
    }
    def wait_until_server_healthy():
        start_time = time.time()
        HEALTH_URL = f"{url}health"
        while time.time() - start_time < timeout:
            if server_finished_event and server_finished_event.is_set():
                print(f"Experiment for {model}: Aborting evaluation — start_model failed.")
                return "start_model_failed"
            try:
                print(f"Experiment for {model}: Pinging {HEALTH_URL}.")
                if requests.get(HEALTH_URL, timeout=10).status_code == 200:
                    print(f"Experiment for {model}: Model is healthy. Proceeding with evaluation.")
                    return "healthy"
            except requests.RequestException:
                print(f"Experiment for {model}: Model not ready yet...")
            time.sleep(interval)
        print(f"Experiment for {model}: Model failed to become healthy within timeout.")
        return "timeout"
    try:
        state = wait_until_server_healthy()
        if state != "healthy":
          log["Status"] = state
          logger.log(log)
          return
        if evaluator_class==ModelContextLengthEvaluator:  
            print(f"Experiment for {model}: Using Model Context Length Evaluator.")
            evaluator_args["url"]= f"{url}"
            evaluator = evaluator_class(**evaluator_args)
            results = evaluator.evaluate()
            data = evaluator.extract_threshold_metrics(results)
            print(f"Experiment for {model}: Data: {data}")
            log["Status"] = "success"
        elif evaluator_class==ModelParallelLoadEvaluator:  
            print(f"Experiment for {model}: Using Model Parallel Load Evaluator.")
            evaluator_args["url"]= f"{url}"
            evaluator = evaluator_class(**evaluator_args)
            results = evaluator.evaluate()
            data = evaluator.extract_threshold_metrics(results)
            print(f"Experiment for {model}: Data: {data}")
            log["Status"] = "success"
        elif evaluator_class==ModelVisionEvaluator:  
            print(f"Experiment for {model}: Using Model Vision Evaluator.")
            evaluator_args["url"]= f"{url}"
            evaluator = evaluator_class(**evaluator_args)
            results = evaluator.evaluate()
            data = evaluator.extract_threshold_metrics(results)
            print(f"Experiment for {model}: Data: {data}")
            log["Status"] = "success"
        elif evaluator_class==ModelClassificationEvaluator:  
            print(f"Experiment for {model}: Using Model Classification Evaluator.")
            evaluator_args["url"]= f"{url}"
            max_prompts = evaluator_args.pop("max_prompts", 500)
            evaluator = evaluator_class(**evaluator_args)
            print(f"Max Prompts: ", max_prompts)
            results = evaluator.evaluate(max_prompts=max_prompts)
            data = evaluator.extract_threshold_metrics(results)
            print(f"Experiment for {model}: Data: {data}")
            log["Status"] = "success"
            
        else:
            raise Exception("Invalid evaluator")
        log.update(data)
        logger.log(log)
        print(f"Experiment for {model}: Evaluation complete.")
    except Exception as e:
        print(f"Experiment for {model}: Exception during evaluation: {e}")
        print(f"Experiment for {model}: Evaluator Args: {evaluator_args}.")
        traceback.print_exc()
        log["Status"] = "error-exception"
        logger.log(log)
        if stop_event:
            stop_event.set()
            
class Experiments:
  def __init__(self, experiments, vllm_version="0.9.2"):
    self.experiments = experiments
    self.vllm_version = vllm_version
    self.remote_threads = []
    self.local_threads = []

  def _build_eval_kwargs(self, experiment):
    return {
      "model": experiment.get("model"),
      "max_model_len": experiment.get("server_options", {}).get("max_model_len"),
      "url": experiment.get("url", "http://localhost:8080/"),
      "quantization": experiment.get("quantization", ""),
      "dtype": experiment.get("dtype", ""),
      "precision": experiment.get("precision", ""),
      "size": experiment.get("size", ""),
      "prefix_caching": experiment.get("prefix_caching", False),
      "env": experiment.get("env", "local"),
      "vllm_version": experiment.get("vllm_version", self.vllm_version),
      "server_finished_event": None,
      "stop_event": None,
      "columns": None,
      "evaluator_class": experiment.get("evaluator", None),
      "evaluator_args": experiment.get("evaluator_args", {}),
      "output_filename": experiment.get("output_filename", "./results/experiment.csv")
    }

  def create_experiments(self):
    for experiment in self.experiments:
      model_name = experiment["model"]
      print(f"Running experiment for model: {model_name}")
      print(experiment)
      eval_kwargs = self._build_eval_kwargs(experiment)
      if "url" in experiment:
        # Remote model evaluation
        t = threading.Thread(target=wait_for_model_and_evaluate, kwargs=eval_kwargs)
        self.remote_threads.append(t)
      else:
        stop_event = threading.Event()
        server_finished_event = threading.Event()
        eval_kwargs["stop_event"] = stop_event
        eval_kwargs["server_finished_event"] = server_finished_event
        # Local model: launch + evaluate
        t1 = threading.Thread(target=start_model, args=(
            experiment["repo_id"],
            experiment["server_options"],
            stop_event,
            server_finished_event,
        ))
        t2 = threading.Thread(target=wait_for_model_and_evaluate, kwargs=eval_kwargs)
        self.local_threads.append((t1, t2, stop_event))

  def launch_experiments(self):
    print("Launching and waiting for local evaluations to finish...")
    for t1 in self.remote_threads:
      t1.start()
    for t1, t2, stop_event in self.local_threads:
      t1.start()
      t2.start()
      t2.join()
      stop_event.set()
      t1.join()
      time.sleep(30)

    print("Waiting for remote evaluations to finish...")
    for t in self.remote_threads:
      t.join()