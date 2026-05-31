import logging
import threading
import time
import requests
from bellmira.evaluators.evaluator_interface import ModelEvaluatorInterface
from bellmira.utils.dict_logger import ExperimentLogger
from bellmira.llm_model.llm_model import LLMModel

logger = logging.getLogger(__name__)

_LOCAL_EXPERIMENT_COOLDOWN_S = 30


# -------------- Thread 1: Start the model -----------------
def start_model(repo_id, server_options, stop_event, server_finished_event):
    try:
        llm_model = LLMModel(
            repo_id=repo_id,
            hf_filename=server_options.get("hf_filename", None),
            hf_file_repo=server_options.get("hf_file_repo", None),
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
        logger.error("Experiment for %s: Exception in start_model: %s", repo_id, e, exc_info=True)


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
    exp_logger = ExperimentLogger(
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
        health_url = f"{url}health"
        while time.time() - start_time < timeout:
            if server_finished_event and server_finished_event.is_set():
                logger.warning("Experiment for %s: Aborting evaluation — start_model failed.", model)
                return "start_model_failed"
            try:
                logger.info("Experiment for %s: Pinging %s.", model, health_url)
                if requests.get(health_url, timeout=10).status_code == 200:
                    logger.info("Experiment for %s: Model is healthy. Proceeding with evaluation.", model)
                    return "healthy"
            except requests.RequestException:
                logger.info("Experiment for %s: Model not ready yet...", model)
            time.sleep(interval)
        logger.warning("Experiment for %s: Model failed to become healthy within timeout.", model)
        return "timeout"

    try:
        state = wait_until_server_healthy()
        if state != "healthy":
            log["Status"] = state
            exp_logger.log(log)
            return
        if evaluator_class is None:
            raise ValueError(f"Experiment for {model}: no evaluator class provided.")
        if not issubclass(evaluator_class, ModelEvaluatorInterface):
            raise TypeError(
                f"Experiment for {model}: evaluator_class must be a subclass of "
                f"ModelEvaluatorInterface, got {evaluator_class}."
            )
        logger.info("Experiment for %s: Using %s.", model, evaluator_class.__name__)
        # Copy args to avoid mutating the caller's dict across parallel threads
        args = dict(evaluator_args)
        args["url"] = url
        # max_prompts is an evaluate() argument, not a constructor argument
        max_prompts = args.pop("max_prompts", None)
        evaluator = evaluator_class(**args)
        results = (
            evaluator.evaluate(max_prompts=max_prompts)
            if max_prompts is not None
            else evaluator.evaluate()
        )
        data = evaluator.extract_threshold_metrics(results)
        logger.info("Experiment for %s: Data: %s", model, data)
        log["Status"] = "success"
        log.update(data)
        exp_logger.log(log)
        logger.info("Experiment for %s: Evaluation complete.", model)
    except Exception as e:
        logger.error(
            "Experiment for %s: Exception during evaluation: %s. Evaluator Args: %s.",
            model, e, evaluator_args, exc_info=True,
        )
        log["Status"] = "error-exception"
        exp_logger.log(log)
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
            "columns": experiment.get("columns"),
            "evaluator_class": experiment.get("evaluator", None),
            "evaluator_args": experiment.get("evaluator_args", {}),
            "output_filename": experiment.get("output_filename", "./results/experiment.csv")
        }

    def create_experiments(self):
        for experiment in self.experiments:
            model_name = experiment["model"]
            logger.info("Running experiment for model: %s", model_name)
            logger.info("%s", experiment)
            eval_kwargs = self._build_eval_kwargs(experiment)
            if "url" in experiment:
                # Remote model evaluation
                eval_thread = threading.Thread(target=wait_for_model_and_evaluate, kwargs=eval_kwargs)
                self.remote_threads.append(eval_thread)
            else:
                stop_event = threading.Event()
                server_finished_event = threading.Event()
                eval_kwargs["stop_event"] = stop_event
                eval_kwargs["server_finished_event"] = server_finished_event
                # Local model: launch server + evaluate in parallel threads
                server_thread = threading.Thread(target=start_model, args=(
                    experiment["repo_id"],
                    experiment["server_options"],
                    stop_event,
                    server_finished_event,
                ))
                eval_thread = threading.Thread(target=wait_for_model_and_evaluate, kwargs=eval_kwargs)
                self.local_threads.append((server_thread, eval_thread, stop_event))

    def launch_experiments(self):
        logger.info("Launching and waiting for local evaluations to finish...")
        for eval_thread in self.remote_threads:
            eval_thread.start()
        for server_thread, eval_thread, stop_event in self.local_threads:
            server_thread.start()
            eval_thread.start()
            eval_thread.join()
            stop_event.set()
            server_thread.join()
            time.sleep(_LOCAL_EXPERIMENT_COOLDOWN_S)

        logger.info("Waiting for remote evaluations to finish...")
        for eval_thread in self.remote_threads:
            eval_thread.join()
