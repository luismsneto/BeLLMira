import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict
import subprocess
import mlflow
import threading
import shutil
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class LLMModel():
    """
    A utility class for downloading, managing, registering, and serving Hugging Face LLM models using MLflow and vLLM.

    Attributes:
        repo_id (str): Hugging Face model repository ID.
        hf_file_repo (str, optional): Hugging Face repository from which to download specific files.
        hf_filename (str, optional): Name of a specific file to download from the Hugging Face repo.
        tmp_model_path (str, optional): Local temporary directory for storing the model.
    """

    def __init__(self, repo_id: str, hf_file_repo: str = None, hf_filename: str = None, local_model_path: str = None):
        """
        Initializes the LLMModel instance.

        Args:
            repo_id (str): Hugging Face repository ID for the model.
            hf_file_repo (str, optional): Repository ID to download auxiliary files.
            hf_filename (str, optional): Specific file name to download from the HF repo.
            local_model_path (str, optional): Path to a local directory where the model should be stored.
        """
        self.repo_id = repo_id
        self.hf_file_repo = hf_file_repo
        self.hf_filename = hf_filename
        self.tmp_model_path = None
        if local_model_path is not None:
            self.create_tmp_model_path(local_model_path)

    def create_tmp_model_path(self, tmp_model_path: str = "/tmp/model") -> str:
        """
        Creates the temporary directory for model storage.

        Args:
            tmp_model_path (str): Path to the temporary directory.

        Returns:
            str: Path to the created temporary directory.
        """
        os.makedirs(tmp_model_path, exist_ok=True)
        self.tmp_model_path = tmp_model_path
        return self.tmp_model_path

    def download_model(self, ignore_patterns=None) -> None:
        """
        Downloads the full model from Hugging Face, optionally ignoring certain file patterns.

        Args:
            ignore_patterns (list[str], optional): List of glob patterns to ignore during download.
        """
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if self.tmp_model_path is None:
            self.create_tmp_model_path()
        snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.tmp_model_path,
            force_download=True,
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns
        )
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        self._clear_huggingface_cache()

    def download_hf_files(self) -> str:
        """
        Downloads a specific file from a Hugging Face repository and the rest of the model (excluding .safetensors).

        Returns:
            str: Local path to the downloaded file.
        """
        if not self.hf_filename:
            raise ValueError("hf_filename is required to call download_hf_files.")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if self.tmp_model_path is None:
            self.create_tmp_model_path()
        file_path = hf_hub_download(
            repo_id=self.hf_file_repo,
            filename=self.hf_filename,
            local_dir=self.tmp_model_path
        )
        self.download_model(ignore_patterns=["*.safetensors"])
        logger.info("Downloaded file to: %s", file_path)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        return file_path

    def _build_mlmodel_content(self, run_id: str, databricks_runtime: str, python_version: str) -> str:
        """Build the MLmodel YAML content string shared by save_model_content and add_mlmodel_to_run."""
        utc_time_created = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        return (
            f"artifact_path: model\n"
            f"databricks_runtime: {databricks_runtime}\n"
            f"flavors:\n"
            f"python_function:\n"
            f"  data: model\n"
            f"  env:\n"
            f"    conda: conda.yaml\n"
            f"    virtualenv: python_env.yaml\n"
            f"  loader_module: transformers\n"
            f"  python_version: {python_version}\n"
            f"mlflow_version: {mlflow.version.VERSION}\n"
            f"model_uuid: {run_id}\n"
            f"run_id: {run_id}\n"
            f"utc_time_created: '{utc_time_created}'\n"
        )

    def add_mlmodel_to_run(
        self,
        run_id: str,
        databricks_runtime: str,
        python_version: str = None
    ) -> None:
        """
        Adds or overwrites an `MLmodel` file in the given open run.

        Args:
            run_id (str): The MLflow run ID where to store the MLmodel.
            databricks_runtime (str): The Databricks runtime version for metadata.
            python_version (str, optional): Python version to record. Defaults to sys.version.
        """
        python_version = python_version or sys.version
        model_content = self._build_mlmodel_content(run_id, databricks_runtime, python_version)

        tmp_mlmodel_path = Path(self.tmp_model_path) / "MLmodel"
        tmp_mlmodel_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_mlmodel_path, "w") as f:
            f.write(model_content)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(local_path=str(tmp_mlmodel_path), artifact_path="model")

    def _clear_huggingface_cache(self):
        """Deletes the Hugging Face cache directory inside the model path."""
        hf_cache_dir = os.path.expanduser(f"{self.tmp_model_path}/.cache/huggingface")
        if os.path.exists(hf_cache_dir):
            logger.info("Clearing Hugging Face cache at: %s", hf_cache_dir)
            shutil.rmtree(hf_cache_dir)
        else:
            logger.info("Hugging Face cache directory not found. Skipping cleanup.")

    def save_model_content(self, run_info: mlflow.entities.RunInfo, databricks_runtime: str) -> None:
        """
        Generates and saves an MLflow `MLmodel` metadata file for the model.

        Args:
            run_info (mlflow.entities.RunInfo): The current MLflow run info.
            databricks_runtime (str): Runtime version used for Databricks compatibility.
        """
        model_content = self._build_mlmodel_content(run_info.run_id, databricks_runtime, sys.version)
        with open(Path(self.tmp_model_path, "MLmodel"), "w") as model_content_file:
            model_content_file.write(model_content)

    def register_model(self, mlflow_experiment: str, databricks_runtime: str, model_name: str) -> Dict[str, Any]:
        """
        Registers the model to MLflow after downloading and preparing artifacts.

        Args:
            mlflow_experiment (str): Name of the MLflow experiment.
            databricks_runtime (str): Runtime version used for compatibility metadata.
            model_name (str): Name of the model to be registered.

        Returns:
            Dict[str, Any]: Dictionary representation of the MLflow run metadata.
        """
        if self.tmp_model_path is None:
            self.create_tmp_model_path()
        if self.hf_file_repo and self.hf_filename:
            self.download_hf_files()
        else:
            self.download_model()
        mlflow.set_experiment(experiment_name=mlflow_experiment)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("model_name", model_name)
            self.save_model_content(run_info=run.info, databricks_runtime=databricks_runtime)
            mlflow.log_artifact(local_path=self.tmp_model_path, artifact_path="")
        mlflow.end_run()
        return mlflow.get_run(run_id).to_dictionary()

    def serve_model(
        self,
        max_model_len: int = 16384,
        reasoning_parser: str = None,
        tensor_parallel_size: str = None,
        pipeline_parallel_size: str = None,
        stop_event: threading.Event = None,
        server_finished_event: threading.Event = None,
        dtype: str = "half",
        quantization: str = None,
        gpu_memory_utilization: float = None,
        cpu_offload: float = None,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        max_num_seqs: int = None,
        disable_logs: bool = False,
    ) -> None:
        """
        Launches a vLLM-based OpenAI-compatible API server to serve the downloaded model.

        Args:
            max_model_len (int): Maximum model context length.
            reasoning_parser (str, optional): Reasoning parser module to use.
            tensor_parallel_size (str, optional): Tensor parallelism setting for model execution.
            pipeline_parallel_size (str, optional): Pipeline parallelism setting for model execution.
            stop_event (threading.Event, optional): Event to stop the server gracefully.
            server_finished_event (threading.Event, optional): Event set when the server terminates.
            dtype (str): Data type for model weights (e.g. ``"half"``, ``"bfloat16"``).
            quantization (str, optional): Quantization method (e.g. ``"awq"``, ``"gptq"``).
            gpu_memory_utilization (float, optional): Fraction of GPU memory to reserve for the model.
            cpu_offload (float, optional): GB of model weights to offload to CPU.
            enable_prefix_caching (bool): Whether to enable automatic prefix caching.
            enforce_eager (bool): Disable torch.compile/CUDA graphs and use eager execution.
                Useful when nvcc is not installed. Slight throughput penalty.
            disable_logs (bool): Whether to disable request and stats logging.
        """
        base_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--trust-remote-code",
            "--max-model-len", str(max_model_len)
        ]
        if pipeline_parallel_size:
            base_cmd.extend(["--pipeline-parallel-size", pipeline_parallel_size])
        if tensor_parallel_size:
            base_cmd.extend(["--tensor-parallel-size", tensor_parallel_size])
        if reasoning_parser:
            base_cmd.extend(["--reasoning-parser", reasoning_parser])
        if self.hf_filename:
            model_path = str(Path(self.tmp_model_path, self.hf_filename).resolve())
            base_cmd.extend(["--model", model_path, "--tokenizer", self.repo_id])
        else:
            base_cmd.extend(["--model", self.repo_id])
        if disable_logs:
            base_cmd.extend(["--disable-log-requests", "--disable-log-stats"])
        if dtype:
            base_cmd.extend([f"--dtype={dtype}"])
        if quantization:
            base_cmd.extend([f"--quantization={quantization}"])
        if gpu_memory_utilization:
            base_cmd.extend([f"--gpu-memory-utilization={gpu_memory_utilization}"])
        if cpu_offload:
            base_cmd.extend([f"--cpu-offload-gb={cpu_offload}"])
        if enable_prefix_caching:
            base_cmd.extend(["--enable-prefix-caching=True"])
        if enforce_eager:
            base_cmd.extend(["--enforce-eager"])
        if max_num_seqs:
            base_cmd.extend([f"--max-num-seqs={max_num_seqs}"])

        if sys.platform == "win32":
            base_cmd = self._wrap_wsl(base_cmd)

        logger.info("Model Server: Running command: %s", " ".join(base_cmd))
        print(f"Model Server: Running command: {' '.join(base_cmd)}", flush=True)
        if stop_event is None and server_finished_event is None:
            logger.info("Model Server: Running on current thread (blocking execution)")
            process = subprocess.Popen(
                base_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            output_lines = []
            for line in process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Model server exited with code {process.returncode}.\n"
                    f"Command: {' '.join(base_cmd)}\n\n"
                    f"--- Full output ---\n{''.join(output_lines)}"
                )
        else:
            logger.info("Model Server: Running on parallel thread")
            process = subprocess.Popen(base_cmd, stderr=subprocess.PIPE)

            if server_finished_event:
                def monitor_process():
                    _, stderr_output = process.communicate()
                    if process.returncode != 0 and stderr_output:
                        logger.error(
                            "Model Server: exited with code %d.\n%s",
                            process.returncode,
                            stderr_output.decode(errors="replace"),
                        )
                    logger.info("Model Server: process exited.")
                    logger.info("Model Server: Set server_finished_event.")
                    server_finished_event.set()

                threading.Thread(target=monitor_process, daemon=True).start()

            if stop_event:
                stop_event.wait()
                logger.info("Model Server: Stop event set, terminating server...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Model Server: did not terminate in time, killing it.")
                    process.kill()
                    process.wait()

        logger.info("Model Server: stopped.")

    @staticmethod
    def _to_wsl_path(windows_path: str) -> str:
        p = Path(windows_path).resolve()
        drive = p.drive.lower().rstrip(":")
        rest = p.as_posix()[3:]  # strip "C:/"
        return f"/mnt/{drive}/{rest}"

    def _wrap_wsl(self, cmd: list) -> list:
        # Remap any Windows path arguments to their /mnt/... WSL equivalents
        wsl_args = []
        for arg in cmd:
            try:
                p = Path(arg)
                if p.is_absolute() or (p.drive and ":" in p.drive):
                    wsl_args.append(self._to_wsl_path(arg))
                    continue
            except (ValueError, OSError):
                pass
            wsl_args.append(arg)

        project_root = self._to_wsl_path(str(Path(__file__).resolve().parents[3]))
        inner = " ".join(f'"{a}"' if " " in a else a for a in wsl_args)
        return ["wsl", "bash", "-lc", f"cd {project_root} && poetry run {inner}"]
