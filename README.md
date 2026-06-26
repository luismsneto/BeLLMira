# BeLLMira

BeLLMira is a laboratory toolkit for exploring, benchmarking, deploying, and experimenting with Large Language Models. It provides a unified framework for serving models via vLLM, evaluating them across multiple performance dimensions, and tracking experiments with MLflow.

## Features

- **Model serving** — Download models from Hugging Face and serve them via a vLLM OpenAI-compatible API with support for quantization, tensor parallelism, prefix caching, and structured output
- **Pluggable evaluators** — Measure context length scaling, time-to-first-token, parallel load throughput, classification accuracy, embedding quality, summarization quality, vision, and regression
- **Multi-model experiments** — Run evaluators concurrently across multiple models and log results to CSV and MLflow
- **Flexible data sources** — Load datasets from local files, Databricks DBFS, or Azure ADLS (`file:`, `dbfs:`, `adls:` path prefixes)
- **MLflow integration** — Track runs, log metrics and artifacts, register models, and generate Databricks-compatible MLmodel manifests

---

## Requirements

- Python ~3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- NVIDIA GPU with driver ≥ 560 (CUDA 12.9 runtime required)
- **WSL2** (Windows only) — vLLM is Linux-only; model serving must run on Linux. Notebooks and evaluators work on Windows.

---

## Platform architecture (Windows + WSL2)

BeLLMira runs across two environments on Windows:

| Environment | Purpose |
|---|---|
| **Windows** (Poetry venv) | Jupyter kernel — notebooks, evaluators, `ModelClient` |
| **WSL2** (Poetry venv) | vLLM server — `LLMModel.serve_model()` routes here automatically |

`serve_model()` detects Windows and transparently launches vLLM inside WSL2. No manual WSL2 interaction is needed from notebooks. Both environments use the same source files under `src/` via editable install — editing a `.py` file is immediately visible in both without reinstalling.

**When `poetry install` must be run on both sides:**
- Adding or removing a dependency in `pyproject.toml`
- Adding a new module that changes the package structure

Editing existing source files requires no reinstall on either side.

---

## Installation

### 1. Windows — notebook kernel

```powershell
git clone <repo-url>
cd BeLLMira
poetry install
```

Register as a Jupyter kernel:

```powershell
poetry run python -m ipykernel install --user --name bellmira --display-name "Python (bellmira)"
```

Verify PyTorch sees the GPU (run in a notebook cell):

```python
import torch
print(torch.__version__)          # should show +cu128 suffix
print(torch.cuda.is_available())  # should return True
```

Verify the correct build is loaded:

```python
from importlib.metadata import version
print(version("bellmira"))  # installed package version

import bellmira
print(bellmira.__file__)    # should point inside the Poetry virtualenv, not system Python
```

### 2. WSL2 — vLLM server

```bash
cd /mnt/c/Projectos/BeLLMira
poetry install
```

#### CUDA toolkit in WSL2 (required for vLLM ≥ 0.20)

vLLM uses `torch.compile` (Inductor) and FlashInfer JIT, both of which call `nvcc` at runtime. Without the CUDA toolkit in WSL2, serving will fail.

WSL2 **inherits the Windows PATH** in non-login shells, which means it can accidentally pick up `nvcc.exe` from a Windows CUDA installation — Linux cannot execute Windows binaries, causing `PermissionError: Permission denied: 'nvcc'`. Install the Linux toolkit and put it first in PATH:

```bash
# Add NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.9 compiler
sudo apt-get install -y cuda-nvcc-12-9

# Put the Linux nvcc BEFORE Windows PATH entries
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.profile
```

> **Why `~/.profile` and not just `~/.bashrc`?**
> `serve_model()` invokes `wsl bash -lc "..."` (a login shell). Login shells load `~/.profile` but not `~/.bashrc`. Without `~/.profile`, the PATH change has no effect and the Windows `nvcc` is found instead.

Verify the right `nvcc` is found:

```bash
which nvcc
# must be /usr/local/cuda-12.9/bin/nvcc — NOT a /mnt/c/... Windows path
nvcc --version
```

---

## Serving a model

```python
from bellmira.llm_model import LLMModel

llmmodel = LLMModel(repo_id="Qwen/Qwen3.5-4B")
llmmodel.serve_model(
    max_model_len=8192,
    max_num_seqs=190,       # must be ≤ available Mamba cache blocks reported at startup
    gpu_memory_utilization=0.9,
)
```

`serve_model()` prints the full command before launching so you can see exactly what is executed.

### Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `max_model_len` | 16384 | Maximum context length in tokens |
| `max_num_seqs` | None (vLLM default 256) | Must be ≤ available Mamba cache blocks — vLLM reports the maximum at startup |
| `gpu_memory_utilization` | None (vLLM default 0.9) | Fraction of GPU VRAM reserved; raising it increases available cache blocks |
| `dtype` | `"half"` | `"half"` = float16; use `"bfloat16"` for models that require it |
| `quantization` | None | `"awq"`, `"gptq"` for quantized models |
| `cpu_offload` | None | GB of weights to offload to RAM — useful when model barely fits VRAM |
| `enforce_eager` | False | Disables `torch.compile` and CUDA graphs; use only for debugging — requires no `nvcc` but reduces throughput |

### Serving a GGUF model

```python
llmmodel = LLMModel(
    repo_id="Qwen/Qwen3.5-9B-Base",
    hf_file_repo="owner/repo-GGUF",
    hf_filename="model.Q4_K_M.gguf",
    local_model_path="../.tmp",
)
llmmodel.download_hf_files()
llmmodel.serve_model(max_model_len=8192)
```

> **GGUF support:** vLLM supports GGUF via the `--model path/to/file.gguf --tokenizer hf/repo` pattern. However, support depends on the architecture. The `qwen35` GGUF architecture requires `transformers ≥ 5.5.1`. Check vLLM release notes before using a new model family.

### Mamba cache block errors

If you see:
```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (190).
```
Either lower `max_num_seqs` to the reported maximum, or increase `gpu_memory_utilization` to give vLLM more VRAM for cache blocks.

---

## vLLM version notes

| vLLM | CUDA wheel | nvcc needed | transformers |
|---|---|---|---|
| 0.19.0 | cu128 (via PyPI) | No | ≥ 4.57.6 |
| 0.23.0 | `cu129` wheel from GitHub releases | **Yes** (CUDA 12.9 toolkit) | ≥ 5.5.1 |

vLLM 0.23.0 is installed via a direct wheel URL (not PyPI) because the PyPI release targets CUDA 13.x (`libcudart.so.13`), which requires a full CUDA 13 toolkit upgrade. The `cu129` GitHub release wheel targets CUDA 12.9, which is compatible with existing CUDA 12.x runtimes.

---

## What NOT to do

**Do not run `!python -m vllm...` from a Windows notebook.**
The `!` shell magic uses the Windows Python kernel. vLLM is not installed on Windows (Linux-only marker in `pyproject.toml`). Even if it were, `uvloop` is also Linux-only and vLLM will fail to import. Use `LLMModel.serve_model()` instead.

**Do not install torch without the CUDA wheel index.**
`pip install torch` / `poetry add torch` on Windows installs the CPU-only build by default (`+cpu` suffix). Verify with `torch.cuda.is_available()` — it must return `True`. Torch is pinned to the `cu128` wheel source in `pyproject.toml`.

**Do not rely on `~/.bashrc` for WSL2 PATH changes.**
`serve_model()` spawns a login shell (`bash -lc`), which reads `~/.profile` but not `~/.bashrc`. Always add PATH exports to both files.

**Do not set `enforce_eager=True` in production.**
It disables `torch.compile` and CUDA graph capture, reducing throughput significantly. It is only useful for diagnosing startup failures when `nvcc` is missing.

**Do not skip `max_num_seqs` tuning on Mamba-architecture models (Qwen3.5, etc.).**
Mamba models require one dedicated cache block per concurrent sequence. vLLM reports the maximum safe value at startup — always set `max_num_seqs` to that value or lower.

---

## Data paths for evaluators

`ModelClassificationEvaluator` (and other data-loading evaluators) require a URI-style `data_path` with a source prefix:

| Prefix | Example |
|---|---|
| `file:` | `file:///tmp/data/dataset.parquet` |
| `dbfs:` | `dbfs:/mnt/data/dataset.parquet` |
| `adls:` | `adls:/container/path/dataset.parquet` |

Passing a bare path without a prefix causes `ValueError: not enough values to unpack`.

---

## Project structure

```
src/bellmira/
├── llm_model/
│   ├── llm_model.py          # LLMModel — download, serve, register
│   └── llm_model_client.py   # ModelClient — HTTP client for the inference API
├── evaluators/               # All inherit ModelEvaluatorInterface
│   ├── model_context_length_evaluator.py
│   ├── model_ttft_evaluator.py
│   ├── model_concurrent_load_evaluator.py
│   ├── model_classification_evaluator.py
│   ├── model_vision_evaluator.py
│   ├── model_embedding_quality_evaluator.py
│   ├── model_summarization_evaluator.py
│   └── model_regression_evaluator.py
├── llm_experiments/          # Multi-model orchestration, CSV logging
└── utils/                    # Context builders, ROUGE metrics, aggregation helpers
notebooks/                    # Tutorial notebooks (one per evaluator)
tutorial/                     # Step-by-step tutorial notebooks
```

## Quick start

```python
from bellmira.llm_model import LLMModel
from bellmira.llm_model.llm_model_client import ModelClient

# 1 — serve a model (blocking; runs vLLM in WSL2 on Windows)
llmmodel = LLMModel(repo_id="Qwen/Qwen3.5-4B")
llmmodel.serve_model(max_model_len=8192, max_num_seqs=190)

# 2 — query it
client = ModelClient(base_url="http://localhost:8080/")
print(client.get_model_name())

req = client.build_chat_request(
    user_prompt="Who are you?",
    model_name=client.get_model_name(),
    temperature=0.0,
    enable_thinking=False,
)
response = client.send_request(req)
print(response.json()["choices"][0]["message"]["content"])

# 3 — stream the response
for token in client.stream_chat_response(req):
    print(token, end="", flush=True)
```

## Dependencies

| Package | Version | Notes |
|---|---|---|
| torch / torchvision / torchaudio | ≥ 2.7.0 | CUDA 12.8 wheels (`cu128`) |
| vllm | 0.23.0+cu129 | Linux only — installed from GitHub release wheel |
| transformers | ≥ 5.5.1 | Required for vLLM 0.23.0; v4 codepath removed in vLLM 0.24.0 |
| mlflow | 3.3.2 | |
| numpy | ≥ 2.0.0 | |
| uvloop | 0.22.1 | Linux only |
| pydantic-extra-types | ≥ 2.10.5 | |
