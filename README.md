# whichllm

[![PyPI version](https://img.shields.io/pypi/v/whichllm)](https://pypi.org/project/whichllm/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml/badge.svg)](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml)

**Find the best local LLM that actually runs on your hardware.**

Auto-detects your GPU/CPU/RAM and ranks the top models from HuggingFace that fit your system.

[日本語版はこちら](docs/README.ja.md)

![demo](assets/demo.gif)

## Why whichllm?

**One command. Real answers.** No TUI to learn, no keybindings to memorize.

| | whichllm | Others (TUI-based) |
|---|---|---|
| **Getting results** | `whichllm` — done | Launch TUI → navigate → search → filter |
| **Model data** | Live from HuggingFace API | Static built-in database |
| **Benchmarks** | Real eval scores with confidence | Fixed quality scores |
| **Scriptable** | `whichllm --json \| jq` | Requires special flags |
| **Learning curve** | Zero | Vim keybindings required |

## Features

- **Auto-detect hardware** — NVIDIA, AMD, Apple Silicon, CPU-only
- **Smart ranking** — Scores models by VRAM fit, speed, and benchmark quality
- **One-command chat** — `whichllm run` downloads and starts a chat session instantly
- **Code snippets** — `whichllm snippet` prints ready-to-run Python for any model
- **Live data** — Fetches models directly from HuggingFace (cached for performance)
- **Benchmark-aware** — Integrates real eval scores with confidence-based dampening
- **Task profiles** — Filter by general, coding, vision, or math use cases
- **GPU simulation** — Test with any GPU: `whichllm --gpu "RTX 4090"`
- **Hardware planning** — Reverse lookup: `whichllm plan "llama 3 70b"`
- **JSON output** — Pipe-friendly: `whichllm --json`

## Run & Snippet

**Try any model with a single command.** No manual installs needed — whichllm creates an isolated environment via `uv`, installs dependencies, downloads the model, and starts an interactive chat.

![run demo](assets/demo-run.gif)

```bash
# Chat with a model (auto-picks the best GGUF variant)
whichllm run "qwen 2.5 1.5b gguf"

# Auto-pick the best model for your hardware and chat
whichllm run

# CPU-only mode
whichllm run "phi 3 mini gguf" --cpu-only
```

Works with **all model formats**:
- **GGUF** — via `llama-cpp-python` (lightweight, fast)
- **AWQ / GPTQ** — via `transformers` + `autoawq` / `auto-gptq`
- **FP16 / BF16** — via `transformers`

Get a **copy-paste Python snippet** instead:

```bash
whichllm snippet "qwen 7b"
```

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
)
print(output["choices"][0]["message"]["content"])
```

## Install

### pipx (recommended)

```bash
pipx install whichllm
```

### Homebrew

```bash
brew tap Andyyyy64/whichllm
brew install whichllm
```

### pip

```bash
pip install whichllm
```

### Development

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --dev
uv run whichllm
uv run pytest
```

## Usage

```bash
# Auto-detect hardware and show best models
whichllm

# Simulate a GPU (e.g. planning a purchase)
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5090"

# CPU-only mode
whichllm --cpu-only

# More results / filters
whichllm --top 20
whichllm --quant Q4_K_M
whichllm --min-speed 30
whichllm --evidence base   # allow id/base-model matches
whichllm --evidence strict # id-exact only (same as --direct)
whichllm --direct

# JSON output
whichllm --json

# Force refresh (ignore cache)
whichllm --refresh

# Show hardware info only
whichllm hardware

# Plan: what GPU do I need for a specific model?
whichllm plan "llama 3 70b"
whichllm plan "Qwen2.5-72B" --quant Q8_0
whichllm plan "mistral 7b" --context-length 32768

# Run: download and chat with a model instantly
whichllm run "qwen 2.5 1.5b gguf"
whichllm run                       # auto-pick best for your hardware

# Snippet: print ready-to-run Python code
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

## Integrations

### Ollama

Find the best model and run it directly:

```bash
# Pick the top model and run it with Ollama
whichllm --top 1 --json | jq -r '.models[0].model_id' | xargs ollama run

# Find the best coding model
whichllm --profile coding --top 1 --json | jq -r '.models[0].model_id' | xargs ollama run
```

### Shell alias

Add to your `.bashrc` / `.zshrc`:

```bash
alias bestllm='whichllm --top 1 --json | jq -r ".models[0].model_id"'
# Usage: ollama run $(bestllm)
```

## Scoring

Each model gets a score from 0 to 100.

| Factor | Points | Description |
|--------|--------|-------------|
| Model size | 0-40 | Larger models generally produce better output |
| Benchmark | 0-10 | Arena ELO / Open LLM Leaderboard scores |
| Speed | 0-20 | Higher tok/s = more practical to use |
| Source trust | -5 to +5 | Official repos get a bonus, repackagers get a penalty |
| Popularity | 0-3 | Downloads and likes as tiebreaker |

Score markers:
- **`~`** (yellow) — No direct benchmark yet. Score estimated from the model family
- **`?`** (yellow) — No benchmark data available

## How it works

### Data pipeline

1. **Model fetching** — Fetches popular models from HuggingFace API:
   - Text-generation (downloads + recently updated)
   - GGUF-filtered (separate query for coverage)
   - Vision models (`image-text-to-text`) when `--profile vision` or `any`
2. **Benchmark sources** — Chatbot Arena ELO (priority) + Open LLM Leaderboard
3. **Benchmark evidence** — Four levels of confidence:
   - `direct` — Exact model ID match
   - `variant` — Suffix-stripped or -Instruct variant
   - `base_model` — Base model from cardData
   - `line_interp` — Size-aware interpolation within model family
4. **Cache** — `~/.cache/whichllm/`:
   - `models.json` — 6h TTL
   - `benchmark.json` — 24h TTL

### Ranking engine

1. **Hardware detection** — NVIDIA (nvidia-ml-py), AMD (dbgpu/ROCm), Apple Silicon (Metal), CPU cores, RAM, disk
2. **VRAM estimation** — Weights + KV cache + activation + framework overhead (~500MB)
3. **Compatibility** — Full GPU / Partial Offload / CPU-only; compute capability and OS checks
4. **Speed** — tok/s from GPU memory bandwidth lookup (constants.py)
5. **Scoring** — Benchmark (with confidence dampening), size, quantization penalty, fit type, speed, popularity, source trust (official vs repackager)
6. **Backend filter** — Apple Silicon and CPU-only restrict to GGUF for stability; Linux+NVIDIA allows AWQ/GPTQ

### Project structure

```
src/whichllm/
├── cli.py              # Typer CLI: main, plan, run, snippet, hardware
├── constants.py        # GPU bandwidth, quantization bytes, compute capability
├── hardware/
│   ├── detector.py     # Orchestrates GPU/CPU/RAM detection
│   ├── nvidia.py       # NVIDIA GPU via nvidia-ml-py
│   ├── amd.py          # AMD GPU (Linux)
│   ├── apple.py        # Apple Silicon (Metal)
│   ├── cpu.py          # CPU name, cores, AVX support
│   ├── memory.py       # RAM and disk free
│   ├── gpu_simulator.py # --gpu flag: synthetic GPU from name
│   └── types.py        # GPUInfo, HardwareInfo
├── models/
│   ├── fetcher.py      # HuggingFace API, model parsing, evalResults
│   ├── benchmark.py    # Arena ELO, Leaderboard (parquet/rows API)
│   ├── grouper.py      # Family grouping by base_model and name
│   ├── cache.py        # JSON cache with TTL
│   └── types.py        # ModelInfo, GGUFVariant, ModelFamily
├── engine/
│   ├── vram.py         # VRAM = weights + KV cache + activation + overhead
│   ├── compatibility.py# Fit type, disk check, compute/OS warnings
│   ├── performance.py  # tok/s from bandwidth
│   ├── quantization.py # Bytes per weight, quality penalty, non-GGUF inference
│   ├── ranker.py       # Scoring, evidence filter, profile/match
│   └── types.py        # CompatibilityResult
└── output/
    └── display.py      # Rich table, JSON output, hardware/plan displays
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Requirements

- Python 3.11+
- NVIDIA GPU detection via `nvidia-ml-py` (included by default)
- AMD / Apple Silicon detected automatically

## License

MIT
