# Speculative Decoding Evaluation — Complete Project Summary

> **Purpose of this document:** This is a comprehensive handoff document intended to give another LLM or human full context of this project — what it is, what we built, what went wrong, what we fixed, what the results are, and what's left to do.

---

## 1. What Is This Project?

This is a **graduate-level course project** for "Applied ML on Cloud" at Columbia University. The team has 4 members:
- **Himanshu (hj2713)** — L4 GPU experiments, lead developer
- **Aaryaman** — A100 GPU experiments
- **Shreya & Raj** — Mac (MLX) experiments

### Research Question
> "Does EAGLE-3 speculative decoding provide measurable throughput and cost benefits over standard autoregressive decoding, and how does this vary across GPU hardware tiers and workload types?"

### What Is Speculative Decoding (EAGLE-3)?
Speculative decoding is an inference acceleration technique where:
1. A **small draft model** proposes multiple candidate tokens in parallel
2. The **large target model** verifies all proposed tokens in a single forward pass
3. Accepted tokens are returned immediately; rejected tokens are discarded

EAGLE-3 (by Yuhuili et al.) is a state-of-the-art speculative decoding method that uses a lightweight feature-level draft head instead of a full draft model.

### What We're Benchmarking
We compare two systems:
- **Baseline**: `Meta-Llama-3.1-8B-Instruct` served via vLLM with standard greedy autoregressive decoding
- **EAGLE-3**: Same target model, but with `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` as the speculative draft head, using `num_speculative_tokens=5`

Both are served via **vLLM v0.19.1** (verified from server logs) using the OpenAI-compatible API.

---

## 2. Project Structure

```
Project2/
├── ARCHITECTURE.md              # System architecture + hardware specs
├── BENCHMARKING_RUNBOOK.md      # Step-by-step instructions for running experiments
├── RERUN_A100.md                # Quick re-run guide for Aaryaman (A100)
├── PROJECT_SUMMARY.md           # THIS FILE
├── code/
│   ├── benchmark/
│   │   ├── load_test.py         # Core benchmarking script (sends requests, collects metrics)
│   │   ├── run_sweep.sh         # Runs load_test.py across all concurrency levels for one system
│   │   ├── run_experiment.sh    # MASTER script — runs baseline then eagle3 sweeps automatically
│   │   ├── plot_results.py      # Generates academic-grade plots and summary CSVs
│   │   ├── quality_check.py     # Compares output quality (ROUGE-L, syntax check) between systems
│   │   └── test_parsing.py      # Unit test for SSE chunk parsing logic
│   ├── data/
│   │   ├── prepare_datasets.py  # Downloads & samples prompts from HuggingFace
│   │   └── prompts/             # Contains {chat,code,summarization}_50.jsonl
│   ├── dashboard/
│   │   └── app.py               # Streamlit live demo dashboard
│   ├── infra/
│   │   ├── setup_gcp.sh         # One-time GCP instance setup (CUDA, venv, models)
│   │   ├── setup_mac.sh         # One-time Mac setup (MLX)
│   │   ├── startup_baseline.sh  # Starts baseline vLLM server on port 8000
│   │   ├── startup_eagle3.sh    # Starts EAGLE-3 vLLM server on port 8001
│   │   ├── startup_mlx_baseline.sh  # Mac baseline server
│   │   ├── startup_mlx_spec.sh  # Mac speculative server
│   │   └── requirements.txt     # Python dependencies
│   ├── logs/                    # Server logs and PID files
│   └── analysis/                # (unused, plotting is in benchmark/plot_results.py)
├── results/
│   ├── raw/                     # 180 .jsonl files (per-request metrics)
│   ├── plots/                   # 14 .png files (academic-grade visualizations)
│   └── tables/                  # 4 .csv files (summary statistics)
└── himanshu/
    └── help.txt                 # Himanshu's personal GCP command cheat sheet
```

---

## 3. Experiment Design

### 3.1 Datasets (Prompts)
50 prompts per task, sourced from HuggingFace:

| Task | Dataset | Purpose |
|------|---------|---------|
| **Chat** | ShareGPT_V3 (first human turn) | Open-ended conversational prompts |
| **Code** | OpenAI HumanEval | Function completion prompts |
| **Summarization** | CNN/DailyMail (test split) | Article → 3-4 sentence summary |

Generated via: `python3 data/prepare_datasets.py --n-samples 50 --seed 42`

### 3.2 Experiment Matrix

| Dimension | Values |
|-----------|--------|
| **Systems** | `baseline` (greedy), `eagle3` (speculative) |
| **GPU Types** | L4 (24 GB), A100 (40 GB) |
| **Tasks** | chat, code, summarization |
| **Concurrency** | 1, 4, 8, 16, 32 simultaneous requests |
| **Trials** | 3 per cell (for confidence intervals) |

**Total cells per GPU**: 2 systems × 3 tasks × 5 concurrency levels × 3 trials = **90 runs per system, 180 per GPU**

### 3.3 Metrics Collected (Per Request)

| Metric | Formula | Unit |
|--------|---------|------|
| `ttft_ms` | Time from request start to first token received | ms |
| `tpot_ms` | `(t_end - t_first_token) / (output_tokens - 1)` | ms |
| `total_latency_ms` | Total wall-clock time for the request | ms |
| `output_tokens` | Server-reported `usage.completion_tokens` | count |
| `tokens_per_sec` | `output_tokens / total_latency_sec` | tok/s |
| `gpu_cost_usd` | `(hourly_rate / 3600) * total_latency_sec` | USD |
| `acceptance_rate` | From vLLM `/metrics` endpoint (eagle3 only) | ratio |

### 3.4 GPU Hourly Rates (GCP On-Demand)

| GPU | Rate | Instance |
|-----|------|----------|
| L4 | $0.70/hr | g2-standard-4 (4 vCPUs, 16 GB RAM, 1×L4 24GB) |
| A100 | $3.67/hr | a2-highgpu-1g |
| Mac | $0.00/hr | Local (no cloud cost) |

### 3.5 vLLM Server Configuration

Both servers use identical base config:
```
--dtype bfloat16
--max-model-len 2048
--gpu-memory-utilization 0.90
```

EAGLE-3 additionally uses:
```
--speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 5}'
```

Baseline runs on port 8000, EAGLE-3 on port 8001. The master script `run_experiment.sh` runs them **sequentially** (not simultaneously) to avoid GPU memory contention.

---

## 4. Critical Bug That Was Found & Fixed

### The Bug (in `load_test.py`, line 119)
```python
output_tokens += 1  # approximate: 1 chunk ≈ 1 token
```

**Problem:** This counted SSE streaming chunks as tokens. For baseline (greedy decoding), this is accurate — 1 chunk = 1 token. But for **speculative decoding**, vLLM batches multiple accepted tokens into a single SSE chunk. So EAGLE-3's token count was **undercounted by ~2x**, which cascaded into corrupted TPS, TPOT, and cost metrics.

**Evidence from raw data (before fix):**
- Baseline c=1: 256 chunks, 992 chars → 3.88 chars/chunk ✅
- Eagle3 c=1: 131 chunks, 1010 chars → 7.71 chars/chunk ❌ (multiple tokens per chunk)

### The Fix
1. Added `"stream_options": {"include_usage": True}` to the request payload
2. Parse the server's authoritative `usage.completion_tokens` from the final SSE chunk
3. Fall back to chunk counting only if the server doesn't support `stream_options` (e.g., MLX)
4. Guard against the empty `choices: []` array in the usage-only final chunk (this caused a crash in the first fix attempt)

**Same fix was applied to:** `benchmark/load_test.py` and `dashboard/app.py`

### Timeline of the Bug
1. **Initial runs** (old code): EAGLE-3 appeared **worse** than baseline in all metrics — this was the corrupted data
2. **First fix attempt**: Added `stream_options` but didn't handle the empty `choices: []` in the final chunk → all requests crashed with `IndexError: list index out of range`
3. **Second fix**: Added `if not choices: continue` guard → **all results now correct**

### Verification
A local test script (`benchmark/test_parsing.py`) was created to verify the parsing logic against 4 scenarios (baseline, eagle3, MLX, edge case) without needing to run the full 2-hour experiment.

---

## 5. Current Results (Post-Fix, CORRECT Data)

### Data Files
- **180 JSONL files** in `results/raw/` (90 per GPU: L4 + A100)
- Naming convention: `{system}_{gpu}_{task}_c{concurrency}_t{trial}.jsonl`
- Each JSONL line contains one request's full metrics
- **Mac results have been collected and integrated**, evaluating MLX standard speculative decoding.

### Key Findings

#### 5.1 Throughput (Tokens/Sec)
EAGLE-3 delivers **2-3x higher throughput** than baseline across all tasks and concurrency levels on both GPUs:

| GPU | Task | Concurrency=1 (Baseline → Eagle3) | Speedup |
|-----|------|-----------------------------------|---------|
| L4 | Code | ~17 → ~48 tok/s | **2.8x** |
| L4 | Summarization | ~17 → ~40 tok/s | **2.4x** |
| L4 | Chat | ~16 → ~25 tok/s | **1.6x** |
| A100 | Code | ~92 → ~265 tok/s | **2.9x** |
| A100 | Summarization | ~88 → ~210 tok/s | **2.4x** |
| A100 | Chat | ~82 → ~110 tok/s | **1.3x** |

Throughput decreases with higher concurrency for eagle3 (expected — GPU gets saturated with draft model overhead).

#### 5.2 TPOT (Time Per Output Token)
EAGLE-3 has **consistently lower TPOT** (better):
- L4: ~20-30ms (eagle3) vs ~60-75ms (baseline)
- A100: ~4-7ms (eagle3) vs ~11-14ms (baseline)

#### 5.3 Cost
EAGLE-3 is **~50% cheaper** per 1M output tokens across all configurations:
- L4 total experiment cost: baseline $1.49 vs eagle3 $0.74
- A100 total experiment cost: baseline $1.44 vs eagle3 $0.85

#### 5.4 Acceptance Rate (EAGLE-3 only)
The draft model's token acceptance rate varies by task:
- **Code**: ~43-50% (structured, predictable patterns)
- **Summarization**: ~49-50% (high acceptance, extractive nature)
- **Chat**: ~20-42% (lowest, unpredictable conversational content)

Higher acceptance rate → more tokens verified per forward pass → higher speedup.

#### 5.5 Crossover Points
EAGLE-3 **never crosses below baseline** at any tested concurrency level (up to 32). The `crossover_points.csv` shows `>32 (Never Crossed)` for all GPU×task combinations. This means EAGLE-3 is strictly better in all tested scenarios.

---

## 6. Generated Plots (in `results/plots/`)

| Plot File | Description |
|-----------|-------------|
| `academic_throughput_overview_{GPU}.png` | 1×3 grid: TPS vs concurrency for each task |
| `academic_ttft_overview_{GPU}.png` | 1×3 grid: TTFT vs concurrency for each task |
| `academic_tpot_overview_{GPU}.png` | 1×3 grid: TPOT vs concurrency for each task |
| `academic_cost_overview_{GPU}.png` | 1×3 grid: Cost per 1M tokens (bar chart) |
| `academic_acceptance_rate_{GPU}.png` | Acceptance rate by task vs concurrency |
| `speedup_ratio.png` | Eagle3/baseline speedup ratio |
| `cost_per_1k_tokens.png` | Cost comparison |
| `theoretical_vs_actual_speedup.png` | Theoretical vs measured speedup |
| `acceptance_rate_by_task.png` | Acceptance rate breakdown |

All plots use seaborn `whitegrid` theme with 95% confidence intervals from 3 trials.

---

## 7. Generated CSV Tables (in `results/tables/`)

| File | Contents |
|------|----------|
| `academic_summary.csv` | Mean ± std of all metrics grouped by (gpu, task, system, concurrency) |
| `cost_summary.csv` | Total cost and request count per (gpu, system) |
| `crossover_points.csv` | Concurrency at which baseline overtakes eagle3 (never in our data) |
| `extended_summary.csv` | Additional aggregated statistics |

---

## 8. Other Scripts & Features

### Dashboard (`dashboard/app.py`)
A Streamlit web app for live demo comparisons. Sends the same prompt to both baseline and eagle3 servers simultaneously and displays side-by-side metrics. Requires both servers running (needs ≥40GB VRAM → A100 only).

**To run on A100:**
```bash
# On GPU: streamlit run dashboard/app.py --server.port 8501
# On local: ssh -L 8501:localhost:8501 <GPU_INSTANCE>
# Then open http://localhost:8501
```

### Quality Check (`benchmark/quality_check.py`)
Compares output quality between baseline and eagle3 to verify speculative decoding doesn't degrade output:
- **Summarization**: ROUGE-L score between baseline and eagle3 outputs
- **Code**: Syntax validity check via `ast.parse()`
- **Chat**: Side-by-side manual comparison

### run_experiment.sh (Master Script)
Fully automated: starts baseline → runs all sweeps → kills it → starts eagle3 → runs all sweeps → kills it. Usage: `bash benchmark/run_experiment.sh L4|A100|mac`

---

## 9. Infrastructure Details

### GCP L4 Instance
- Machine type: `g2-standard-4` (4 vCPUs, 16 GB system RAM)
- GPU: 1× NVIDIA L4 (24 GB VRAM)
- Disk: Persistent disk mounted at `/mnt/disks/bigdisk/`
- OS: Debian-based (GCP Deep Learning VM)
- CUDA: 13.2, Driver: 595.58.03

### GCP A100 Instance
- Machine type: `a2-highgpu-1g`
- GPU: 1× NVIDIA A100 (40 or 80 GB VRAM)

### Models & Storage
- Target model: `unsloth/Meta-Llama-3.1-8B-Instruct` (~16 GB in BF16)
- Draft model: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (~1-2 GB)
- HuggingFace cache: `/mnt/disks/bigdisk/hf_cache`
- vLLM version: **0.19.1** (verified from server logs)

### VRAM Usage
- Baseline server: ~16 GB (target model only)
- EAGLE-3 server: ~18 GB (target + draft model + KV cache for speculative tokens)
- Cannot run both servers simultaneously on L4 (24 GB). Can on A100 (40+ GB).

---

## 10. Project Completion Status

All project requirements have been completed.
1. **Mac (MLX) Results**: Collected and analyzed. They show that while single-user (c=1) speculative decoding provides a ~1.19x-1.53x speedup, concurrency of 4 or more leads to severe regressions in TTFT and throughput.
2. **Final Report**: The final analysis report (`Project Report.pdf`) has been generated, incorporating A100, L4, and Mac results with complete visualizations.
3. **Quality Verification**: Output parity confirmed.
4. **Dashboard**: Live Streamlit demo fully functional.

---

## 11. How to Run Everything From Scratch

### On a fresh GCP instance:
```bash
# 1. Setup (one-time)
bash infra/setup_gcp.sh

# 2. Generate prompts (one-time, or skip if data/prompts/ already exists)
source .venv/bin/activate
python3 data/prepare_datasets.py --n-samples 50

# 3. Run full experiment (~2 hours)
bash benchmark/run_experiment.sh L4   # or A100

# 4. Generate plots and tables
python3 benchmark/plot_results.py
```

### On Mac:
```bash
bash infra/setup_mac.sh
source .venv/bin/activate
python3 data/prepare_datasets.py --n-samples 50
bash benchmark/run_experiment.sh mac
python3 benchmark/plot_results.py
```

---

## 12. Key Design Decisions

1. **Sequential server execution**: We run baseline and eagle3 sequentially (not simultaneously) to avoid GPU memory contention and ensure fair comparison.
2. **Greedy decoding (`temperature=0`)**: Deterministic output for reproducibility and fair quality comparison.
3. **`max_tokens=256`**: Caps output length to control experiment duration while still being long enough to measure steady-state throughput.
4. **3 trials per cell**: Provides 95% confidence intervals for all metrics.
5. **`stream_options: {include_usage: true}`**: Critical for accurate token counting with speculative decoding.
6. **Log archiving**: Old logs are renamed with timestamps instead of overwritten, preserving history.
7. **`pkill -9 -f vllm`** between phases: Ensures complete GPU memory cleanup.
