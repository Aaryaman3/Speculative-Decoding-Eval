# Speculative Decoding Eval — EAGLE-3 Benchmark

Benchmarking EAGLE-3 speculative decoding against greedy baseline using `unsloth/Meta-Llama-3.1-8B-Instruct` on NVIDIA A100 (80GB) and L4 (24GB) GPUs via vLLM, and standard speculative decoding on Apple Silicon via MLX.

---

## Repository Structure

```text
Code/
├── benchmark/       Load tester, sweep runner, plot generator, quality checker
├── data/prompts/    50 prompts each for chat, code, summarization (seed=42)
├── infra/           Server startup scripts and setup for GCP + Mac
├── dashboard/       Streamlit live demo app
└── results/
    ├── raw/         JSONL result files (per-request metrics for A100 & L4)
    ├── plots/       Academic-grade PNG plots (A100, L4, and Mac comparisons)
    └── tables/      CSV summary tables (A100, L4, and Mac metrics)
Documentation/
├── ARCHITECTURE.md          System architecture + hardware specs
├── BENCHMARKING_RUNBOOK.md  Step-by-step instructions for running experiments
└── PROJECT_SUMMARY.md       Complete project summary and findings
Report & Presentation/
└── Project Report.pdf         Final academic report with all metrics and analysis
└── Project Presentation.pdf   Presentation Slides
└── Project Proposal.pdf       Academic Proposal which we submitted
```

---

## Hardware

| GPU / Device          | Memory         | Role                                                  |
| --------------------- | -------------- | ----------------------------------------------------- |
| NVIDIA A100-SXM4-80GB | 80 GB VRAM     | Aaryaman — both servers fit simultaneously            |
| NVIDIA L4             | 24 GB VRAM     | Himanshu — servers run sequentially                   |
| Apple M-Series        | Unified Memory | Shreya & Raj — MLX baseline vs standard spec decoding |

---

## Models

| Role            | Model                                 |
| --------------- | ------------------------------------- |
| Target          | `unsloth/Meta-Llama-3.1-8B-Instruct`  |
| Draft (EAGLE-3) | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |

---

## Benchmark Design

- **Tasks:** Chat (ShareGPT), Code (HumanEval), Summarization (CNN/DailyMail)
- **Concurrency levels:** 1, 4, 8, 16, 32
- **Trials per cell:** 3
- **Max output tokens:** 256
- **Temperature:** 0 (greedy, deterministic)
- **Total result files:** 180 sweep files + 2 quality files for A100/L4, plus complete Mac metric summaries.

---

## Key Findings

### Throughput (tokens/sec)

| GPU / Device | System   | c=1  | c=8  | c=32 |
| ------------ | -------- | ---- | ---- | ---- |
| A100         | Baseline | ~92  | ~87  | ~68  |
| A100         | EAGLE-3  | ~218 | ~173 | ~74  |
| L4           | Baseline | ~17  | ~16  | ~13  |
| L4           | EAGLE-3  | ~33  | ~25  | ~14  |
| Mac M-Series | Baseline | ~17  | ~8.5 | ~5.4 |
| Mac M-Series | MLX Spec | ~24  | ~8.0 | ~8.1 |

> **Note:** An earlier version of `load_test.py` counted SSE streaming _chunks_ instead of actual tokens. With speculative decoding, each chunk may contain multiple accepted tokens, so TPS was severely undercounted. The table above reflects corrected measurements using `stream_options: {"include_usage": true}` (authoritative `completion_tokens` from the vLLM server).

### EAGLE-3 Speedup Summary

| GPU / Device | Task          | c=1 speedup | c=8 speedup        | Crossover  |
| ------------ | ------------- | ----------- | ------------------ | ---------- |
| A100         | Code          | **2.87×**   | 1.8×               | >32 (None) |
| A100         | Chat          | **2.33×**   | 1.98×              | >32 (None) |
| A100         | Summarization | **2.33×**   | ~1.7×              | >32 (None) |
| L4           | Code          | **~1.9×**   | ~1.5×              | ~32 (None) |
| L4           | Chat          | **~1.8×**   | ~1.4×              | ~32 (None) |
| Mac M-Series | Code          | **1.53×**   | 1.06×              | c=4        |
| Mac M-Series | Chat          | **1.40×**   | 0.76× (Regression) | c=4        |
| Mac M-Series | Summarization | **1.19×**   | 1.01×              | c=4        |

**EAGLE-3 consistently improves throughput on both GPUs.** The benefit is largest at low concurrency (memory-bandwidth-bound regime) and narrows as concurrency increases toward the compute-bound regime. The A100's higher compute density amplifies the acceptance-rate gains most.

### EAGLE-3 Acceptance Rate (~50%)

The draft model accepted ~50% of speculative tokens across all tasks and concurrency levels. With k=5 speculative tokens, the Leviathan et al. (2023) theoretical speedup formula predicts S = (1 + 0.5×5)/(1 + 0.5) ≈ **2.33×** — closely matching observed results at c=1.

### Quality Check (A100, summarization, c=1, trial=99)

Both baseline and EAGLE-3 produced **identical output text**, confirming speculative decoding preserves output distribution at temperature=0.

### Hardware Comparison: A100 vs L4 vs Mac (Apple Silicon)

- **A100 (80GB)**: The high compute density and massive memory bandwidth makes it the absolute winner. It offers ~5–6× faster raw baseline throughput than the L4. It sees the largest absolute EAGLE-3 gains because it has the compute headroom to evaluate the draft model without bottlenecking the target model.
- **L4 (24GB)**: The budget option. It degrades more at high concurrency under EAGLE-3 due to its tighter VRAM budget (24 GB vs 80 GB), but speculative decoding is still universally beneficial and saves ~50% in cost.
- **Mac (Unified Memory)**: The edge/local option. Standard MLX speculative decoding is beneficial **only for a single user (c=1)**, offering up to a 1.53× speedup. At higher concurrency (c≥4), it suffers catastrophic regressions. For instance, Chat at c=32 sees TTFT skyrocket to over 24 seconds with extreme variance, making it completely unsuitable for multi-user server deployment.

---

## Reproducing the Benchmark

Exact, step-by-step instructions for running the project on both GCP (vLLM) and Mac (MLX) are documented in [`Documentation/BENCHMARKING_RUNBOOK.md`](Documentation/BENCHMARKING_RUNBOOK.md).

Quick reference:

### 1. Setup (GCP GPU instance)

```bash
export HF_TOKEN=hf_xxxx
bash Code/infra/setup_gcp.sh
```

### 2. Prepare datasets

```bash
python3 Code/data/prepare_datasets.py --n-samples 50
```

### 3. Run full experiment

```bash
bash Code/benchmark/run_experiment.sh A100   # or L4 or mac
```

### 4. Generate plots

```bash
python3 Code/benchmark/plot_results.py
```

---

## Results

Plots are in [`Code/results/plots/`](results/plots/) and summary tables in [`Code/results/tables/`](results/tables/).
