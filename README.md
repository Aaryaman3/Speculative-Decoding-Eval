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

| GPU / Device          | Memory         | Role                                                   |
| --------------------- | -------------- | ------------------------------------------------------ |
| NVIDIA A100-SXM4-80GB | 80 GB VRAM     | Aaryaman — both servers fit simultaneously             |
| NVIDIA L4             | 24 GB VRAM     | Himanshu — servers run sequentially                    |
| Apple M-Series        | Unified Memory | Shreya & Raj — MLX baseline vs standard spec decoding  |

---

## Models

| Role            | Model                                 |
| --------------- | ------------------------------------- |
| Target          | `unsloth/Meta-Llama-3.1-8B-Instruct`  |
| Draft (EAGLE-3) | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |

---

## Benchmark Design

- **Tasks:** Chat (ShareGPT), Code (HumanEval), Summarization (CNN/DailyMail)
- **Concurrency levels:** 1, 4, 8, 16, 32, 64, 128 (A100 and L4); 1, 4, 8, 16, 32 (Mac)
- **Trials per cell:** 3
- **Max output tokens:** 256
- **Temperature:** 0 (greedy, deterministic)
- **Total result files:** 252 sweep files + 2 quality files for A100/L4, plus complete Mac metric summaries.

---

## Key Findings

### Throughput (tokens/sec)

| GPU / Device | System   | c=1  | c=8  | c=32 | c=64 | c=128 |
| ------------ | -------- | ---- | ---- | ---- | ---- | ----- |
| A100         | Baseline | ~92  | ~88  | ~73  | ~55  | ~62   |
| A100         | EAGLE-3  | ~205 | ~180 | ~114 | ~85  | ~85   |
| L4           | Baseline | ~17  | ~16  | ~14  | ~12  | ~10   |
| L4           | EAGLE-3  | ~38  | ~38  | ~26  | ~17  | ~12   |
| Mac M-Series | Baseline | ~17  | ~8.5 | ~5.4 | —    | —     |
| Mac M-Series | MLX Spec | ~24  | ~8.0 | ~8.1 | —    | —     |

> **Note:** An earlier version of `load_test.py` counted SSE streaming _chunks_ instead of actual tokens. With speculative decoding, each chunk may contain multiple accepted tokens, so TPS was severely undercounted. The table above reflects corrected measurements using `stream_options: {"include_usage": true}` (authoritative `completion_tokens` from the vLLM server).

### EAGLE-3 Speedup Summary

| GPU / Device | Task          | c=1 speedup | c=8 speedup        | c=64 speedup | c=128 speedup | Crossover         |
| ------------ | ------------- | ----------- | ------------------ | ------------ | ------------- | ----------------- |
| A100         | Code          | **2.86×**   | 2.37×              | **1.58×**    | **1.58×**     | >128 (none)       |
| A100         | Chat          | **1.49×**   | 1.94×              | **1.32×**    | **1.32×**     | >128 (none)       |
| A100         | Summarization | **2.35×**   | 1.80×              | **1.87×**    | **1.17×**     | >128 (none)       |
| L4           | Code          | **2.88×**   | 2.77×              | **1.69×**    | **1.42×**     | >128 (none)       |
| L4           | Chat          | **1.53×**   | 2.18×              | **1.50×**    | **1.27×**     | >128 (none)       |
| L4           | Summarization | **2.41×**   | 2.17×              | **1.25×**    | **~1.04×**    | ~128 (near parity)|
| Mac M-Series | Code          | **1.53×**   | 1.06×              | —            | —             | c=4               |
| Mac M-Series | Chat          | **1.40×**   | 0.76× (Regression) | —            | —             | c=4               |
| Mac M-Series | Summarization | **1.19×**   | 1.01×              | —            | —             | c=4               |

**EAGLE-3 consistently improves throughput on the A100 across the full tested range (c=1 to c=128) — no crossover point was found.** Speedup is task-dependent: code generation benefits most (up to 2.86× at c=1), while chat gains are more modest (1.49× at c=1, likely due to lower draft acceptance on diverse conversational prompts). At high concurrency the advantage narrows but remains positive across all tasks. On L4, the crossover occurs around c=32 due to tighter VRAM. Mac MLX crossover hits at c=4.

### EAGLE-3 Acceptance Rate (~50%)

The draft model accepted ~50% of speculative tokens across all tasks and concurrency levels. With k=5 speculative tokens, the Leviathan et al. (2023) theoretical speedup formula predicts S = (1 + 0.5×5)/(1 + 0.5) ≈ **2.33×** — closely matching observed results for code (2.86×) and summarization (2.35×) at c=1. Chat shows a lower c=1 speedup (~1.49×), consistent with a lower effective acceptance rate on diverse conversational prompts.

### Quality Check (A100, summarization, c=1, trial=99)

Both baseline and EAGLE-3 produced **identical output text**, confirming speculative decoding preserves output distribution at temperature=0.

### Hardware Comparison: A100 vs L4 vs Mac (Apple Silicon)

- **A100 (80GB)**: The high compute density and massive memory bandwidth makes it the absolute winner. It offers ~5–6× faster raw baseline throughput than the L4. It sees the largest absolute EAGLE-3 gains because it has the compute headroom to evaluate the draft model without bottlenecking the target model.
- **L4 (24GB)**: The budget option. EAGLE-3 speedup holds across the full tested range (c=1 to c=128) for code and chat, with summarization reaching near-parity (~1.04×) at c=128. Speedup narrows more steeply than on the A100 due to tighter VRAM (24 GB vs 80 GB), but speculative decoding is still beneficial at all tested concurrency levels.
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
