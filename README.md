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
- **Concurrency levels:** 1, 4, 8, 16, 32, 64, 128 (A100 and L4); 1, 4, 8, 16, 32 (Mac)
- **Trials per cell:** 3
- **Max output tokens:** 256
- **Temperature:** 0 (greedy, deterministic)
- **Total result files:** 252 sweep files + 2 quality files for A100/L4, plus complete Mac metric summaries.

---

## Key Findings

### Throughput (tokens/sec) — Full Concurrency Range

**NVIDIA A100 (80GB)**

| Task          | System   | c=1  | c=4  | c=8  | c=16 | c=32 | c=64 | c=128 |
| ------------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| Chat          | Baseline | ~85  | ~91  | ~88  | ~85  | ~73  | ~59  | ~64   |
| Chat          | EAGLE-3  | ~111 | ~136 | ~173 | ~151 | ~115 | ~70  | ~88   |
| Code          | Baseline | ~92  | ~91  | ~89  | ~85  | ~74  | ~62  | ~64   |
| Code          | EAGLE-3  | ~265 | ~242 | ~212 | ~182 | ~132 | ~93  | ~101  |
| Summarization | Baseline | ~91  | ~88  | ~85  | ~78  | ~65  | ~44  | ~59   |
| Summarization | EAGLE-3  | ~211 | ~164 | ~150 | ~121 | ~86  | ~59  | ~68   |

**NVIDIA L4 (24GB)**

| Task          | System   | c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=128 |
| ------------- | -------- | --- | --- | --- | ---- | ---- | ---- | ----- |
| Chat          | Baseline | ~17 | ~16 | ~16 | ~16  | ~14  | ~13  | ~11   |
| Chat          | EAGLE-3  | ~25 | ~34 | ~36 | ~36  | ~26  | ~19  | ~14   |
| Code          | Baseline | ~17 | ~16 | ~16 | ~15  | ~14  | ~12  | ~10   |
| Code          | EAGLE-3  | ~49 | ~47 | ~44 | ~40  | ~29  | ~20  | ~14   |
| Summarization | Baseline | ~17 | ~16 | ~16 | ~15  | ~12  | ~10  | ~9    |
| Summarization | EAGLE-3  | ~40 | ~32 | ~33 | ~28  | ~20  | ~13  | ~9    |

**Apple M-Series (Unified Memory)**

| Task          | System   | c=1 | c=4 | c=8  | c=16 | c=32 |
| ------------- | -------- | --- | --- | ---- | ---- | ---- |
| Chat          | Baseline | ~17 | ~12 | ~8.5 | —    | ~5.4 |
| Chat          | MLX Spec | ~24 | ~11 | ~8.0 | —    | ~8.1 |
| Code          | Baseline | ~24 | ~18 | ~12  | —    | —    |
| Code          | MLX Spec | ~37 | ~19 | ~12  | —    | —    |
| Summarization | Baseline | ~18 | ~14 | ~9.5 | —    | —    |
| Summarization | MLX Spec | ~21 | ~14 | ~9.6 | —    | —    |

> **Note:** An earlier version of `load_test.py` counted SSE streaming _chunks_ instead of actual tokens. With speculative decoding, each chunk may contain multiple accepted tokens, so TPS was severely undercounted. The table above reflects corrected measurements using `stream_options: {"include_usage": true}` (authoritative `completion_tokens` from the vLLM server).

### EAGLE-3 Speedup Summary — Full Concurrency Range

**NVIDIA A100 (80GB)**

| Task          | c=1 speedup | c=4 speedup | c=8 speedup | c=16 speedup | c=32 speedup | c=64 speedup | c=128 speedup | Crossover |
| ------------- | ----------- | ----------- | ----------- | ------------ | ------------ | ------------ | ------------- | --------- |
| Code          | **2.87×**   | **2.67×**   | **2.37×**   | **2.13×**    | **1.80×**    | **1.50×**    | **1.57×**     | >128      |
| Chat          | **1.30×**   | **1.50×**   | **1.98×**   | **1.79×**    | **1.59×**    | **1.18×**    | **1.36×**     | >128      |
| Summarization | **2.33×**   | **1.88×**   | **1.77×**   | **1.55×**    | **1.33×**    | **1.33×**    | **1.16×**     | >128      |

**NVIDIA L4 (24GB)**

| Task          | c=1 speedup | c=4 speedup | c=8 speedup | c=16 speedup | c=32 speedup | c=64 speedup | c=128 speedup | Crossover |
| ------------- | ----------- | ----------- | ----------- | ------------ | ------------ | ------------ | ------------- | --------- |
| Code          | **2.88×**   | **2.90×**   | **2.79×**   | **2.62×**    | **2.11×**    | **1.70×**    | **1.41×**     | >128      |
| Chat          | **1.50×**   | **2.12×**   | **2.25×**   | **2.28×**    | **1.87×**    | **1.51×**    | **1.27×**     | >128      |
| Summarization | **2.40×**   | **2.00×**   | **2.10×**   | **1.91×**    | **1.57×**    | **1.25×**    | **1.05×**     | ~128      |

**Apple M-Series (Unified Memory)**

| Task          | c=1 speedup | c=4 speedup | c=8 speedup | c=16 speedup | c=32 speedup | Crossover |
| ------------- | ----------- | ----------- | ----------- | ------------ | ------------ | --------- |
| Code          | **1.54×**   | **1.06×**   | 1.00×       | —            | —            | c≥4       |
| Chat          | **1.41×**   | 0.92× ⚠     | 0.94× ⚠     | —            | 1.50×        | c≥4       |
| Summarization | **1.17×**   | 1.00×       | **1.01×**   | —            | —            | c≥4       |

**EAGLE-3 consistently improves throughput on the A100 across the full tested range (c=1 to c=128) — no crossover point was found.** Speedup is task-dependent: code generation benefits most (up to 2.86× at c=1), while chat gains are more modest (1.49× at c=1, likely due to lower draft acceptance on diverse conversational prompts). At high concurrency the advantage narrows but remains positive across all tasks. On L4, summarization approaches parity around c=128 due to tighter VRAM constraints. Mac MLX crossover occurs immediately at c≥4, with catastrophic regressions on chat and code tasks.

---

## Complete Test Matrix

All experiments executed with **3 trials per cell**. Raw results are stored in [`Code/results/raw/`](Code/results/raw/).

### A100 (80GB) — 7 Concurrency Levels × 2 Systems × 3 Tasks × 3 Trials = 126 files

- **Baseline:** 63 files (7 c-levels × 3 tasks × 3 trials)
- **EAGLE-3:** 63 files (7 c-levels × 3 tasks × 3 trials)
- **Concurrency levels tested:** c={1, 4, 8, 16, 32, 64, 128}
- **Tasks:** Chat (ShareGPT, 50 prompts), Code (HumanEval, 50 samples), Summarization (CNN/DailyMail, 50 samples)

### L4 (24GB) — 7 Concurrency Levels × 2 Systems × 3 Tasks × 3 Trials = 126 files

- **Baseline:** 63 files (7 c-levels × 3 tasks × 3 trials)
- **EAGLE-3:** 63 files (7 c-levels × 3 tasks × 3 trials)
- **Concurrency levels tested:** c={1, 4, 8, 16, 32, 64, 128}
- **Tasks:** Chat (ShareGPT, 50 prompts), Code (HumanEval, 50 samples), Summarization (CNN/DailyMail, 50 samples)

### Mac M-Series (Unified Memory) — 5 Concurrency Levels × 2 Systems × 3 Tasks

- **Baseline & MLX Spec:** partial matrix (higher c-levels OOM)
- **Concurrency levels tested:** c={1, 4, 8, 16, 32} (with gaps where memory exhausted)
- **Tasks:** Chat (ShareGPT, 50 prompts), Code (HumanEval, 50 samples), Summarization (CNN/DailyMail, 50 samples)

---

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
