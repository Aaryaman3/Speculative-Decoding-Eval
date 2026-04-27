# Speculative Decoding Eval — EAGLE-3 Benchmark

Benchmarking EAGLE-3 speculative decoding against greedy baseline using `unsloth/Meta-Llama-3.1-8B-Instruct` on NVIDIA A100 (80GB) and L4 (24GB) GPUs via vLLM.

---

## Repository Structure

```
benchmark/         Load tester, sweep runner, plot generator, quality checker
data/prompts/      50 prompts each for chat, code, summarization (seed=42)
infra/             Server startup scripts and setup for GCP + Mac
dashboard/         Streamlit live demo app
results/raw/       182 JSONL result files (A100 + L4, baseline + eagle3 + quality)
results/plots/     10 academic-grade PNG plots
results/tables/    3 CSV summary tables
```

---

## Hardware

| GPU | VRAM | Role |
|-----|------|------|
| NVIDIA A100-SXM4-80GB | 80 GB | Aaryaman — both servers fit simultaneously |
| NVIDIA L4 | 24 GB | Himanshu — servers run sequentially |

---

## Models

| Role | Model |
|------|-------|
| Target | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| Draft (EAGLE-3) | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |

---

## Benchmark Design

- **Tasks:** Chat (ShareGPT), Code (HumanEval), Summarization (CNN/DailyMail)
- **Concurrency levels:** 1, 4, 8, 16, 32
- **Trials per cell:** 3
- **Max output tokens:** 256
- **Temperature:** 0 (greedy, deterministic)
- **Total result files:** 45 baseline + 45 eagle3 per GPU = 180 sweep files + 2 quality files

---

## Key Findings

### Throughput (tokens/sec)

| GPU | System | c=1 | c=8 | c=32 |
|-----|--------|-----|-----|------|
| A100 | Baseline | ~89 | ~88 | ~70 |
| A100 | EAGLE-3 | ~65 | ~53 | ~32 |
| L4 | Baseline | ~17 | ~16 | ~13 |
| L4 | EAGLE-3 | ~13 | ~11 | ~7 |

**EAGLE-3 did not improve throughput on either GPU.** The A100 is compute-fast enough that draft model overhead outweighs acceptance-rate gains. On the L4, VRAM pressure from the draft model compounds latency.

### EAGLE-3 Acceptance Rate (~50%)
The draft model accepted ~50% of speculative tokens across all tasks and concurrency levels — theoretically sufficient for speedup, but batching overhead at high concurrency negates the benefit.

### Quality Check (A100, summarization, c=1, trial=99)
Both baseline and EAGLE-3 produced **identical output text**, confirming speculative decoding preserves output distribution at temperature=0.

### A100 vs L4
- A100 is **~5.5× faster** in throughput across all conditions
- L4 degrades more severely at high concurrency under EAGLE-3 (TPOT reaches 128ms/token at c=32)

---

## Reproducing the Benchmark

### 1. Setup (GCP GPU instance)
```bash
export HF_TOKEN=hf_xxxx
bash infra/setup_gcp.sh
```

### 2. Prepare datasets
```bash
python3 data/prepare_datasets.py --n-samples 50
```

### 3. Run full experiment
```bash
bash benchmark/run_experiment.sh A100   # or L4
```

### 4. Generate plots
```bash
python3 benchmark/plot_results.py
```

---

## Results

Plots are in [`results/plots/`](results/plots/) and summary tables in [`results/tables/`](results/tables/).
