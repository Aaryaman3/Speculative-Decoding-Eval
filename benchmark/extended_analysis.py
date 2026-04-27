"""
Extended analysis: speedup ratios, acceptance rate breakdown,
theoretical vs actual speedup, ROUGE quality scores, cost per 1k tokens.

Run from the project root:
    python3 benchmark/extended_analysis.py
"""

import json
import glob
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

RESULTS_DIR  = Path("results/raw")
PLOTS_DIR    = Path("results/plots")
TABLES_DIR   = Path("results/tables")
PROMPTS_DIR  = Path("data/prompts")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"baseline": "#2196F3", "eagle3": "#FF5722"}
TASKS   = ["chat", "code", "summarization"]
GPUS    = ["A100", "L4"]
CONCS   = [1, 4, 8, 16, 32]
NUM_SPEC_TOKENS = 5  # --num_speculative_tokens used in startup_eagle3.sh


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    rows = []
    for f in RESULTS_DIR.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[df["error"].isna()] if "error" in df.columns else df
    df = df[df["ttft_ms"].notna() & df["tokens_per_sec"].notna()]
    return df


# ── Plot 1: Speedup ratio (eagle3 TPS / baseline TPS) ─────────────────────────

def plot_speedup_ratio(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(GPUS), figsize=(13, 5), sharey=True)
    fig.suptitle("EAGLE-3 Speedup Ratio vs Baseline\n(>1 = faster, <1 = slower)",
                 fontsize=13, fontweight="bold")

    for ax, gpu in zip(axes, GPUS):
        sub = df[df["gpu_type"] == gpu]
        for task in TASKS:
            ratios = []
            for c in CONCS:
                b = sub[(sub["system"] == "baseline") & (sub["concurrency"] == c) & (sub["task"] == task)]["tokens_per_sec"]
                e = sub[(sub["system"] == "eagle3")   & (sub["concurrency"] == c) & (sub["task"] == task)]["tokens_per_sec"]
                if len(b) and len(e):
                    ratios.append(e.mean() / b.mean())
                else:
                    ratios.append(np.nan)
            ax.plot(CONCS, ratios, marker="o", linewidth=2, label=task.capitalize())

        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Break-even")
        ax.axhspan(0, 1.0, alpha=0.05, color="red")
        ax.set_title(f"{gpu}", fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_xticks(CONCS)
        ax.set_ylim(0, 1.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Speedup Ratio (EAGLE-3 / Baseline)")
    plt.tight_layout()
    out = PLOTS_DIR / "speedup_ratio.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Plot 2: Acceptance rate by task and concurrency ───────────────────────────

def plot_acceptance_rate(df: pd.DataFrame):
    eagle = df[df["system"] == "eagle3"].copy()
    eagle = eagle[eagle["acceptance_rate"].notna()]

    fig, axes = plt.subplots(1, len(GPUS), figsize=(13, 5), sharey=True)
    fig.suptitle("EAGLE-3 Draft Token Acceptance Rate by Task",
                 fontsize=13, fontweight="bold")

    task_colors = {"chat": "#E91E63", "code": "#9C27B0", "summarization": "#009688"}

    for ax, gpu in zip(axes, GPUS):
        sub = eagle[eagle["gpu_type"] == gpu]
        for task in TASKS:
            t = sub[sub["task"] == task].groupby("concurrency")["acceptance_rate"].mean()
            ax.plot(t.index, t.values, marker="s", linewidth=2,
                    color=task_colors[task], label=task.capitalize())

        ax.set_title(f"{gpu}", fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_xticks(CONCS)
        ax.set_ylim(0, 0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Acceptance Rate")
    plt.tight_layout()
    out = PLOTS_DIR / "acceptance_rate_by_task.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Plot 3: Theoretical vs actual speedup ─────────────────────────────────────

def plot_theoretical_vs_actual(df: pd.DataFrame):
    """
    Theoretical speedup with speculative decoding:
        S = (1 + alpha * k) / (1 + alpha)
    where alpha = acceptance rate, k = num speculative tokens.

    Source: Leviathan et al. (2023) — Speculative Decoding paper.
    """
    eagle = df[(df["system"] == "eagle3") & df["acceptance_rate"].notna()].copy()

    fig, axes = plt.subplots(1, len(GPUS), figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"Theoretical vs Actual Speedup (k={NUM_SPEC_TOKENS} speculative tokens)\n"
        "Theoretical from Leviathan et al. (2023): S = (1 + α·k) / (1 + α)",
        fontsize=11, fontweight="bold"
    )

    for ax, gpu in zip(axes, GPUS):
        sub = eagle[eagle["gpu_type"] == gpu]
        base = df[(df["system"] == "baseline") & (df["gpu_type"] == gpu)]

        for task in TASKS:
            theoretical, actual = [], []
            for c in CONCS:
                e = sub[(sub["task"] == task) & (sub["concurrency"] == c)]
                b = base[(base["task"] == task) & (base["concurrency"] == c)]
                if len(e) and len(b):
                    alpha = e["acceptance_rate"].mean()
                    theo = (1 + alpha * NUM_SPEC_TOKENS) / (1 + alpha)
                    act  = e["tokens_per_sec"].mean() / b["tokens_per_sec"].mean()
                    theoretical.append(theo)
                    actual.append(act)
                else:
                    theoretical.append(np.nan)
                    actual.append(np.nan)

            color = {"chat": "#E91E63", "code": "#9C27B0", "summarization": "#009688"}[task]
            ax.plot(CONCS, theoretical, marker="^", linestyle="--", color=color,
                    alpha=0.6, linewidth=1.5, label=f"{task.capitalize()} (theory)")
            ax.plot(CONCS, actual, marker="o", linestyle="-", color=color,
                    linewidth=2, label=f"{task.capitalize()} (actual)")

        ax.axhline(1.0, color="black", linewidth=1.2, linestyle=":", label="Break-even")
        ax.set_title(f"{gpu}", fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_xticks(CONCS)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))
        ax.set_ylim(0, 2.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel("Speedup Ratio")
    plt.tight_layout()
    out = PLOTS_DIR / "theoretical_vs_actual_speedup.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Plot 4: Cost per 1000 tokens ──────────────────────────────────────────────

def plot_cost_per_1k_tokens(df: pd.DataFrame):
    df2 = df.copy()
    df2 = df2[df2["gpu_cost_usd"].notna() & df2["output_tokens"].notna()]
    df2 = df2[df2["output_tokens"] > 0]
    df2["cost_per_1k"] = df2["gpu_cost_usd"] / df2["output_tokens"] * 1000

    summary = df2.groupby(["gpu_type", "system", "concurrency"])["cost_per_1k"].mean().reset_index()

    fig, axes = plt.subplots(1, len(GPUS), figsize=(13, 5), sharey=False)
    fig.suptitle("GPU Cost per 1,000 Output Tokens (USD)", fontsize=13, fontweight="bold")

    for ax, gpu in zip(axes, GPUS):
        sub = summary[summary["gpu_type"] == gpu]
        for system in ["baseline", "eagle3"]:
            s = sub[sub["system"] == system]
            ax.plot(s["concurrency"], s["cost_per_1k"], marker="o", linewidth=2,
                    color=PALETTE[system], label=system.capitalize())

        ax.set_title(f"{gpu}", fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_xticks(CONCS)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.4f"))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Cost per 1,000 tokens (USD)")
    plt.tight_layout()
    out = PLOTS_DIR / "cost_per_1k_tokens.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── ROUGE quality comparison ──────────────────────────────────────────────────

def compute_rouge_scores():
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge_score not installed, skipping ROUGE computation.")
        return

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Load reference summaries from prompt file
    refs = {}
    prompt_file = PROMPTS_DIR / "summarization_50.jsonl"
    if not prompt_file.exists():
        print("Prompt file not found, skipping ROUGE.")
        return
    with open(prompt_file) as f:
        for line in f:
            r = json.loads(line)
            refs[r["id"]] = r.get("reference_summary", "")

    results = []
    for system in ["baseline", "eagle3"]:
        path = RESULTS_DIR / f"{system}_A100_summarization_c01_t99.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                prompt_id = r["request_id"].split("_c")[0]
                ref = refs.get(prompt_id, "")
                hyp = r.get("output_text", "")
                if ref and hyp:
                    scores = scorer.score(ref, hyp)
                    results.append({
                        "system": system,
                        "prompt_id": prompt_id,
                        "rouge1": scores["rouge1"].fmeasure,
                        "rouge2": scores["rouge2"].fmeasure,
                        "rougeL": scores["rougeL"].fmeasure,
                    })

    if not results:
        print("No ROUGE results computed.")
        return

    rouge_df = pd.DataFrame(results)
    summary = rouge_df.groupby("system")[["rouge1", "rouge2", "rougeL"]].mean().round(4)
    print("\nROUGE Scores (quality_tiny, A100, c=1, trial=99):")
    print(summary.to_string())
    summary.to_csv(TABLES_DIR / "rouge_scores.csv")
    print(f"Saved results/tables/rouge_scores.csv")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    metrics = ["rouge1", "rouge2", "rougeL"]
    labels  = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    width = 0.3
    for i, (system, color) in enumerate([("baseline", PALETTE["baseline"]), ("eagle3", PALETTE["eagle3"])]):
        row = summary.loc[system] if system in summary.index else pd.Series([0,0,0], index=metrics)
        ax.bar(x + i*width, [row[m] for m in metrics], width, label=system.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 Score")
    ax.set_title("Output Quality: ROUGE Scores\nBaseline vs EAGLE-3 (Summarization, A100)")
    ax.legend()
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "rouge_quality_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    return rouge_df


# ── Summary stats table ───────────────────────────────────────────────────────

def generate_extended_summary(df: pd.DataFrame):
    eagle = df[df["system"] == "eagle3"].copy()
    base  = df[df["system"] == "baseline"].copy()

    rows = []
    for gpu in GPUS:
        for task in TASKS:
            for c in CONCS:
                b = base[(base["gpu_type"]==gpu)&(base["task"]==task)&(base["concurrency"]==c)]
                e = eagle[(eagle["gpu_type"]==gpu)&(eagle["task"]==task)&(eagle["concurrency"]==c)]
                if len(b) == 0 or len(e) == 0:
                    continue
                alpha = e["acceptance_rate"].mean() if e["acceptance_rate"].notna().any() else np.nan
                theo  = (1 + alpha*NUM_SPEC_TOKENS)/(1+alpha) if not np.isnan(alpha) else np.nan
                actual= e["tokens_per_sec"].mean() / b["tokens_per_sec"].mean()
                overhead_pct = (1 - actual/theo)*100 if theo else np.nan
                rows.append({
                    "gpu": gpu, "task": task, "concurrency": c,
                    "baseline_tps": round(b["tokens_per_sec"].mean(), 2),
                    "eagle3_tps":   round(e["tokens_per_sec"].mean(), 2),
                    "speedup_actual": round(actual, 3),
                    "speedup_theoretical": round(theo, 3) if not np.isnan(theo) else "",
                    "acceptance_rate": round(alpha, 3) if not np.isnan(alpha) else "",
                    "overhead_vs_theory_pct": round(overhead_pct, 1) if not np.isnan(overhead_pct) else "",
                })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "extended_summary.csv", index=False)
    print(f"Saved results/tables/extended_summary.csv")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records.\n")

    print("Generating speedup ratio plot...")
    plot_speedup_ratio(df)

    print("Generating acceptance rate by task plot...")
    plot_acceptance_rate(df)

    print("Generating theoretical vs actual speedup plot...")
    plot_theoretical_vs_actual(df)

    print("Generating cost per 1k tokens plot...")
    plot_cost_per_1k_tokens(df)

    print("Computing ROUGE scores...")
    compute_rouge_scores()

    print("Generating extended summary table...")
    extended = generate_extended_summary(df)

    print("\n── Extended Summary (sample) ──")
    print(extended[extended["gpu"] == "A100"].to_string(index=False))

    print("\nDone. New outputs in results/plots/ and results/tables/")
