"""
plot_results.py — Load all JSONL result files and generate publication-quality
plots and CSV summary tables.

Outputs:
  results/plots/  — PNG figures
  results/tables/ — CSV summaries

Usage:
  python3 benchmark/plot_results.py
  python3 benchmark/plot_results.py --results-dir results/raw --out-dir results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONCURRENCIES = [1, 4, 8, 16, 32]
TASKS = ["chat", "code", "summarization"]
TASK_LABELS = {"chat": "Chat (ShareGPT)", "code": "Code (HumanEval)", "summarization": "Summarization (CNN/DM)"}

# Map system names → display labels and colors
SYSTEM_STYLE = {
    "baseline":     {"label": "vLLM Baseline",    "color": "#2196F3", "ls": "-",  "marker": "o"},
    "eagle3":       {"label": "vLLM EAGLE-3",      "color": "#F44336", "ls": "--", "marker": "s"},
    "mlx_baseline": {"label": "MLX Baseline (M4)", "color": "#4CAF50", "ls": "-",  "marker": "^"},
    "mlx_spec":     {"label": "MLX Spec Dec (M4)", "color": "#FF9800", "ls": "--", "marker": "D"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path) -> pd.DataFrame:
    records = []
    for fpath in sorted(results_dir.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not records:
        raise RuntimeError(f"No JSONL result files found in {results_dir}")
    df = pd.DataFrame(records)
    # Coerce numeric columns
    for col in ["ttft_ms", "tpot_ms", "total_latency_ms", "output_tokens", "tokens_per_sec", "acceptance_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per (system, task, concurrency)."""
    metrics = ["ttft_ms", "tpot_ms", "total_latency_ms", "tokens_per_sec", "acceptance_rate"]
    metrics = [m for m in metrics if m in df.columns]
    grp = df.groupby(["system", "gpu_type", "task", "concurrency"])[metrics]
    mean_df = grp.mean().add_suffix("_mean")
    std_df  = grp.std().add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def savefig(fig: plt.Figure, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _systems_in(agg: pd.DataFrame) -> list[str]:
    return [s for s in SYSTEM_STYLE if s in agg["system"].unique()]


def plot_metric_vs_concurrency(agg: pd.DataFrame, metric: str, ylabel: str, title_prefix: str, plots_dir: Path) -> None:
    systems = _systems_in(agg)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), sharey=False)
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, TASKS):
        for sys in systems:
            sub = agg[(agg["system"] == sys) & (agg["task"] == task)].sort_values("concurrency")
            if sub.empty:
                continue
            style = SYSTEM_STYLE[sys]
            mean_col = f"{metric}_mean"
            std_col  = f"{metric}_std"
            y = sub[mean_col].values
            yerr = sub[std_col].fillna(0).values if std_col in sub.columns else None
            ax.errorbar(
                sub["concurrency"].values, y, yerr=yerr,
                label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"],
                linewidth=2, markersize=7, capsize=4,
            )
        ax.set_title(TASK_LABELS.get(task, task), fontsize=11, fontweight="bold")
        ax.set_xlabel("Concurrency", fontsize=10)
        ax.set_ylabel(ylabel if ax == axes[0] else "", fontsize=10)
        ax.set_xticks(CONCURRENCIES)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title_prefix, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    safe_name = metric.replace("/", "_")
    savefig(fig, plots_dir / f"{safe_name}_vs_concurrency.png", title_prefix)


def plot_speedup(agg: pd.DataFrame, plots_dir: Path) -> None:
    """Speedup of spec-dec vs baseline (tokens/sec ratio), per task."""
    pairs = [
        ("baseline",     "eagle3",    "GPU: EAGLE-3 vs Baseline"),
        ("mlx_baseline", "mlx_spec",  "Mac: MLX Spec-Dec vs Baseline"),
    ]
    for base_sys, spec_sys, pair_label in pairs:
        if base_sys not in agg["system"].values or spec_sys not in agg["system"].values:
            continue
        n_tasks = len(TASKS)
        fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), sharey=True)
        if n_tasks == 1:
            axes = [axes]

        for ax, task in zip(axes, TASKS):
            base = agg[(agg["system"] == base_sys) & (agg["task"] == task)].sort_values("concurrency")
            spec = agg[(agg["system"] == spec_sys) & (agg["task"] == task)].sort_values("concurrency")
            merged = base[["concurrency", "tokens_per_sec_mean"]].merge(
                spec[["concurrency", "tokens_per_sec_mean"]], on="concurrency", suffixes=("_base", "_spec")
            )
            if merged.empty:
                continue
            speedup = merged["tokens_per_sec_mean_spec"] / merged["tokens_per_sec_mean_base"]
            color = SYSTEM_STYLE[spec_sys]["color"]
            ax.plot(merged["concurrency"], speedup, color=color, marker="o", linewidth=2, markersize=7)
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="No speedup")
            ax.fill_between(merged["concurrency"], 1.0, speedup, alpha=0.15, color=color)
            ax.set_title(TASK_LABELS.get(task, task), fontsize=11, fontweight="bold")
            ax.set_xlabel("Concurrency", fontsize=10)
            ax.set_ylabel("Speedup (×)" if ax == axes[0] else "", fontsize=10)
            ax.set_xticks(CONCURRENCIES)
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle(f"Speedup — {pair_label}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        safe = pair_label.replace(":", "").replace(" ", "_").replace("/", "_")
        savefig(fig, plots_dir / f"speedup_{safe}.png", pair_label)


def plot_acceptance_rate(agg: pd.DataFrame, plots_dir: Path) -> None:
    spec_systems = [s for s in ["eagle3", "mlx_spec"] if s in agg["system"].values]
    if not spec_systems:
        return
    agg_ar = agg[agg["system"].isin(spec_systems) & agg["acceptance_rate_mean"].notna()]
    if agg_ar.empty:
        return

    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, TASKS):
        for sys in spec_systems:
            sub = agg_ar[(agg_ar["system"] == sys) & (agg_ar["task"] == task)].sort_values("concurrency")
            if sub.empty:
                continue
            style = SYSTEM_STYLE[sys]
            ax.plot(
                sub["concurrency"], sub["acceptance_rate_mean"],
                label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"], linewidth=2, markersize=7,
            )
        ax.set_title(TASK_LABELS.get(task, task), fontsize=11, fontweight="bold")
        ax.set_xlabel("Concurrency", fontsize=10)
        ax.set_ylabel("Acceptance Rate" if ax == axes[0] else "", fontsize=10)
        ax.set_xticks(CONCURRENCIES)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Draft Token Acceptance Rate vs Concurrency", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, plots_dir / "acceptance_rate_vs_concurrency.png", "Acceptance Rate")


def plot_latency_breakdown(agg: pd.DataFrame, plots_dir: Path) -> None:
    """Grouped bar chart: TTFT vs TPOT per system at each concurrency level for each task."""
    systems = _systems_in(agg)
    for task in TASKS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (metric, ylabel) in zip(axes, [("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms)")]):
            x = np.arange(len(CONCURRENCIES))
            width = 0.8 / max(len(systems), 1)
            for i, sys in enumerate(systems):
                sub = agg[(agg["system"] == sys) & (agg["task"] == task)].sort_values("concurrency")
                if sub.empty:
                    continue
                vals = [sub[sub["concurrency"] == c][f"{metric}_mean"].values[0] if not sub[sub["concurrency"] == c].empty else 0 for c in CONCURRENCIES]
                errs = [sub[sub["concurrency"] == c][f"{metric}_std"].values[0] if not sub[sub["concurrency"] == c].empty else 0 for c in CONCURRENCIES]
                style = SYSTEM_STYLE[sys]
                ax.bar(x + i * width - (len(systems) - 1) * width / 2, vals, width * 0.9,
                       yerr=errs, label=style["label"], color=style["color"], capsize=3, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in CONCURRENCIES])
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Latency Breakdown — {TASK_LABELS.get(task, task)}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig(fig, plots_dir / f"latency_breakdown_{task}.png", task)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def save_summary_tables(agg: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Full summary table
    summary = agg[
        ["system", "gpu_type", "task", "concurrency",
         "tokens_per_sec_mean", "tokens_per_sec_std",
         "ttft_ms_mean", "ttft_ms_std",
         "tpot_ms_mean", "tpot_ms_std",
         "acceptance_rate_mean"]
    ].sort_values(["system", "task", "concurrency"])
    summary.columns = [
        "System", "HW", "Task", "Concurrency",
        "TPS_mean", "TPS_std",
        "TTFT_mean_ms", "TTFT_std_ms",
        "TPOT_mean_ms", "TPOT_std_ms",
        "AcceptanceRate",
    ]
    out = tables_dir / "benchmark_summary.csv"
    summary.to_csv(out, index=False, float_format="%.2f")
    print(f"  Saved {out.name}")

    # Speedup table (spec vs baseline, at each concurrency)
    speedup_rows = []
    pairs = [
        ("baseline", "eagle3"),
        ("mlx_baseline", "mlx_spec"),
    ]
    for base_sys, spec_sys in pairs:
        for task in TASKS:
            for conc in CONCURRENCIES:
                base_row = agg[(agg["system"] == base_sys) & (agg["task"] == task) & (agg["concurrency"] == conc)]
                spec_row = agg[(agg["system"] == spec_sys) & (agg["task"] == task) & (agg["concurrency"] == conc)]
                if base_row.empty or spec_row.empty:
                    continue
                bv = base_row["tokens_per_sec_mean"].values[0]
                sv = spec_row["tokens_per_sec_mean"].values[0]
                speedup_rows.append({
                    "Baseline": base_sys,
                    "Speculative": spec_sys,
                    "Task": task,
                    "Concurrency": conc,
                    "Baseline_TPS": round(bv, 2),
                    "Speculative_TPS": round(sv, 2),
                    "Speedup": round(sv / bv, 3) if bv > 0 else None,
                })
    if speedup_rows:
        sp_df = pd.DataFrame(speedup_rows)
        sp_path = tables_dir / "speedup_summary.csv"
        sp_df.to_csv(sp_path, index=False)
        print(f"  Saved {sp_path.name}")


def print_summary(agg: pd.DataFrame) -> None:
    print("\n=== Benchmark Summary ===")
    for sys in _systems_in(agg):
        print(f"\n  [{sys}]")
        for task in TASKS:
            sub = agg[(agg["system"] == sys) & (agg["task"] == task)].sort_values("concurrency")
            if sub.empty:
                continue
            print(f"    {task}:")
            for _, row in sub.iterrows():
                ar = f"  acc={row['acceptance_rate_mean']:.2f}" if pd.notna(row.get("acceptance_rate_mean")) else ""
                print(f"      c={int(row['concurrency']):2d}  "
                      f"tps={row['tokens_per_sec_mean']:6.1f}  "
                      f"ttft={row['ttft_ms_mean']:6.0f}ms  "
                      f"tpot={row['tpot_ms_mean']:5.1f}ms{ar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate plots and tables from benchmark results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    plots_dir  = args.out_dir / "plots"
    tables_dir = args.out_dir / "tables"

    print(f"Loading results from {args.results_dir} ...")
    df = load_all_results(args.results_dir)
    print(f"  Loaded {len(df)} records from {df['system'].nunique()} system(s): {sorted(df['system'].unique())}")

    agg = aggregate(df)

    print(f"\nGenerating plots → {plots_dir}")
    plot_metric_vs_concurrency(agg, "tokens_per_sec", "Tokens / sec", "Throughput vs Concurrency", plots_dir)
    plot_metric_vs_concurrency(agg, "ttft_ms", "TTFT (ms)", "Time to First Token vs Concurrency", plots_dir)
    plot_metric_vs_concurrency(agg, "tpot_ms", "TPOT (ms)", "Time per Output Token vs Concurrency", plots_dir)
    plot_metric_vs_concurrency(agg, "total_latency_ms", "Total Latency (ms)", "End-to-End Latency vs Concurrency", plots_dir)
    plot_speedup(agg, plots_dir)
    plot_acceptance_rate(agg, plots_dir)
    plot_latency_breakdown(agg, plots_dir)

    print(f"\nGenerating tables → {tables_dir}")
    save_summary_tables(agg, tables_dir)

    print_summary(agg)
    print("\nDone!")


if __name__ == "__main__":
    main()
