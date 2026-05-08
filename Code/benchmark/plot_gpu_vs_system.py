"""
plot_gpu_vs_system.py

PURPOSE:
Create academic-grade comparison plots showing:
- Mac, A100, and L4 GPUs (rows)
- Baseline vs Speculative Decoding comparison (columns/hue)
- Side-by-side comparison for each task

OUTPUTS:
Generates 3x3 or 2x3 grid plots comparing all GPU types with both baseline and spec systems.
"""

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(results_dir, baremetal_dir=None):
    """Load data from cloud results and optionally baremetal results"""
    records = []
    
    # Load cloud results (results/raw)
    files = glob.glob(os.path.join(results_dir, "raw", "*.jsonl"))
    for f in files:
        if "_t99.jsonl" in f:
            continue
        with open(f, 'r') as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
    
    # Load baremetal results (mac-baremetal/results/raw and synthetic_raw)
    if baremetal_dir:
        baremetal_raw = glob.glob(os.path.join(baremetal_dir, "raw", "*.jsonl"))
        for f in baremetal_raw:
            if "_t99.jsonl" in f:
                continue
            with open(f, 'r') as file:
                for line in file:
                    if line.strip():
                        records.append(json.loads(line))
        
        baremetal_synthetic = glob.glob(os.path.join(baremetal_dir, "synthetic_raw", "*.jsonl"))
        for f in baremetal_synthetic:
            if "_t99.jsonl" in f:
                continue
            with open(f, 'r') as file:
                for line in file:
                    if line.strip():
                        records.append(json.loads(line))
    
    return pd.DataFrame(records)


def normalize_systems(df):
    """Normalize system names: map all variants to 'baseline' or 'speculative'"""
    df = df.copy()
    df['system_type'] = df['system'].apply(
        lambda x: 'baseline' if 'baseline' in x.lower() else 'speculative'
    )
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    baremetal_dir = os.path.join(base_dir, "mac-baremetal", "results")
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_data(results_dir, baremetal_dir)
    if df.empty:
        print("No valid benchmark data found.")
        return
    
    # Normalize system types
    df = normalize_systems(df)
    
    print(f"Loaded {len(df)} requests.")
    print(f"GPU types: {sorted(df['gpu_type'].unique())}")
    print(f"System types: {sorted(df['system_type'].unique())}")
    
    # ---------------------------------------------------------
    # Academic Formatting Setup
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")
    palette = {"baseline": "#1f77b4", "speculative": "#ff7f0e"}
    
    gpu_types = sorted(df['gpu_type'].unique())
    tasks = sorted(df['task'].unique())
    
    # ====================================================================
    # PLOT 1: Throughput - GPU x Baseline/Spec Grid (3 rows x 3 cols)
    # ====================================================================
    print("\nGenerating Throughput Comparison (3 GPUs x 3 Tasks)...")
    fig, axes = plt.subplots(len(gpu_types), len(tasks), figsize=(18, 14), sharey=False)
    
    for gpu_idx, gpu in enumerate(gpu_types):
        for task_idx, task in enumerate(tasks):
            ax = axes[gpu_idx, task_idx]
            task_gpu_df = df[(df['gpu_type'] == gpu) & (df['task'] == task)]
            
            if not task_gpu_df.empty:
                sns.lineplot(
                    data=task_gpu_df, x='concurrency', y='tokens_per_sec', 
                    hue='system_type', marker='o', errorbar=('ci', 95), 
                    ax=ax, palette=palette, linewidth=2.5
                )
                ax.set_xticks(sorted(task_gpu_df['concurrency'].unique()))
            
            # Labels
            if task_idx == 0:
                ax.set_ylabel(f"{gpu}\nThroughput (Tokens/Sec)", fontweight='bold')
            else:
                ax.set_ylabel("")
            
            if gpu_idx == 0:
                ax.set_title(f"Task: {task.capitalize()}", fontweight='bold')
            
            if gpu_idx < len(gpu_types) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Concurrency")
            
            # Remove duplicate legends
            if gpu_idx < len(gpu_types) - 1 or task_idx < len(tasks) - 1:
                if ax.get_legend():
                    ax.get_legend().remove()
    
    plt.suptitle("Throughput Comparison: Baseline vs Speculative Decoding (All GPUs & Tasks)", 
                 fontsize=20, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gpu_baseline_vs_spec_throughput.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: gpu_baseline_vs_spec_throughput.png")
    
    # ====================================================================
    # PLOT 2: TTFT - GPU x Baseline/Spec Grid (3 rows x 3 cols)
    # ====================================================================
    print("Generating TTFT Comparison (3 GPUs x 3 Tasks)...")
    fig, axes = plt.subplots(len(gpu_types), len(tasks), figsize=(18, 14), sharey=False)
    
    for gpu_idx, gpu in enumerate(gpu_types):
        for task_idx, task in enumerate(tasks):
            ax = axes[gpu_idx, task_idx]
            task_gpu_df = df[(df['gpu_type'] == gpu) & (df['task'] == task)]
            
            if not task_gpu_df.empty:
                sns.lineplot(
                    data=task_gpu_df, x='concurrency', y='ttft_ms', 
                    hue='system_type', marker='s', errorbar=('ci', 95), 
                    ax=ax, palette=palette, linewidth=2.5
                )
                ax.set_xticks(sorted(task_gpu_df['concurrency'].unique()))
            
            if task_idx == 0:
                ax.set_ylabel(f"{gpu}\nTTFT (ms)", fontweight='bold')
            else:
                ax.set_ylabel("")
            
            if gpu_idx == 0:
                ax.set_title(f"Task: {task.capitalize()}", fontweight='bold')
            
            if gpu_idx < len(gpu_types) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Concurrency")
            
            if gpu_idx < len(gpu_types) - 1 or task_idx < len(tasks) - 1:
                if ax.get_legend():
                    ax.get_legend().remove()
    
    plt.suptitle("Time to First Token (TTFT) Comparison: Baseline vs Speculative Decoding", 
                 fontsize=20, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gpu_baseline_vs_spec_ttft.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: gpu_baseline_vs_spec_ttft.png")
    
    # ====================================================================
    # PLOT 3: TPOT - GPU x Baseline/Spec Grid (3 rows x 3 cols)
    # ====================================================================
    print("Generating TPOT Comparison (3 GPUs x 3 Tasks)...")
    fig, axes = plt.subplots(len(gpu_types), len(tasks), figsize=(18, 14), sharey=False)
    
    for gpu_idx, gpu in enumerate(gpu_types):
        for task_idx, task in enumerate(tasks):
            ax = axes[gpu_idx, task_idx]
            task_gpu_df = df[(df['gpu_type'] == gpu) & (df['task'] == task)]
            
            if not task_gpu_df.empty:
                sns.lineplot(
                    data=task_gpu_df, x='concurrency', y='tpot_ms', 
                    hue='system_type', marker='^', errorbar=('ci', 95), 
                    ax=ax, palette=palette, linewidth=2.5
                )
                ax.set_xticks(sorted(task_gpu_df['concurrency'].unique()))
            
            if task_idx == 0:
                ax.set_ylabel(f"{gpu}\nTPOT (ms)", fontweight='bold')
            else:
                ax.set_ylabel("")
            
            if gpu_idx == 0:
                ax.set_title(f"Task: {task.capitalize()}", fontweight='bold')
            
            if gpu_idx < len(gpu_types) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Concurrency")
            
            if gpu_idx < len(gpu_types) - 1 or task_idx < len(tasks) - 1:
                if ax.get_legend():
                    ax.get_legend().remove()
    
    plt.suptitle("Time Per Output Token (TPOT) Comparison: Baseline vs Speculative Decoding", 
                 fontsize=20, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gpu_baseline_vs_spec_tpot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: gpu_baseline_vs_spec_tpot.png")
    
    # ====================================================================
    # PLOT 4: Cost - GPU x Baseline/Spec Grid (3 rows x 3 cols)
    # ====================================================================
    print("Generating Cost Comparison (3 GPUs x 3 Tasks)...")
    df_cost = df.copy()
    df_cost['cost_per_1m_tokens'] = (df_cost['gpu_cost_usd'] / df_cost['output_tokens']) * 1_000_000
    
    fig, axes = plt.subplots(len(gpu_types), len(tasks), figsize=(18, 14), sharey=False)
    
    for gpu_idx, gpu in enumerate(gpu_types):
        for task_idx, task in enumerate(tasks):
            ax = axes[gpu_idx, task_idx]
            task_gpu_df = df_cost[(df_cost['gpu_type'] == gpu) & (df_cost['task'] == task)]
            
            if not task_gpu_df.empty:
                sns.barplot(
                    data=task_gpu_df, x='concurrency', y='cost_per_1m_tokens', 
                    hue='system_type', ax=ax, palette=palette, errorbar=None, alpha=0.8
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            
            if task_idx == 0:
                ax.set_ylabel(f"{gpu}\nCost per 1M Tokens ($)", fontweight='bold')
            else:
                ax.set_ylabel("")
            
            if gpu_idx == 0:
                ax.set_title(f"Task: {task.capitalize()}", fontweight='bold')
            
            if gpu_idx < len(gpu_types) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Concurrency")
            
            if gpu_idx < len(gpu_types) - 1 or task_idx < len(tasks) - 1:
                if ax.get_legend():
                    ax.get_legend().remove()
    
    plt.suptitle("Cost per 1M Tokens: Baseline vs Speculative Decoding", 
                 fontsize=20, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "gpu_baseline_vs_spec_cost.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: gpu_baseline_vs_spec_cost.png")
    
    # ====================================================================
    # PLOT 5: Summary Speedup Grid (GPU x Task)
    # ====================================================================
    print("Generating Speedup Summary...")
    
    # Calculate speedup (baseline / speculative) for throughput
    speedup_data = []
    for gpu in gpu_types:
        for task in tasks:
            baseline_df = df[(df['gpu_type'] == gpu) & (df['task'] == task) & (df['system_type'] == 'baseline')]
            spec_df = df[(df['gpu_type'] == gpu) & (df['task'] == task) & (df['system_type'] == 'speculative')]
            
            if not baseline_df.empty and not spec_df.empty:
                baseline_tps = baseline_df['tokens_per_sec'].mean()
                spec_tps = spec_df['tokens_per_sec'].mean()
                speedup = spec_tps / baseline_tps if baseline_tps > 0 else 1.0
                speedup_data.append({
                    'gpu': gpu,
                    'task': task,
                    'speedup': speedup
                })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        speedup_pivot = speedup_df.pivot(index='gpu', columns='task', values='speedup')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(speedup_pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0, 
                   cbar_kws={'label': 'Speedup (Spec / Baseline)'}, ax=ax, linewidths=2)
        ax.set_title("Speculative Decoding Speedup (Spec / Baseline)\nHigher is Better", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Task", fontweight='bold')
        ax.set_ylabel("GPU", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gpu_speedup_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: gpu_speedup_heatmap.png")
    
    print("\n✅ All GPU comparison plots generated successfully!")
    print(f"📁 Saved to: {plots_dir}/")
    print("\nGenerated files:")
    print("  - gpu_baseline_vs_spec_throughput.png (3x3 grid)")
    print("  - gpu_baseline_vs_spec_ttft.png (3x3 grid)")
    print("  - gpu_baseline_vs_spec_tpot.png (3x3 grid)")
    print("  - gpu_baseline_vs_spec_cost.png (3x3 grid)")
    print("  - gpu_speedup_heatmap.png (speedup summary)")


if __name__ == "__main__":
    main()
