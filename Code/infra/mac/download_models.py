"""Pre-download models to HuggingFace cache so servers start instantly."""
from mlx_lm import load

MODELS = [
    ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "8B target (baseline + speculative)"),
    ("mlx-community/Llama-3.2-1B-Instruct-4bit",      "1B draft (speculative only)"),
]

for model_id, desc in MODELS:
    print(f"\nDownloading {desc}: {model_id}")
    model, tokenizer = load(model_id)
    print(f"  Done — {model_id}")
    del model, tokenizer

print("\nAll models cached. Servers will start in seconds.")
