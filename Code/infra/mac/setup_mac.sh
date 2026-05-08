#!/bin/bash
# One-time setup for Mac/MLX environment
set -e

echo "=== Setting up Project 2 Mac/MLX Environment ==="

# Install Python dependencies
pip3 install --upgrade pip
pip3 install mlx-lm datasets aiohttp tqdm numpy pandas matplotlib

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. python3 data/prepare_datasets.py --n-samples 50"
echo "  2. bash infra/startup_mlx_baseline.sh"
echo "  3. bash benchmark/run_sweep.sh mlx_baseline mac http://localhost:8000"
