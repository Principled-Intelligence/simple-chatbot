#!/usr/bin/env bash
set -euo pipefail

export TRITON_CACHE_DIR=~/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --quantization fp8 --enable-auto-tool-choice --tool-call-parser hermes
