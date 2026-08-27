#!/usr/bin/env bash
# 生产环境 LLM 服务：vLLM 拉起 Qwen2.5-7B-Instruct（需宿主机 GPU）
# 用法: bash scripts/start_vllm.sh
# 验证: curl http://127.0.0.1:8000/v1/models

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[vLLM] 启动模型: ${MODEL} (端口 ${PORT})"
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.85
