#!/bin/bash
# 启动 OpenAI 兼容推理后端
# ----------------------------------------------------------------
# 【真 vLLM 命令】需 Linux/WSL2 + NVIDIA GPU + pip install vllm：
#   export HF_ENDPOINT=https://hf-mirror.com
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-7B-Instruct \
#       --served-model-name qwen2.5-7b \
#       --max-model-len 4096 \
#       --gpu-memory-utilization 0.90 \
#       --port 8000 \
#       --enable-prefix-caching
# ----------------------------------------------------------------
# 【本机纯 CPU】vLLM 不支持 Windows 原生 + 需 GPU，改用 Ollama 作为
# OpenAI 兼容后端（API 形态完全一致，仅性能不同）。Ollama 已常驻运行，
# 本脚本只做环境自检 + 模型就绪确认，不重复造常驻服务。

set -e
MODEL="qwen2.5:0.5b"
BASE="http://localhost:11434"

echo "🚀 推理后端就绪检查（CPU + Ollama 替代 vLLM）"

# 1. 检查 Ollama 服务
if ! curl -s --max-time 3 "$BASE/api/tags" >/dev/null 2>&1; then
    echo "❌ Ollama 未运行，请先启动：ollama serve"
    exit 1
fi
echo "✅ Ollama 服务已运行"

# 2. 检查模型
if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "📦 模型未拉取，执行：ollama pull $MODEL"
    ollama pull "$MODEL"
fi
echo "✅ 模型就绪：$MODEL"

# 3. 提示 OpenAI 兼容端点
echo ""
echo "👉 OpenAI 兼容端点：$BASE/v1"
echo "   客户端配置：base_url=$BASE/v1  model=$MODEL  api_key=任意非空"
echo "   验证：curl $BASE/v1/models"
