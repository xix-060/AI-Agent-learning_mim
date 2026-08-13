#!/bin/bash
# Ollama 本地部署脚本

echo "🚀 Ollama 部署脚本"

# 1. 检查 Ollama 是否运行
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama 未启动，请先安装并启动"
    exit 1
fi

echo "✅ Ollama 已运行"

# 2. 拉取模型
MODELS=("qwen2.5:0.5b" "qwen2.5:7b")

for model in "${MODELS[@]}"; do
    echo "📦 拉取 $model..."
    ollama pull "$model"
done

echo "✅ 部署完成"
echo "👉 测试：ollama run qwen2.5:0.5b '你好'"
