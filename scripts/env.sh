#!/bin/bash
# 一键配置开发环境
set -e echo " 创建 Conda 环境 ai-agent (Python 3.11)..."
conda create -n ai-agent python=3.11 -y
echo " 安装依赖..."
conda run -n ai-agent pip install -r requirements.txt
echo " 环境配置完成！"
echo " 激活环境：conda activate ai-agent"
