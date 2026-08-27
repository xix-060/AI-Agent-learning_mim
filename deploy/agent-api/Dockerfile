# agent-api 镜像：复用 src/async_api/main.py 改造的 app.py
FROM python:3.11-slim

WORKDIR /app

# 日志直接刷到 stdout（docker logs 可见）；pip 不留缓存；走清华 PyPI 镜像加速
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 先装依赖再拷代码，充分利用层缓存（requirements 不变时跳过安装）
COPY deploy/agent-api/requirements.txt .
RUN pip install -r requirements.txt

COPY deploy/agent-api/app.py .

EXPOSE 8000

# 容器内由 CMD 启动，不需要 main.py 的 __main__ 块
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
