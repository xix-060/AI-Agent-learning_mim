# knowledge_agent API 镜像：FastAPI + LangGraph + Chroma（向量库挂载卷）
FROM python:3.11-slim

WORKDIR /app

# 日志刷 stdout；pip 清华源加速；不缓存
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 依赖分层缓存：requirements 不变时跳过安装
COPY knowledge_agent/deploy/ka-api/requirements.txt .
RUN pip install -r requirements.txt

# 应用代码（含 src/、api.py、初始 data/）
COPY knowledge_agent/ /app/knowledge_agent/

EXPOSE 8000

# chroma_db/uploads 通过 compose 卷挂载，容器重建数据不丢
CMD ["uvicorn", "knowledge_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
