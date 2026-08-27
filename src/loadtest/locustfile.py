"""Locust 压测：异步 Agent API"""

import random
from locust import HttpUser, task, between


class AgentAPIUser(HttpUser):
    """模拟一个真实用户"""

    wait_time = between(0.5, 1.5)  # 每次请求间隔 0.5-1.5s（模拟人类）
    host = "http://localhost:8080"

    def on_start(self):
        """每个虚拟用户启动时执行一次"""
        self.prompts = [
            "一句话解释 RAG",
            "用 Python 写 hello world",
            "计算 123 乘 456",
            "什么是 KV Cache",
            "总结 Attention 机制",
        ]

    @task(8)
    def chat_short(self):
        """权重 8：短对话（主流流量）"""
        prompt = random.choice(self.prompts)
        with self.client.post(
            "/chat",
            json={"message": prompt, "max_tokens": 64},
            catch_response=True,
            timeout=180,  # CPU 推理慢，必须放宽超时
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status {resp.status_code}")

    @task(2)
    def chat_long(self):
        """权重 2：长生成（重流量）"""
        with self.client.post(
            "/chat",
            json={"message": "写一段 200 字的 AI 简介", "max_tokens": 256},
            catch_response=True,
            timeout=180,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status {resp.status_code}")
