"""对比实验：同步逐个 vs 异步并发的差距"""

import time
import httpx
import asyncio

API = "http://localhost:8080"


def sync_way(n: int = 10):
    """同步：逐个请求"""
    with httpx.Client(timeout=120) as c:
        start = time.time()
        for i in range(n):
            c.post(
                f"{API}/chat", json={"message": f"一句话介绍数字 {i}", "max_tokens": 32}
            )
        return time.time() - start


async def async_way(n: int = 10):
    """异步：并发请求"""
    async with httpx.AsyncClient(timeout=120) as c:
        start = time.time()
        await asyncio.gather(
            *[
                c.post(
                    f"{API}/chat",
                    json={"message": f"一句话介绍数字 {i}", "max_tokens": 32},
                )
                for i in range(n)
            ]
        )
        return time.time() - start


if __name__ == "__main__":
    n = 10
    t1 = sync_way(n)
    t2 = asyncio.run(async_way(n))
    print(f"{n} 个请求：")
    print(f"  同步逐个: {t1:.1f}s")
    print(f"  异步并发: {t2:.1f}s")
    print(f"  提速: {t1 / t2:.1f}x")
