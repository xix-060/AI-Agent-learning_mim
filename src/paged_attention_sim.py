"""PagedAttention 内存分配模拟：对比连续分配 vs 分页分配"""

import random


class KVCacheSimulator:
    """KV Cache 分配模拟器"""

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks  # 总物理块数
        self.block_size = block_size  # 每块存多少 token
        self.free_blocks = list(range(total_blocks))

    def alloc_continuous(self, max_seq_len: int) -> list[int] | None:
        """传统：按请求最大长度连续预留"""
        need = -(-max_seq_len // self.block_size)  # 向上取整
        if need > len(self.free_blocks):
            return None
        blocks = self.free_blocks[:need]
        self.free_blocks = self.free_blocks[need:]
        return blocks

    def alloc_paged(self, actual_len: int) -> list[int] | None:
        """Paged：按实际长度按需分配"""
        need = -(-actual_len // self.block_size)
        if need > len(self.free_blocks):
            return None
        blocks = self.free_blocks[:need]
        self.free_blocks = self.free_blocks[need:]
        return blocks


def simulate(num_requests: int = 32, max_len: int = 2048):
    """模拟并发请求的显存利用对比"""
    total_memory = 1024  # 假设 1024 个块
    random.seed(42)

    # 每个请求实际用的长度远小于声明的 max_len（真实场景）
    actual_lens = [random.randint(100, 600) for _ in range(num_requests)]

    # ---- 传统连续分配 ----
    sim = KVCacheSimulator(total_memory)
    served_trad = 0
    used_trad = 0
    for actual in actual_lens:
        blocks = sim.alloc_continuous(max_len)
        if blocks is None:
            break
        served_trad += 1
        used_trad += len(blocks)  # 预留的块全占住

    # ---- Paged 分配 ----
    sim = KVCacheSimulator(total_memory)
    served_paged = 0
    used_paged = 0
    for actual in actual_lens:
        # 模拟请求，按实际长度分配
        blocks = sim.alloc_paged(actual)
        if blocks is None:
            break
        served_paged += 1
        used_paged += len(blocks)

    print(f"模拟：{num_requests} 个请求，声明最大长度 {max_len}，实际长度 100-600")
    print(f"{'':20}{'连续分配':>12}{'PagedAttention':>16}")
    print(f"{'服务请求数':20}{served_trad:>12}{served_paged:>16}")
    print(f"{'占用块数':20}{used_trad:>12}{used_paged:>16}")
    print(
        f"{'碎片浪费(token)':20}{max(0, used_trad*16 - sum(actual_lens[:served_trad])):>12}{used_paged*16 - sum(actual_lens[:served_paged]):>16}"
    )
    print(
        f"\n→ PagedAttention 多服务 {served_paged - served_trad} 个请求（{(served_paged/max(served_trad,1)-1)*100:.0f}% 提升）"
    )


if __name__ == "__main__":
    simulate()
