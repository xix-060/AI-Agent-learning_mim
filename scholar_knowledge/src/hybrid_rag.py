"""混合 GraphRAG：向量检索 + 图谱检索"""

import sys
import time
from pathlib import Path

# 项目记忆：跨目录 import 需 sys.path 配置 + # noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # 项目根，使 src.* 可导入
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # scholar_knowledge/src，使 graph_builder 可导入

import chromadb  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.models import Message, RoleEnum  # noqa: E402
from graph_builder import ScholarGraph  # noqa: E402
from graph_rag import GraphRAG  # noqa: E402

# DashScope embedding 重试策略：3 次指数退避（1s/2s/4s）
_EMBED_MAX_RETRIES = 3
_EMBED_RETRY_DELAYS = (1.0, 2.0, 4.0)


def _embed_with_retry(embedder: Embedder, texts: str | list[str]):
    """调用 embedder.embed，失败按指数退避重试，最后仍失败抛异常。

    解决 DashScope 偶发 "Connection errorr." 导致整次 benchmark 向量 0% 的问题。
    """
    last_exc: Exception | None = None
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            return embedder.embed(texts)
        except Exception as e:
            last_exc = e
            if attempt < _EMBED_MAX_RETRIES - 1:
                delay = _EMBED_RETRY_DELAYS[attempt]
                print(
                    f"  ⏳ embed 失败（第 {attempt + 1}/{_EMBED_MAX_RETRIES} 次，{delay:.0f}s 后重试）: {e}"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"embed 重试 {_EMBED_MAX_RETRIES} 次仍失败：{last_exc}"
    ) from last_exc


class HybridRAG:
    """向量 + 图谱 混合检索问答"""

    def __init__(self, graph: ScholarGraph, llm: LLMClient, embedder: Embedder):
        self.graph = graph
        self.llm = llm
        self.embedder = embedder
        self.graph_rag = GraphRAG(graph, llm)
        self._build_vector_index()

    def _build_vector_index(self):
        """构建论文文本的向量索引。

        DashScope embedding API 限制 batch ≤ 10，超过会 400 InvalidParameter，
        因此对论文文本分批 embed 后再合并入 chromadb。
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("scholar_papers")

        texts = list(self.graph.data["texts"].values())
        ids = list(self.graph.data["texts"].keys())
        if not texts:
            print("⚠️ 无论文文本可索引")
            return

        batch_size = 10  # DashScope embedding API 限制
        all_vectors: list[list[float]] = []
        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vec = _embed_with_retry(self.embedder, batch).tolist()
                all_vectors.extend(vec)
            self.collection.add(ids=ids, embeddings=all_vectors, documents=texts)
            print(
                f"✅ 向量索引构建完成：{len(ids)} 篇论文（分 {total_batches} 批 embed）"
            )
        except Exception as e:
            print(f"⚠️ 向量索引构建失败：{e}")
            self.collection = None

    # ===== 向量检索 =====

    def vector_search(self, query: str, top_k: int = 3) -> list[str]:
        """向量语义检索，返回最相似的 top_k 篇论文文本。

        若向量索引未构建（`self.collection is None`）返回空列表；
        调用方可通过 `self.has_vector_index` 属性判断是"无索引"还是"检索真无命中"。
        """
        if not self.collection:
            return []
        try:
            query_vec = _embed_with_retry(self.embedder, query).tolist()
            results = self.collection.query(
                query_embeddings=[query_vec], n_results=top_k
            )
            return results["documents"][0]
        except Exception as e:
            print(f"⚠️ 向量检索失败：{e}")
            return []

    @property
    def has_vector_index(self) -> bool:
        """向量索引是否成功构建（benchmark 可据此跳过向量评分，避免"无索引=0%"误导）。"""
        return self.collection is not None

    # ===== 混合检索 =====

    def hybrid_search(self, query: str) -> tuple[list[str], list[str]]:
        """混合检索：返回 (图谱证据, 向量证据)"""
        graph_evidence = self.graph_rag.retrieve_graph_evidence(query)
        vector_evidence = self.vector_search(query)
        return graph_evidence, vector_evidence

    def query(self, question: str, verbose: bool = False) -> str:
        """混合问答"""
        # 结构化优先
        structural = self.graph_rag.answer_structural(question)
        if structural:
            return structural

        # 混合检索
        graph_ev, vector_ev = self.hybrid_search(question)
        if verbose:
            print(f"  [图谱证据 {len(graph_ev)} 条 + 向量证据 {len(vector_ev)} 条]")

        prompt = f"""你是学术知识图谱问答助手。基于以下两类证据回答：
1. 图谱结构化证据（关系路径）
2. 论文文本证据（语义内容）

【图谱证据】
{chr(10).join(graph_ev) if graph_ev else "（无）"}

【论文文本证据】
{chr(10).join(f"- {t}" for t in vector_ev) if vector_ev else "（无）"}

【问题】
{question}

请综合回答，并说明依据了哪类证据。"""

        # simple_chat 不支持 temperature 参数，改用 chat() 直接构造 messages
        messages = [
            Message(
                role=RoleEnum.SYSTEM,
                content="你是学术知识图谱问答助手，基于图谱+向量证据回答。",
            ),
            Message(role=RoleEnum.USER, content=prompt),
        ]
        try:
            return self.llm.chat(messages, temperature=0.2).content
        except Exception as e:
            return f"⚠️ LLM 调用失败：{e}\n图谱证据（前 3 条）：\n" + "\n".join(
                graph_ev[:3]
            )

    # ===== 评测对比 =====

    def benchmark(self, questions: list[str], expected_kws: list[list[str]]) -> dict:
        """对比三种检索方式命中率。

        当向量索引未构建（通常是断网导致 embedding API 不可达）时，
        纯向量分支会被**跳过**（不算分），避免把"无索引"误报为"0% 命中"。

        Args:
            questions: 评测问题列表
            expected_kws: 每题期望命中的关键词列表（用于简单命中评测）

        Returns:
            三种方式命中率汇总 dict（缺省值不存在于结果中）
        """
        results: dict[str, list[float]] = {"graph": [], "hybrid": []}
        vector_enabled = self.has_vector_index
        if not vector_enabled:
            print(
                "\nℹ️ 向量索引未构建（可能是网络不可达/embedding API 断连），跳过纯向量评分。"
            )
            results["vector"] = []  # 占位，汇总时过滤
        else:
            results["vector"] = []

        for idx, (q, expected) in enumerate(zip(questions, expected_kws), 1):
            if not expected:
                continue
            print(f"\n  [{idx}/{len(questions)}] {q}")

            # 纯向量：索引未构建则跳过（防止"无证据空 prompt → LLM 乱答 → 0%"假象）
            vec_hit_rate = 0.0
            if vector_enabled:
                vec_ev = self.vector_search(q)
                vec_prompt = f"基于以下文本回答：\n{chr(10).join(vec_ev)}\n问题：{q}"
                vec_messages = [
                    Message(role=RoleEnum.SYSTEM, content="基于给定文本回答。"),
                    Message(role=RoleEnum.USER, content=vec_prompt),
                ]
                try:
                    vec_answer = self.llm.chat(vec_messages, temperature=0.2).content
                except Exception as e:
                    vec_answer = f"LLM 调用失败：{e}"
                vec_hit_rate = sum(1 for k in expected if k in vec_answer) / len(
                    expected
                )
                results["vector"].append(vec_hit_rate)
                print(
                    f"    向量: 召回 {len(vec_ev)} 段 → 命中 {sum(1 for k in expected if k in vec_answer)}/{len(expected)}  {vec_hit_rate:.0%}"
                )

            # 纯图谱：通过 graph_rag 走结构化+图谱证据流程
            graph_answer = self.graph_rag.query(q)
            graph_hit_rate = sum(1 for k in expected if k in graph_answer) / len(
                expected
            )
            results["graph"].append(graph_hit_rate)
            print(
                f"    图谱: 命中 {sum(1 for k in expected if k in graph_answer)}/{len(expected)}  {graph_hit_rate:.0%}"
            )

            # 混合：图谱 + 向量
            hybrid_answer = self.query(q)
            hybrid_hit_rate = sum(1 for k in expected if k in hybrid_answer) / len(
                expected
            )
            results["hybrid"].append(hybrid_hit_rate)
            print(
                f"    混合: 命中 {sum(1 for k in expected if k in hybrid_answer)}/{len(expected)}  {hybrid_hit_rate:.0%}"
            )

        summary: dict[str, float] = {}
        if results.get("vector"):
            summary["vector"] = sum(results["vector"]) / len(results["vector"])
        if results["graph"]:
            summary["graph"] = sum(results["graph"]) / len(results["graph"])
        if results["hybrid"]:
            summary["hybrid"] = sum(results["hybrid"]) / len(results["hybrid"])

        print("\n📊 三种检索命中率对比：")
        for k in ("vector", "graph", "hybrid"):
            if k in summary:
                print(f"  {k:8} {summary[k]:.1%}")
            else:
                print(
                    f"  {k:8} 跳过（{('索引未构建' if k == 'vector' else '无有效样本')}）"
                )
        return summary


if __name__ == "__main__":
    graph = ScholarGraph()
    llm = LLMClient()
    embedder = Embedder()
    rag = HybridRAG(graph, llm, embedder)

    questions = [
        "RAG 相关的论文有哪些？",
        "Agent 领域研究涉及哪些关键词？",
        "Transformer 架构有什么特点？",
    ]
    expected = [
        ["Retrieval", "RAG", "Lewis"],
        ["Agent", "ReAct", "CoT"],
        ["Attention", "Transformer", "Vaswani"],
    ]
    rag.benchmark(questions, expected)
