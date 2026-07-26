**#** ⾼级 **RAG** 技术

**## 1. Query** 改写（**Query Rewriting**）

原始问题可能表述不清或缺少关键词，⽤ LLM 改写成更适合检索的版本。

⽰例：

\- 原始："他发明的"

\- 改写："Python 编程语⾔是谁发明的？"

**## 2. HyDE**（**Hypothetical Document Embeddings**）

先让 LLM ⽣成⼀个"假设性答案"，⽤这个答案的向量去检索（⽽不是⽤问题向量）。

原理：答案和⽂档的表述更接近，⽐问题和⽂档的相似度更⾼。

⽰例：

\- 问题："谁发明了 Python？"

\- HyDE ⽣成假设答案："Python 是由 Guido van Rossum 在 1991 年创建的..."

\- ⽤假设答案向量检索 → 命中率更⾼

**## 3. Multi-Query**

让 LLM ⽣成多个不同⻆度的查询，分别检索后合并结果。

适⽤：复杂问题需要多⽅⾯信息。

**## 4. Reranker**（重排序）

先⽤ Embedding 快速召回 Top-20，再⽤ Cross-Encoder 精排 Top-5。

原理：Cross-Encoder（把 query 和 doc 拼接做分类）⽐ Bi-Encoder（分别编码再算相似度）更准确，但更慢。

常⽤：bge-reranker、Cohere Reranker

**## 5.** 优化策略对应的问题

\| 问题 | 解决⽅案 |

\|---|---|

\| 检索不到相关⽂档 | Query 改写 / HyDE / Multi-Query |

\| 检索到但不精确 | Reranker |

\| ⽂档太⻓ | 更好的切块策略 |

\| 答案不忠实上下⽂ | 更好的 Prompt / 限制 LLM 只⽤上下⽂ |
