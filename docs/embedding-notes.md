**# Embedding** 原理

**## 1.** 什么是 **Embedding**？

把⽂本转换成固定维度的向量（如 1536 维），使语义相近的⽂本在向量空间中距离相近。

**## 2.** 为什么需要 **Embedding**？

\- 计算机不理解⽂字，只理解数字

\- 向量化后可以计算语义相似度（余弦相似度）

\- 这是 RAG 检索的基础

**## 3.** 常⽤ **Embedding** 模型

\| 模型 | 维度 | 特点 | 来源 |

\|---|---|---|---|

\| text-embedding-3-small | 1536 | 便宜，效果好 | OpenAI |

\| text-embedding-3-large | 3072 | 效果最好 | OpenAI |

\| bge-large-zh-v1.5 | 1024 | 中⽂最佳开源 | 智源 |

\| Qwen3-Embedding | 1024 | 阿⾥开源 | Qwen |

\| jina-embeddings-v3 | 1024 | 多语⾔ | Jina |

**## 4.** 相似度计算

\- 余弦相似度（最常⽤）：cos(A, B) = A·B / (|A|·|B|)

\- 范围：\[-1, 1]，越接近 1 越相似

\- 欧⽒距离：直线距离

**## 5. RAG** 中的 **Embedding** 流程

用户问题 → Embedding → 向量

                           ↓

⽂档 → 切块 → Embedding → 向量库

                           ↓

               余弦相似度检索 Top-K

                           ↓

             拼接到 Prompt → LLM ⽣成
