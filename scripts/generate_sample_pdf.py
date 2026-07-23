"""生成 RAG 示例 PDF"""

import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("正在安装 fpdf2...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF


class ChinesePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", size=10)
        self.cell(0, 8, "RAG Knowledge Base - Sample Document", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


pdf = ChinesePDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()
pdf.set_font("Helvetica", size=14)


content = """RAG (Retrieval-Augmented Generation) Technology

1. Introduction to RAG

RAG is a technique that combines information retrieval with text generation to improve the accuracy and reliability of large language models. The core idea is simple: instead of relying solely on the model's pre-trained knowledge, we retrieve relevant documents from a knowledge base and use them as context for generating answers.

2. How RAG Works - Three Core Steps

Step 1: Indexing
- Split documents into manageable chunks
- Convert each chunk into a vector using an embedding model
- Store vectors in a vector database along with their text content

Step 2: Retrieval
- Convert the user's question into a vector using the same embedding model
- Search the vector database for the top-K most similar chunks
- Return the retrieved chunks as context

Step 3: Generation
- Combine the retrieved context with the user's question
- Send to the LLM with a prompt like "Answer based on this context..."
- The LLM generates a grounded, accurate response

3. Advantages of RAG

- Reduces Hallucination: The model answers based on real documents, not guesses
- Knowledge Updates: Simply add new documents without retraining the model
- Source Attribution: Every answer can show which documents were used
- Data Privacy: Documents stay in your control, no need to send to model providers

4. Vector Databases

Vector databases are specialized for storing and searching embeddings:

- Chroma: Lightweight, local-first, great for prototyping
- FAISS: Facebook's library, high performance, no server needed
- Milvus: Distributed, production-scale, handles billions of vectors
- Pinecone: Cloud-hosted, fully managed
- Weaviate: Open-source, supports hybrid search (vector + keyword)

Recommendation: Start with Chroma for development, migrate to Milvus for production.

5. Embedding Models

Embedding quality directly impacts RAG performance:

- text-embedding-3-small (OpenAI): 1536 dimensions, good general-purpose
- bge-small-zh-v1.5 (BAAI): Chinese-optimized, 512 dimensions
- jina-embeddings-v2: 768 dimensions, multilingual support
- text-embedding-v3 (Alibaba): 1024 dimensions, good for Chinese

6. Chunking Strategies

How you split documents affects retrieval quality:

- Fixed Size: Split by character count (e.g., 500 chars) with overlap
- Sentence-based: Split at sentence boundaries using punctuation
- Paragraph-based: Split at paragraph boundaries (double newlines)
- Semantic Chunking: Use embeddings to find natural break points

Tip: Use overlap (50-100 chars) between chunks to avoid losing context at boundaries.

7. Frameworks for Building RAG

- LangChain: Most popular, modular components, large ecosystem
- LlamaIndex: Focused on RAG, better for document processing
- Haystack: Production-ready, flexible pipeline architecture
- DSPy: Programmatic control over RAG pipelines

8. Advanced RAG Techniques

- Query Rewriting: Expand or clarify user queries before retrieval
- Hybrid Search: Combine vector similarity with keyword matching (BM25)
- Reranking: Use a cross-encoder to reorder retrieved results
- Multi-Query: Generate multiple query variations and combine results
- Graph RAG: Use knowledge graphs for more structured retrieval
"""

lines = content.strip().split("\n\n")
for para in lines:
    pdf.multi_cell(0, 7, para.strip())
    pdf.ln(4)


output = Path(__file__).parent.parent / "data" / "sample.pdf"
output.parent.mkdir(exist_ok=True)
pdf.output(str(output))
print(f"PDF generated: {output}")
