"""学术 GraphRAG 问答系统 - Streamlit 界面"""

import sys
from pathlib import Path

# 跨目录 import：项目根使 src.* 可导入；scholar_knowledge/src 使本地图谱模块可导入。
# 注意：本地模块必须用扁平 import（如 from graph_builder import ...），
# 不能写 from src.graph_builder import ... —— 否则 src 会先被绑定为
# scholar_knowledge/src 命名空间包并缓存，后续 from src.embedder 就找不到了。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402
from graph_builder import ScholarGraph  # noqa: E402
from hybrid_rag import HybridRAG  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

st.set_page_config(page_title="学术知识图谱问答", layout="wide")


@st.cache_resource
def load_rag():
    graph = ScholarGraph()
    llm = LLMClient()
    embedder = Embedder()
    return HybridRAG(graph, llm, embedder), graph


rag, graph = load_rag()

st.title("🎓 学术知识图谱问答系统")
st.markdown("基于 **NetworkX 图谱 + Chroma 向量** 的混合 GraphRAG")

# 侧边栏：图谱信息
with st.sidebar:
    st.header("📊 图谱信息")
    st.write(f"节点数：{graph.G.number_of_nodes()}")
    st.write(f"关系数：{graph.G.number_of_edges()}")
    st.markdown("---")
    st.header("💡 试试问")
    st.markdown("- 哪些论文被引用最多？")
    st.markdown("- ReAct 引用了哪些论文？")
    st.markdown("- RAG 领域有哪些关键词？")
    st.markdown("- Transformer 有什么特点？")

# 主区域：问答
tab1, tab2 = st.tabs(["💬 问答", "🌐 图谱可视化"])

with tab1:
    question = st.text_input("输入你的问题：", placeholder="如：哪些论文被引用最多？")
    if st.button("回答") and question:
        with st.spinner("检索知识图谱..."):
            answer = rag.query(question, verbose=True)
        st.success(answer)

with tab2:
    st.subheader("学术知识图谱可视化")
    if st.button("生成图谱图"):
        img_path = graph.visualize(Path(__file__).parent / "docs" / "graph-live.png")
        st.image(img_path, use_container_width=True)
