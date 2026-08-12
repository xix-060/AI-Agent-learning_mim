"""我的第一个 Streamlit 应用：AI Agent 学习演示"""

import streamlit as st


def greet(name: str) -> str:
    if not name or not name.strip():
        return "请先输入你的名字哦~"
    return f"你好 {name.strip()}！这是一个 AI Agent 学习项目。"


st.set_page_config(page_title="我的第一个 Streamlit App", page_icon="🚀")
st.title("🚀 我的第一个 Streamlit App")
st.caption("AI-Agent-learning_mim 任务 3.3：Streamlit Community Cloud 演示")

name = st.text_input("你的名字", placeholder="请输入名字，例如 mim")

if st.button("问候", type="primary"):
    result = greet(name)
    st.success(result)

with st.expander("试试这些名字"):
    st.write("mim / Alice / Bob")
