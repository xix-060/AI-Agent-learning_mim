"""知识库 Agent 入口

用法:
  python main.py import --path <文件路径或URL>
  python main.py chat
  python main.py stats
"""

import os
import sys
import argparse

# 添加项目父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agent.src.agent import KnowledgeAgent


def cmd_import(args):
    """导入文档到知识库"""
    source = args.path_opt or args.path_pos
    if not source:
        print("错误: 请指定文件路径或网页 URL")
        print("  用法: python main.py import <文件路径>")
        print("        python main.py import --path <文件路径>")
        return
    print(f"正在导入: {source}")

    # 只需要 RAG 引擎，不需要完整 Agent
    from knowledge_agent.src.rag import RAGEngine

    rag = RAGEngine()

    result = rag.import_document(source)
    if result.get("success"):
        print("导入成功!")
        print(f"  来源: {result['source']}")
        print(f"  文档数: {result['documents']}")
        print(f"  分块数: {result['chunks']}")
    else:
        print(f"导入失败: {result.get('message', '未知错误')}")

    # 显示当前知识库状态
    stats = rag.get_stats()
    print(f"\n知识库状态: {stats}")


def cmd_chat(args):
    """交互式对话"""
    print("=" * 60)
    print("  个人知识库 Agent")
    print("=" * 60)

    try:
        agent = KnowledgeAgent()
        stats = agent.get_stats()
        print(f"  知识库: {stats}")
        print("  输入 'help' 查看帮助，'quit' 退出")
        print("=" * 60)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    while True:
        try:
            user_input = input("\n[您] ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("再见!")
                break

            if user_input.lower() == "help":
                _show_help()
                continue

            if user_input.lower() == "stats":
                print(f"  知识库: {agent.get_stats()}")
                continue

            if user_input.lower() == "clear":
                agent.clear_memory()
                print("  对话记忆已清空")
                continue

            print("\n[Agent] ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n[错误] {e}")


def cmd_stats(args):
    """显示知识库统计"""
    from knowledge_agent.src.rag import RAGEngine

    rag = RAGEngine()
    stats = rag.get_stats()
    print("知识库统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _show_help():
    print("""
可用命令:
  help    显示帮助
  stats   查看知识库状态
  clear   清空对话记忆
  quit    退出

示例对话:
  [您] 你好
  [您] 现在几点了？
  [您] 帮我算一下 2 * 3 + 4
  [您] 列出上传的文件
""")


def main():
    parser = argparse.ArgumentParser(
        description="个人知识库 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py import ./data/sample.pdf
  python main.py import https://example.com/article
  python main.py import --path ./data/sample.pdf
  python main.py chat
  python main.py stats
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # import 子命令（支持位置参数和 --path 两种写法）
    p_import = subparsers.add_parser("import", help="导入文档到知识库")
    p_import.add_argument("path_pos", nargs="?", help="文件路径或网页 URL")
    p_import.add_argument("--path", dest="path_opt", help="文件路径或网页 URL")

    # chat 子命令
    subparsers.add_parser("chat", help="交互式对话")

    # stats 子命令
    subparsers.add_parser("stats", help="查看知识库统计")

    args = parser.parse_args()

    if args.command == "import":
        cmd_import(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
