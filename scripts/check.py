"""一键检查所有 Python 文件的语法和结构"""

import ast
import sys
from pathlib import Path


def check_file(filepath: str) -> list[str]:
    """检查单个文件，返回错误列表"""
    errors = []

    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"  ❌ 语法错误: {e}")
        return errors

    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            actual_indent = len(lines[node.lineno - 1]) - len(
                lines[node.lineno - 1].lstrip()
            )
            expected_indent = (node.col_offset // 4) * 4

            if actual_indent != expected_indent and actual_indent != node.col_offset:
                errors.append(
                    f"  ⚠️  缩进警告: {node.name} (L{node.lineno}) "
                    f"实际={actual_indent}空格, 预期={node.col_offset}"
                )

    return errors


def check_path(path: str) -> int:
    """检查路径（文件或目录），返回错误数"""
    errors_found = 0
    path = Path(path)

    if path.is_file() and path.suffix == ".py":
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.py"))
    else:
        print(f"❌ 路径不存在: {path}")
        return 1

    print(f"🔍 检查 {len(files)} 个 Python 文件...\n")

    for f in files:
        try:
            rel = f.relative_to(Path.cwd())
        except ValueError:
            rel = f
        errors = check_file(str(f))
        if errors:
            print(f"📄 {rel}")
            for e in errors:
                print(e)
            print()
            errors_found += len(errors)

    if errors_found == 0:
        print("✅ 所有文件检查通过！")
    else:
        print(f"❌ 发现 {errors_found} 个问题")

    return errors_found


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "."

    exit_code = check_path(target)
    sys.exit(exit_code)
