"""修复 Unicode 同形字 - 将 Kangxi 部首区字符替换为标准 CJK 字符"""

HOMOGLYPH_MAP = {
    "\u2f63": "\u751f",  # 生
    "\u2f64": "\u7528",  # 用
    "\u2f08": "\u4eba",  # 人
    "\u2f2f": "\u5de5",  # 工
    "\u2f00": "\u4e00",  # 一
    "\u2f40": "\u652f",  # 支
    "\u2f12": "\u529b",  # 力
    "\u2f45": "\u65b9",  # 方
    "\u2fb8": "\u9996",  # 首
    "\u2f06": "\u4e8c",  # 二
    "\u2f84": "\u81f3",  # 至
    "\u2f79": "\u7f51",  # 网
    "\u2f24": "\u5927",  # 大
    "\u2f3c": "\u5fc3",  # 心
    "\u2f26": "\u5b50",  # 子
    "\u2f7d": "\u800c",  # 而
    "\u2f46": "\u65e0",  # 无
    "\u2f83": "\u81ea",  # 自
    "\u2f94": "\u8a00",  # 言
    "\u2fa5": "\u91cc",  # 里
    "\u2f4c": "\u6b62",  # 止
    "\u2f9c": "\u8db3",  # 足
    "\u2f42": "\u6587",  # 文
    "\u2f8f": "\u884c",  # 行
    "\u2f3e": "\u6237",  # 户
    "\u2ec5": "\u89c1",  # 见
}


def fix_file(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for old, new in HOMOGLYPH_MAP.items():
        content = content.replace(old, new)

    count = sum(1 for a, b in zip(original, content) if a != b)

    if count > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {count} homoglyphs in {filepath}")
    else:
        print(f"No homoglyphs found in {filepath}")

    return count


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = ["scripts/gen_sample_data.py"]

    total = 0
    for f in files:
        total += fix_file(f)

    print(f"\nTotal fixed: {total} characters")
