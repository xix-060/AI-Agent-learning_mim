"""准备 SFT 训练数据：情感分类任务"""

import json
from pathlib import Path

# 情感分类的指令数据
SFT_DATA = [
    # 正面
    {
        "instruction": "判断以下文本的情感（正面/负面/中性）",
        "input": "这家餐厅太好吃了，强烈推荐！",
        "output": "正面",
    },
    {
        "instruction": "分析情感倾向",
        "input": "产品质量很棒，物超所值",
        "output": "正面",
    },
    {"instruction": "情感分类", "input": "今天阳光明媚，心情真好", "output": "正面"},
    {"instruction": "判断情感", "input": "服务态度热情，环境优雅", "output": "正面"},
    {
        "instruction": "分析以下文本的情感",
        "input": "这本书写得真好，受益匪浅",
        "output": "正面",
    },
    {"instruction": "情感分析", "input": "电影太好看了，演技炸裂", "output": "正面"},
    {"instruction": "判断情感", "input": "快递很给力，包装完好", "output": "正面"},
    {
        "instruction": "分析情感倾向",
        "input": "老师讲课生动有趣，收获满满",
        "output": "正面",
    },
    {"instruction": "情感分类", "input": "旅行体验超棒，风景如画", "output": "正面"},
    {
        "instruction": "判断以下文本的情感",
        "input": "手机运行流畅，拍照清晰",
        "output": "正面",
    },
    # 负面
    {
        "instruction": "判断以下文本的情感（正面/负面/中性）",
        "input": "质量太差了，用了一周就坏了",
        "output": "负面",
    },
    {
        "instruction": "分析情感倾向",
        "input": "服务态度恶劣，再也不来了",
        "output": "负面",
    },
    {"instruction": "情感分类", "input": "等了两个小时才上菜，无语", "output": "负面"},
    {"instruction": "判断情感", "input": "产品描述与实物严重不符", "output": "负面"},
    {
        "instruction": "分析以下文本的情感",
        "input": "客服态度敷衍，问题没解决",
        "output": "负面",
    },
    {"instruction": "情感分析", "input": "酒店卫生很差，床单有污渍", "output": "负面"},
    {"instruction": "判断情感", "input": "物流太慢了，等了半个月", "output": "负面"},
    {
        "instruction": "分析情感倾向",
        "input": "课程内容水，不值这个价",
        "output": "负面",
    },
    {"instruction": "情感分类", "input": "噪音太大，影响休息", "output": "负面"},
    {
        "instruction": "判断以下文本的情感",
        "input": "软件经常闪退，体验极差",
        "output": "负面",
    },
    # 中性
    {
        "instruction": "判断以下文本的情感（正面/负面/中性）",
        "input": "今天气温25度",
        "output": "中性",
    },
    {"instruction": "分析情感倾向", "input": "这家店在二楼", "output": "中性"},
    {"instruction": "情感分类", "input": "会议定在下午三点", "output": "中性"},
    {"instruction": "判断情感", "input": "产品重量为500克", "output": "中性"},
    {
        "instruction": "分析以下文本的情感",
        "input": "火车预计十点到站",
        "output": "中性",
    },
    {"instruction": "情感分析", "input": "这本书有300页", "output": "中性"},
    {"instruction": "判断情感", "input": "办公室在五楼", "output": "中性"},
    {"instruction": "分析情感倾向", "input": "快递显示已发货", "output": "中性"},
    {"instruction": "情感分类", "input": "课程安排在周三", "output": "中性"},
    {"instruction": "判断以下文本的情感", "input": "手机电量50%", "output": "中性"},
]

# 扩展数据（通过模板生成更多）
TEMPLATES = {
    "正面": [
        "太{adj}了，强烈推荐！",
        "非常{adj}，值得购买",
        "{adj}，体验很好",
    ],
    "负面": [
        "太{adj}了，不推荐",
        "非常{adj}，浪费钱",
        "{adj}，体验很差",
    ],
}

ADJS = {
    "正面": ["好吃", "好用", "好看", "好玩", "方便", "舒服", "漂亮", "划算"],
    "负面": ["难吃", "难用", "难看", "难玩", "麻烦", "难受", "丑陋", "亏本"],
}


def generate_more_data(base_count=100):
    """生成更多训练数据"""
    import random

    random.seed(42)

    data = list(SFT_DATA)

    while len(data) < base_count:
        for sentiment, adjs in ADJS.items():
            for adj in adjs:
                for tmpl in TEMPLATES[sentiment]:
                    text = tmpl.format(adj=adj)
                    instruction = random.choice(
                        [
                            "判断情感",
                            "分析情感",
                            "情感分类",
                            "判断以下文本的情感",
                        ]
                    )
                    data.append(
                        {
                            "instruction": instruction,
                            "input": text,
                            "output": sentiment,
                        }
                    )

    return data[:base_count]


def save_dataset(data, path="data/sft_sentiment.json"):
    """保存数据集"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存 {len(data)} 条数据到 {path}")

    # 统计
    from collections import Counter

    labels = Counter(d["output"] for d in data)
    print(f"📊 标签分布：{dict(labels)}")


if __name__ == "__main__":
    data = generate_more_data(100)
    save_dataset(data)

    # 打印前 3 条
    print("\n前 3 条数据：")
    for d in data[:3]:
        print(f"  {d['instruction']} | {d['input']} → {d['output']}")
