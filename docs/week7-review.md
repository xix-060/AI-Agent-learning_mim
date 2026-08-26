# 第 7 周复盘

📅 **周期**: LoRA 论文精读 → SFT 情感微调 → 数据蒸馏 → QLoRA 4bit 量化 → DPO 偏好对齐 → 微调评估
📊 **代码统计**: 6 个 Commit（08-07\~08-23） | 约 1,793 行代码 | 9 个新源文件 + 6 篇笔记

***

## 1. 最重要的 3 个收获

### 1️⃣ LoRA 低秩分解原理

**核心思想**: 冻结原始权重 W，只训练两个小矩阵 A（降维）和 B（升维），用 ΔW = B·A 近似全参微调的权重更新。因为 A、B 的秩 r 远小于原始维度，参数量减少 99%+，效果却接近全参微调。

**关键数学**（[lora-paper-notes.md](file:///e:/git/AI-Agent-learning_mim/docs/lora-paper-notes.md)）:

| 项        | 说明                                           |
| -------- | -------------------------------------------- |
| 原始权重 W   | 维度 d×d，冻结不训练                                 |
| 低秩矩阵 A、B | A: r×d（高斯初始化）、B: d×r（零初始化），训练前 ΔW=0 保证不破坏原模型 |
| 缩放因子 α   | ΔW 实际乘以 α/r，通常 α=2r                          |
| 参数量对比    | 全参 d² vs LoRA 2dr，r=8 时省 99%+                |

**我的实现**（[lora\_train.py](file:///e:/git/AI-Agent-learning_mim/src/lora_train.py) 的 `create_lora_config`）:

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                # 低秩维度
    lora_alpha=16,      # 缩放因子（= 2r）
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 插在 Attention 投影上
)
```

**numpy 演示**（[lora\_numpy\_demo.py](file:///e:/git/AI-Agent-learning_mim/src/lora_numpy_demo.py)）: 手算了一遍 d=64, r=8 的参数量——全参 4096 vs LoRA 1024，省 75%。

**关键洞察**: LoRA 的本质是 **"权重更新的低秩假设"**——微调改变的行为用低秩就够表达，不需要动全部参数。这也是为什么几百万参数能改变几十亿参数模型的行为。

***

### 2️⃣ SFT 微调全流程（数据→训练→评估闭环）

第一次跑通完整的微调三段式，对应三个文件：

```
prepare_sft_data.py  →  lora_train.py  →  evaluate_model.py
   造数据(100条)        训练(3 epoch)      评估(base vs 微调)
```

**数据准备**（[prepare\_sft\_data.py](file:///e:/git/AI-Agent-learning_mim/src/prepare_sft_data.py)）: 自建 100 条情感分类 SFT 数据，格式 `instruction/input/output`，用模板拼成 `指令：…\n输入：…\n输出：…` 喂给 `SFTTrainer`。

**训练**（[lora\_train.py](file:///e:/git/AI-Agent-learning_mim/src/lora_train.py)）: 骨架和上周 MLP 一模一样——前向算 loss → 反向算梯度 → 优化器更新。区别只是模型换成 Transformer、损失换成语言模型 cross-entropy。`SFTTrainer` 把数据拼装、padding、mask 都封装了，核心只需配 `SFTConfig`。

**评估**（[evaluate\_model.py](file:///e:/git/AI-Agent-learning_mim/src/evaluate_model.py)）: base vs 微调双模型对比，用关键词覆盖率做粗糙指标。

**关键洞察**: SFT 不是"训模型"，而是 **"教模型按特定格式/风格回答"**。微调后模型学会直接输出"正面/负面"标签词，而 base 倾向长篇解释——这是格式对齐，不是知识注入。

***

### 3️⃣ DPO 偏好对齐

**核心思想**: 不用强化学习（PPO 太复杂），直接用"chosen 好 / rejected 差"的偏好对，通过对比损失让模型偏好好回答。DPO 不需要单独的参考模型——配合 PEFT 时，base 模型（关掉 adapter）就是参考模型。

**数据格式**（[dpo\_dataset.json](file:///e:/git/AI-Agent-learning_mim/data/dpo_dataset.json)）:

```json
{
  "prompt": "如何学习 AI？",
  "chosen": "建议分三步：1)学 Python…2)学经典算法…3)学深度学习…",
  "rejected": "AI 很难，你学不会的。"
}
```

**训练**（[dop\_train.py](file:///e:/git/AI-Agent-learning_mim/src/dop_train.py)）: `DPOTrainer` + `DPOConfig`，关键参数 `beta=0.1`（温度，控制偏离参考模型的程度）。

**实际结果**:

| 指标                 | 值      | 解读                              |
| ------------------ | ------ | ------------------------------- |
| train\_loss        | 0.6778 | 起始≈ln(2)=0.693，已下降 ✓            |
| rewards/margins    | +0.031 | 正值 → chosen 奖励 > rejected ✓ 方向对 |
| rewards/accuracies | 0.5    | 50%，8 条数据太少                     |

**关键洞察**: DPO 比 SFT 难在 **"数据比算法贵"**——chosen/rejected 对要精心构造，8 条数据只够验证流程跑通，真正对齐需要几十\~上百条高质量偏好对。

***

## 2. 最难的部分

### trl / transformers API 变化连环坑

本周最折磨的是 **库版本一升级，参数全变了**，每个训练脚本都踩一遍：

1. **`max_prompt_length`** **被删**: trl 1.10 的 `DPOConfig` 移除了这个参数，只剩 `max_length`。原代码照抄旧教程的 `max_prompt_length=256` → 直接 `TypeError`。
   → 解：查 `inspect.signature(DPOConfig.__init__)` 确认有效参数，删掉无效项。
2. **`use_cpu=True`** **必须显式声明**: transformers 5.x 在纯 CPU 机器上跑训练，不写 `use_cpu=True` 直接校验失败。
   → 解：`SFTConfig`/`DPOConfig` 都加 `use_cpu=True` + `bf16=False`。
3. **`device_map="auto"`** **和 PEFT 打架**: CPU 上用 `device_map="auto"` 加载 base 模型，再用 `PeftModel.from_pretrained` 加载 adapter 会触发 meta device 卸载冲突。
   → 解：测试/推理函数里去掉 `device_map="auto"`（训练函数里 trainer 自己管设备，可留）。
4. **`torch_dtype`** **弃用**: transformers 5.x 改名 `dtype`，不报错但刷警告。
   → 解：`torch_dtype=` → `dtype=`。

### HuggingFace 网络超时（老问题新形态）

上周踩过 `HF_ENDPOINT` 必须设，本周又踩一次**新形态**：transformers 5.13.1 加载 tokenizer 时会联网调 `list_repo_templates` 查 chat template，直连 huggingface.co 超时。即使模型已缓存，这步网络调用还是会卡。

→ 解：和上周一样，`os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`，且**必须在** **`from transformers import`** **之前**。三个训练/评估脚本顶部都加了这段。

### CPU 训练慢

DPO 在 CPU 上每步 \~147 秒（要算 chosen+rejected 双倍序列 + 参考模型前向），2 个 epoch 跑了快 5 分钟；评估要 10 次生成 × 200 token，又是几分钟。0.5B 已经如此，7B 在 CPU 上基本不可行——这就是为什么需要 QLoRA 量化 + GPU。

***

## 3. 你的微调效果

### SFT 情感分类（lora\_output）

| 指标     | Base  | 微调    | 提升           |
| ------ | ----- | ----- | ------------ |
| 关键词覆盖率 | 28.0% | 49.0% | **+21.0%** ✅ |

**分题型看**:

| # | 问题            | Base | 微调       | 说明                         |
| - | ------------- | ---- | -------- | -------------------------- |
| 1 | 情感：餐厅太好吃了     | 100% | 100%     | 平，都会                       |
| 2 | 情感：质量太差了      | 0%   | **100%** | 微调直接输出"负面"，base 啰嗦不含词      |
| 3 | 情感：气温25度      | 0%   | 0%       | 都没说"中性"                    |
| 4 | 什么是 AI Agent？ | 40%  | 20%      | **微调反而降**——情感 LoRA 偏移了通用能力 |
| 5 | LoRA 是什么？     | 0%   | 25%      | 两个都答错（base 当成 LoRaWAN 物联网） |

**结论**: 微调在 **目标任务（情感分类）上有效**，主要让输出格式更规整（直接吐标签词）；但 **0.5B + 任务特化会损害通用知识**，且关键词覆盖率是粗糙指标（答对没用指定词得 0%，答错碰巧含词得分）。

### DPO 偏好对齐（dpo\_output）

- loss 0.6778（轻微下降）、margins +0.031（正向）、accuracies 50%
- 数据太少（8 条），只验证了流程跑通，效果不显著

详细报告见 [finetune-eval-report.json](file:///e:/git/AI-Agent-learning_mim/docs/finetune-eval-report.json)。

***

## 4. 微调 vs RAG 怎么选？

本周同时做了 RAG（项目1）和微调，对比清晰了：

| 维度     | 微调（LoRA/SFT/DPO） | RAG               |
| ------ | ---------------- | ----------------- |
| 改什么    | 行为/风格/格式         | 知识/事实             |
| 加新知识   | 差（要重训，易过时）       | **好**（改库即可，实时）    |
| 改输出格式  | **好**（SFT 强对齐格式） | 差（靠 prompt 引导，不稳） |
| 改语气/风格 | **好**（DPO 偏好对齐）  | 差                 |
| 可溯源    | 否（知识融进权重）        | **是**（能指回文档）      |
| 计算成本   | 高（训练）+ 低（推理）     | 低（无训练）+ 高（每查都检索）  |
| 数据要求   | 需标注数据（几十\~上千条）   | 需文档库              |
| 幻觉控制   | 弱（可能更自信地错）       | **强**（有据可依）       |

**我的判断法则**:

- **加新知识/事实更新** → RAG（微调会让模型"自信地记错"，不可溯源）
- **固定输出格式/标签任务**（情感分类、JSON 抽取）→ SFT 微调
- **改语气/安全/偏好** → DPO 微调
- **两者结合最常见**: RAG 注入知识 + 微调调格式风格，互补

本周的评估也印证了——情感 LoRA 让格式更规整（+21%），但知识题反而退化（-20%）。**微调是"窄而深"，RAG 是"宽而浅"**，各管一头。

***

## 5. 下周（AI Coding）想重点学什么？

### 🎯 核心目标

| 优先级 | 任务                 | 预期产出                           |
| --- | ------------------ | ------------------------------ |
| ⭐⭐⭐ | **Agent + IDE 集成** | 理解 Cursor/Copilot 底层的 Agent 架构 |
| ⭐⭐⭐ | **代码生成/补全 Agent**  | 自建一个代码补全或 review 的小 Agent      |
| ⭐⭐  | **工具调用进阶**         | 让 Agent 真正读写文件、跑测试、改代码         |
| ⭐⭐  | **多 Agent 协作**     | coder + reviewer + tester 分工   |
| ⭐   | **微调模型用于代码**       | 试试微调后的小模型能否做简单代码任务             |

### 💡 想搞清楚的问题

1. **AI Coding 的 Agent 和本周的微调模型怎么结合？** 微调能改格式，代码生成能微调出更好的 coder 吗？
2. **Cursor 类工具的核心是什么？** 是 RAG（检索代码库）+ Agent（规划改哪）+ 模型（生成代码）的组合？
3. **怎么让 Agent 安全地改代码？** HITL 审批、测试闭环怎么搭？
4. **0.5B 微调后能写代码吗？** 本周发现 0.5B 知识弱，代码任务可能更难。

### 🔗 前置知识（本周已掌握）

| 下周需要         | 本周学的                   | 关联                  |
| ------------ | ---------------------- | ------------------- |
| Agent 工具调用   | 第4-6周的 LangGraph/Agent | 工具使用是 Agent 核心      |
| 代码库检索        | 项目1的 RAG               | AI Coding 离不开代码 RAG |
| 生成质量评估       | 本周的 evaluate\_model    | 评估思维可迁移到代码评估        |
| 微调 coder（可选） | 本周 SFT/LoRA 流程         | 同一套微调骨架             |

***

## 📊 本周代码统计

### 提交记录（6 个 Commit）

| Commit  | 核心产出                            | 行数  |
| ------- | ------------------------------- | --- |
| 18c69b3 | LLM 训练流程 + LoRA 论文精读 + numpy 演示 | 273 |
| d9dfcc7 | LoRA 微调 Qwen2.5-0.5B 情感分类       | 410 |
| 5da8f8f | 数据蒸馏 + 领域微调                     | 263 |
| d82b839 | QLoRA 4bit 量化微调 7B 模型           | 282 |
| 3c1528d | DPO 偏好对齐训练                      | 267 |
| ef89c98 | 微调评估 + 推理优化笔记                   | 298 |

### 新增文件

```
src/
├── lora_numpy_demo.py        # LoRA 低秩 numpy 手算
├── prepare_sft_data.py       # SFT 数据生成
├── lora_train.py             # LoRA/SFT 训练（含领域微调分支）
├── distill_dataset.py        # LLM 蒸馏生成训练数据
├── qlora_train.py            # QLoRA 4bit 量化训练
├── prepare_dop_data.py       # DPO 数据准备
├── dop_train.py              # DPO 偏好对齐训练
└── evaluate_model.py         # 微调效果评估
docs/
├── llm-training-pipeline.md  # LLM 训练全流程
├── lora-paper-notes.md       # LoRA 论文精读
├── sft-data-format.md        # SFT 数据格式
├── data-distillation.md      # 数据蒸馏
├── quantization-notes.md     # 量化笔记
├── dop-notes.md              # DPO 笔记
├── inference-optimization.md# 推理优化
└── finetune-eval-report.json # 评估报告
```

### 📈 代码量

```
本周新增约 1,793 行（6 个 commit）
```

***

## 📌 本周踩坑沉淀（已进项目记忆）

| 坑                                | 教训                                        |
| -------------------------------- | ----------------------------------------- |
| `simple_chat()` 不接 `temperature` | 要控温用底层 `chat(messages, temperature=…)`    |
| trl 1.10 删了 `max_prompt_length`  | 用 `inspect.signature()` 查有效参数，别照抄旧教程      |
| transformers 5.x CPU 训练          | 必须显式 `use_cpu=True` + `bf16=False`        |
| `device_map="auto"` + PEFT       | 推理/测试函数里去掉，会和 adapter 加载冲突                |
| `torch_dtype` 弃用                 | 改 `dtype=`                                |
| HF tokenizer 联网查 template        | `HF_ENDPOINT` 要在 `import transformers` 前设 |
| DPO 数据太少                         | 8 条只够跑通流程，accuracies 卡在 50%               |

***

## 📝 本周反思

### ✅ 做得好的地方

1. **完整闭环**: 从论文精读 → 数据 → 训练 → 评估，一周打通整条微调流水线，没有停留在"只跑通一个"。
2. **多技术对比**: SFT vs DPO、LoRA vs QLoRA、微调 vs RAG，做了实打实对比，理解了各自的边界。
3. **踩坑即沉淀**: 每个 API 变化、网络问题都进了项目记忆，下周不重蹈。
4. **量化理解到位**: 算清了 4bit 省多少显存（权重 75%、训练总显存 90%+），知道为什么需要 QLoRA。

### ⚠️ 需要改进

1. **数据规模太小**: SFT 100 条、DPO 8 条，都只够验证流程。真要看效果得扩到几百\~上千条。
2. **CPU 训练太慢**: DPO 5 分钟、评估 5 分钟，迭代效率低。下周如有 GPU 条件应优先用上。
3. **评估指标粗糙**: 关键词覆盖率有偏差（答对不用词得 0、答错碰巧含词得分），应加人工判断或语义相似度。

### 🎯 下周期待

> 本周从"用别人的模型"推进到"改别人的模型"——理解了微调改的是行为/格式，不是知识。
>
> 下周进入 AI Coding，把 Agent 和模型结合到代码场景，从"改模型"到"让 Agent 帮我写代码"。

***

**文档生成**: 2026-08-23
**下周目标**: AI Coding（Agent + 代码生成/审查）
