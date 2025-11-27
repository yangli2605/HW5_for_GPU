# SFT (Supervised Fine-Tuning) 项目

这是一个完整的监督微调项目，对比 LoRA 微调和全量微调的效果。

## 项目结构

```
SFT/
├── 1_prepare_dataset.py      # 步骤1: 数据准备和格式化
├── 2_lora_finetuning.py      # 步骤2: LoRA 微调 (30min-1hr)
├── 3_full_finetuning.py      # 步骤3: 全量微调 (2+ hrs)
├── 4_evaluate_compare.py     # 步骤4: 评估和对比
├── run_all.py                 # 一键运行所有步骤
├── requirements.txt           # 依赖包
├── README.md                  # 本文件
├── configs/
│   └── ds_config.json        # DeepSpeed 配置
├── data/                      # 数据目录 (自动创建)
│   ├── train/                # 训练集
│   ├── eval/                 # 验证集
│   └── test_prompts.json     # 测试提示
└── outputs/                   # 输出目录 (自动创建)
    ├── lora_model/           # LoRA 模型
    ├── full_model/           # 全量微调模型
    ├── evaluation_results.json
    └── evaluation_report.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行项目

#### 方式 1: 一键运行

```bash
python run_all.py
```

然后按照提示选择要运行的步骤。

#### 方式 2: 逐步运行

```bash
# 步骤 1: 准备数据集
python 1_prepare_dataset.py

# 步骤 2: LoRA 微调 (30分钟-1小时)
python 2_lora_finetuning.py

# 步骤 3: 全量微调 (2+小时)
python 3_full_finetuning.py

# 步骤 4: 评估和对比
python 4_evaluate_compare.py
```

## 各步骤说明

### 步骤 1: 数据准备

- 从 Hugging Face 下载 `gbharti/finance-alpaca` 数据集
- 格式化为 ChatML 格式
- 划分训练集和验证集 (90/10)
- 保存处理后的数据

**输出**:
- `./data/train/` - 训练数据
- `./data/eval/` - 验证数据
- `./data/test_prompts.json` - 测试提示

### 步骤 2: LoRA 微调

使用 PEFT 和 TRL 进行高效的 LoRA 微调。

**特点**:
- 参数高效: 只训练 < 1% 的参数
- 内存高效: 使用 4-bit 量化
- 速度快: 30分钟 - 1小时

**配置**:
- LoRA rank: 16
- LoRA alpha: 32
- 学习率: 2e-4
- Batch size: 4
- 训练轮数: 3

**输出**:
- `./outputs/lora_model/final_lora_adapter/` - LoRA 适配器
- `./outputs/lora_model/training_log.json` - 训练日志

### 步骤 3: 全量微调

更新模型的所有权重。

**特点**:
- 更新所有参数
- 通常效果更好
- 需要更多内存和时间 (2+ 小时)

**配置**:
- 学习率: 5e-5 (更小)
- Batch size: 2 (更小以适应内存)
- 梯度累积: 8
- 训练轮数: 3

**输出**:
- `./outputs/full_model/final_model/` - 完整模型
- `./outputs/full_model/training_log.json` - 训练日志

### 步骤 4: 评估和对比

比较三个模型的表现:
- 基础模型 (未微调)
- LoRA 微调模型
- 全量微调模型

**评估指标**:
- 困惑度 (Perplexity)
- 回答质量
- 风格匹配
- 有用性

**输出**:
- `./outputs/evaluation_results.json` - JSON 格式结果
- `./outputs/evaluation_report.md` - Markdown 报告

## 配置说明

### 修改基础模型

在各个脚本中修改 `MODEL_NAME` 变量:

```python
MODEL_NAME = "microsoft/phi-2"  # 默认
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### 使用 DeepSpeed

在 `3_full_finetuning.py` 中:

```python
USE_DEEPSPEED = True
```

确保已安装 DeepSpeed:

```bash
pip install deepspeed
```

### 调整超参数

在各脚本顶部的配置部分修改参数，例如:

```python
# LoRA 配置
LORA_R = 16  # 增加以提高容量
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练配置
NUM_EPOCHS = 3  # 增加训练轮数
BATCH_SIZE = 4  # 根据显存调整
LEARNING_RATE = 2e-4  # 调整学习率
```

## 硬件要求

### 最低配置 (使用 4-bit 量化)
- GPU: 8GB VRAM (如 RTX 3060)
- RAM: 16GB
- 存储: 20GB

### 推荐配置
- GPU: 24GB VRAM (如 RTX 3090/4090)
- RAM: 32GB
- 存储: 50GB

### 全量微调大模型 (7B+)
- GPU: 40GB+ VRAM (如 A100)
- 或使用 DeepSpeed ZeRO-3 offload

## 监控训练

### 使用 TensorBoard

```bash
tensorboard --logdir ./outputs/
```

然后在浏览器打开 `http://localhost:6006`

### 查看日志

```bash
# LoRA 训练日志
cat ./outputs/lora_model/training_log.json

# 全量微调日志
cat ./outputs/full_model/training_log.json
```

## 故障排除

### 内存不足 (OOM)

1. 减小 batch size
2. 增加梯度累积步数
3. 启用 4-bit 量化
4. 使用更小的模型
5. 减小序列长度

### 训练速度慢

1. 使用更小的模型进行测试
2. 减少训练数据量
3. 使用更少的训练轮数
4. 考虑使用多 GPU

### 模型加载失败

1. 检查模型名称是否正确
2. 确保有足够的磁盘空间
3. 检查网络连接 (下载模型)
4. 尝试手动下载模型到本地

## 结果解读

### 困惑度 (Perplexity)

- 越低越好
- 表示模型对文本的预测能力
- 典型范围: 5-50

### 损失 (Loss)

- 训练损失应该稳定下降
- 验证损失 < 训练损失: 正常
- 验证损失 > 训练损失: 可能过拟合

### 过拟合检查

如果验证损失开始上升而训练损失继续下降，说明模型过拟合。

解决方法:
- 增加数据量
- 使用更强的正则化
- 减少训练轮数
- 使用 dropout

## 数据集选择

项目默认使用 `gbharti/finance-alpaca`，你也可以使用其他数据集:

```python
# 在 1_prepare_dataset.py 中修改
dataset = load_dataset("your-dataset-name")
```

推荐的金融数据集:
- `gbharti/finance-alpaca` (51K 样本)
- `takala/financial_phrasebank` (情感分析)
- `AdaptLLM/finance-tasks` (多任务)
- `Josephgflowers/Finance-Instruct-500k` (500K 样本)

## 进阶使用

### 自定义提示格式

修改 `format_to_chatml()` 函数来使用不同的格式:

```python
# Alpaca 格式
def format_alpaca(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

# Llama 格式
def format_llama(example):
    return f"<s>[INST] {example['instruction']} [/INST] {example['output']} </s>"
```

### 多 GPU 训练

使用 `accelerate`:

```bash
accelerate config
accelerate launch 2_lora_finetuning.py
```

### 使用自己的数据

创建 JSON 文件，格式如下:

```json
[
  {
    "instruction": "问题",
    "input": "上下文 (可选)",
    "output": "答案"
  }
]
```

然后修改数据加载部分。

## 参考资源

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [DeepSpeed](https://www.deepspeed.ai/)

## License

MIT License

## 贡献

欢迎提交 issues 和 pull requests!
