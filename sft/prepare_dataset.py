"""
步骤 1: 准备数据集并格式化为 ChatML 格式
"""
from datasets import load_dataset
import json

def format_to_chatml(example):
    """
    将数据格式化为 ChatML 格式
    ChatML 格式:
    <|im_start|>system
    You are a helpful financial assistant.
    <|im_end|>
    <|im_start|>user
    {instruction}
    <|im_end|>
    <|im_start|>assistant
    {output}
    <|im_end|>
    """
    # 构建用户消息
    if example['input'].strip():
        user_message = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_message = example['instruction']
    
    # ChatML 格式
    chatml_text = (
        "<|im_start|>system\n"
        "You are a helpful financial assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_message}\n"
        "<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{example['output']}\n"
        "<|im_end|>"
    )
    
    return {
        'text': chatml_text,
        'instruction': example['instruction'],
        'input': example['input'],
        'output': example['output']
    }

def main():
    print("=" * 60)
    print("步骤 1: 加载和准备数据集")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n1. 正在加载 finance-alpaca 数据集...")
    dataset = load_dataset("gbharti/finance-alpaca")
    print(f"✓ 数据集加载成功！训练样本数: {len(dataset['train'])}")
    
    # 2. 查看原始数据
    print("\n2. 原始数据示例:")
    print("-" * 60)
    print(f"Instruction: {dataset['train'][0]['instruction'][:100]}...")
    print(f"Input: {dataset['train'][0]['input']}")
    print(f"Output: {dataset['train'][0]['output'][:100]}...")
    
    # 3. 格式化为 ChatML
    print("\n3. 正在格式化为 ChatML 格式...")
    formatted_dataset = dataset.map(format_to_chatml)
    print("✓ 格式化完成！")
    
    # 4. 查看格式化后的数据
    print("\n4. ChatML 格式示例:")
    print("-" * 60)
    print(formatted_dataset['train'][0]['text'][:300])
    print("...")
    
    # 5. 划分训练集和验证集
    print("\n5. 划分训练集和验证集 (90% 训练, 10% 验证)...")
    train_test = formatted_dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 验证集大小: {len(eval_dataset)}")
    
    # 6. 保存数据集
    print("\n6. 保存处理后的数据集...")
    
    # 保存为磁盘格式 (推荐用于训练)
    train_dataset.save_to_disk("./data/train")
    eval_dataset.save_to_disk("./data/eval")
    print("✓ 已保存到 ./data/train 和 ./data/eval")
    
    # 保存为 JSON 格式 (方便查看)
    train_dataset.to_json("./data/train.json", orient='records', lines=True)
    eval_dataset.to_json("./data/eval.json", orient='records', lines=True)
    print("✓ 已保存 JSON 格式到 ./data/train.json 和 ./data/eval.json")
    
    # 7. 统计信息
    print("\n7. 数据集统计:")
    print("-" * 60)
    
    # 计算平均长度
    train_lengths = [len(example['text']) for example in train_dataset]
    avg_length = sum(train_lengths) / len(train_lengths)
    max_length = max(train_lengths)
    min_length = min(train_lengths)
    
    print(f"平均文本长度: {avg_length:.0f} 字符")
    print(f"最长文本: {max_length} 字符")
    print(f"最短文本: {min_length} 字符")
    
    # 统计有 input 字段的样本比例
    with_input = sum(1 for ex in train_dataset if ex['input'].strip())
    print(f"包含 input 字段的样本: {with_input} ({with_input/len(train_dataset)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ 步骤 1 完成！数据已准备好用于训练。")
    print("=" * 60)
    
    # 8. 创建一个测试集用于最后的评估
    print("\n8. 创建测试集用于最终评估...")
    test_prompts = [
        {
            "instruction": "What is the difference between stocks and bonds?",
            "input": ""
        },
        {
            "instruction": "Explain what happens when the Federal Reserve raises interest rates.",
            "input": ""
        },
        {
            "instruction": "Should I invest in index funds or individual stocks?",
            "input": "I'm a beginner investor with a 10-year time horizon."
        },
        {
            "instruction": "What is diversification and why is it important?",
            "input": ""
        },
        {
            "instruction": "Explain the concept of compound interest.",
            "input": ""
        }
    ]
    
    with open('./data/test_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(test_prompts, f, indent=2, ensure_ascii=False)
    print("✓ 测试提示已保存到 ./data/test_prompts.json")
    
    return train_dataset, eval_dataset

if __name__ == "__main__":
    # 创建数据目录
    import os
    os.makedirs("./data", exist_ok=True)
    
    # 运行主函数
    train_dataset, eval_dataset = main()
