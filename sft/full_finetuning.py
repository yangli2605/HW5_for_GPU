"""
步骤 3: 全量微调 (Full Fine-Tuning)
预计时间: 2+ 小时
注意: 全量微调需要更多的 GPU 内存和时间
"""
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os
from datetime import datetime

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize 函数"""
    # Tokenize 文本
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None
    )
    
    # 设置 labels (用于计算 loss)
    result["labels"] = result["input_ids"].copy()
    
    return result

def main():
    print("=" * 60)
    print("步骤 3: 全量微调 (Full Fine-Tuning)")
    print("=" * 60)
    
    # ============= 配置参数 =============
    
    # 模型配置
    MODEL_NAME = "microsoft/phi-2"  # 使用较小的模型
    # 注意: 全量微调对于大模型 (7B+) 需要大量显存
    # 可选择: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (更小更快)
    
    # 训练配置
    OUTPUT_DIR = "./outputs/full_model"
    NUM_EPOCHS = 3
    BATCH_SIZE = 2  # 全量微调需要更多内存，减小批大小
    GRADIENT_ACCUMULATION_STEPS = 8  # 增加梯度累积以补偿小批大小
    LEARNING_RATE = 5e-5  # 全量微调使用更小的学习率
    MAX_SEQ_LENGTH = 512
    WEIGHT_DECAY = 0.01
    
    # 是否使用 DeepSpeed (需要配置文件)
    USE_DEEPSPEED = False
    
    print(f"\n配置:")
    print(f"  模型: {MODEL_NAME}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  有效批大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  最大序列长度: {MAX_SEQ_LENGTH}")
    print(f"  使用 DeepSpeed: {USE_DEEPSPEED}")
    
    # ============= 1. 加载数据集 =============
    print("\n1. 加载数据集...")
    train_dataset = load_from_disk("./data/train")
    eval_dataset = load_from_disk("./data/eval")
    
    # 可选: 使用数据子集以加快训练 (用于测试)
    USE_SUBSET = False
    if USE_SUBSET:
        train_dataset = train_dataset.select(range(1000))
        eval_dataset = eval_dataset.select(range(100))
        print(f"  [使用数据子集进行测试]")
    
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(eval_dataset)} 样本")
    
    # ============= 2. 加载 Tokenizer =============
    print("\n2. 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ Tokenizer 加载完成")
    print(f"  词汇表大小: {len(tokenizer)}")
    
    # ============= 3. Tokenize 数据集 =============
    print("\n3. Tokenizing 数据集...")
    print("  这可能需要几分钟...")
    
    # Tokenize
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset"
    )
    
    print(f"✓ Tokenization 完成")
    print(f"  训练样本: {len(tokenized_train)}")
    print(f"  验证样本: {len(tokenized_eval)}")
    
    # ============= 4. 加载模型 =============
    print("\n4. 加载基础模型...")
    print("  注意: 全量微调将更新所有参数")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    print(f"✓ 模型加载完成")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练参数比例: {100 * trainable_params / total_params:.2f}%")
    
    # ============= 5. 配置训练参数 =============
    print("\n5. 配置训练参数...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="tensorboard",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # DeepSpeed 配置 (如果使用)
        deepspeed="./configs/ds_config.json" if USE_DEEPSPEED else None,
    )
    
    print(f"✓ 训练参数配置完成")
    total_steps = len(tokenized_train) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print(f"  预计总训练步数: {total_steps}")
    
    # ============= 6. 创建 Data Collator =============
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型，不使用 MLM
    )
    
    # ============= 7. 创建 Trainer =============
    print("\n7. 创建 Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    print("✓ Trainer 创建完成")
    
    # ============= 8. 评估初始性能 =============
    print("\n8. 评估初始性能...")
    initial_eval = trainer.evaluate()
    print(f"✓ 初始验证损失: {initial_eval['eval_loss']:.4f}")
    
    # ============= 9. 开始训练 =============
    print("\n9. 开始全量微调...")
    print("=" * 60)
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚠️  全量微调通常需要 2+ 小时，请耐心等待...")
    print("=" * 60)
    
    # 训练
    trainer.train()
    
    print("\n" + "=" * 60)
    print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # ============= 10. 保存模型 =============
    print("\n10. 保存模型...")
    
    # 保存完整模型
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print(f"✓ 完整模型已保存到: {OUTPUT_DIR}/final_model")
    
    # 保存训练日志
    import json
    log_history = trainer.state.log_history
    with open(f"{OUTPUT_DIR}/training_log.json", 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"✓ 训练日志已保存到: {OUTPUT_DIR}/training_log.json")
    
    # ============= 11. 最终评估 =============
    print("\n11. 最终评估...")
    final_eval = trainer.evaluate()
    
    print("\n训练总结:")
    print("-" * 60)
    print(f"初始验证损失: {initial_eval['eval_loss']:.4f}")
    print(f"最终验证损失: {final_eval['eval_loss']:.4f}")
    print(f"损失降低: {initial_eval['eval_loss'] - final_eval['eval_loss']:.4f}")
    
    # 从日志中提取训练损失
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    if train_losses:
        print(f"\n训练损失:")
        print(f"  初始: {train_losses[0]:.4f}")
        print(f"  最终: {train_losses[-1]:.4f}")
        print(f"  降低: {train_losses[0] - train_losses[-1]:.4f}")
    
    # 检查过拟合
    if len(train_losses) > 0:
        overfitting_gap = train_losses[-1] - final_eval['eval_loss']
        print(f"\n过拟合检查:")
        print(f"  训练-验证损失差距: {overfitting_gap:.4f}")
        if overfitting_gap < -0.5:
            print(f"  ⚠️  可能存在过拟合 (验证损失 > 训练损失)")
        else:
            print(f"  ✓ 没有明显过拟合")
    
    print("\n" + "=" * 60)
    print("✓ 步骤 3 完成！全量微调完成。")
    print("=" * 60)
    print(f"\n模型保存在: {OUTPUT_DIR}/final_model")
    print("下一步: 运行 4_evaluate_compare.py 进行评估和对比")

if __name__ == "__main__":
    main()
