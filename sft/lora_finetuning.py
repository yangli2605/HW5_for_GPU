"""
步骤 2: LoRA 微调 (使用 peft 和 trl)
预计时间: 30分钟 - 1小时
"""
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
from datetime import datetime

def main():
    print("=" * 60)
    print("步骤 2: LoRA 微调")
    print("=" * 60)
    
    # ============= 配置参数 =============
    
    # 模型配置
    MODEL_NAME = "microsoft/phi-2"  # 使用较小的模型以加快训练速度
    # 其他选择:
    # "meta-llama/Llama-2-7b-hf"
    # "mistralai/Mistral-7B-v0.1"
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # LoRA 配置
    LORA_R = 16  # LoRA 秩
    LORA_ALPHA = 32  # LoRA alpha 参数
    LORA_DROPOUT = 0.05
    
    # 训练配置
    OUTPUT_DIR = "./outputs/lora_model"
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 512
    
    # 是否使用量化 (4-bit)
    USE_4BIT = True
    
    print(f"\n配置:")
    print(f"  模型: {MODEL_NAME}")
    print(f"  LoRA 秩: {LORA_R}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  最大序列长度: {MAX_SEQ_LENGTH}")
    print(f"  使用 4-bit 量化: {USE_4BIT}")
    
    # ============= 1. 加载数据集 =============
    print("\n1. 加载数据集...")
    train_dataset = load_from_disk("./data/train")
    eval_dataset = load_from_disk("./data/eval")
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
    print(f"  PAD token: {tokenizer.pad_token}")
    
    # ============= 3. 配置量化 =============
    if USE_4BIT:
        print("\n3. 配置 4-bit 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✓ 量化配置完成")
    else:
        bnb_config = None
        print("\n3. 不使用量化")
    
    # ============= 4. 加载模型 =============
    print("\n4. 加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if USE_4BIT else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not USE_4BIT else None,
    )
    
    # 如果使用量化，准备模型
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    print(f"✓ 模型加载完成")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # ============= 5. 配置 LoRA =============
    print("\n5. 配置 LoRA...")
    
    # 自动检测目标模块
    # 对于不同的模型架构，目标模块名称可能不同
    target_modules = ["q_proj", "v_proj"]  # 默认值
    
    # 根据模型类型调整
    if "phi" in MODEL_NAME.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    elif "llama" in MODEL_NAME.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ LoRA 配置完成")
    print(f"  目标模块: {target_modules}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练参数比例: {100 * trainable_params / total_params:.2f}%")
    
    # ============= 6. 配置训练参数 =============
    print("\n6. 配置训练参数...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",  # 修改: 使用 eval_strategy 代替 evaluation_strategy
        fp16=True,
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        report_to="tensorboard",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    print(f"✓ 训练参数配置完成")
    print(f"  有效批大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  总训练步数: ~{len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
    
    # ============= 7. 创建 Trainer =============
    print("\n7. 创建 SFT Trainer...")
    
    # 格式化函数
    def formatting_func(example):
        return example['text']
    
    # 尝试不同的参数组合以兼容不同版本的 TRL
    try:
        # 首先尝试新版本参数
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # 直接指定文本字段
            tokenizer=tokenizer,
            packing=False,
        )
        print("✓ Trainer 创建完成 (使用 dataset_text_field)")
    except TypeError as e:
        print(f"尝试备选参数...")
        # 如果失败，尝试使用 formatting_func
        try:
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                formatting_func=formatting_func,
                tokenizer=tokenizer,
            )
            print("✓ Trainer 创建完成 (使用 formatting_func)")
        except TypeError as e2:
            print(f"✗ 创建失败: {e2}")
            print("\n尝试使用标准 Trainer 而不是 SFTTrainer...")
            
            # 最后的备选方案：使用标准 Trainer
            from transformers import Trainer, DataCollatorForLanguageModeling
            
            # Tokenize 数据集
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    padding='max_length'
                )
            
            tokenized_train = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            tokenized_eval = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
            
            # 设置 labels
            tokenized_train = tokenized_train.map(
                lambda x: {"labels": x["input_ids"]},
                batched=True
            )
            tokenized_eval = tokenized_eval.map(
                lambda x: {"labels": x["input_ids"]},
                batched=True
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=data_collator,
            )
            print("✓ Trainer 创建完成 (使用标准 Trainer)")
    
    print("✓ Trainer 已准备就绪")
    
    # ============= 8. 开始训练 =============
    print("\n8. 开始训练...")
    print("=" * 60)
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 训练
    trainer.train()
    
    print("\n" + "=" * 60)
    print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # ============= 9. 保存模型 =============
    print("\n9. 保存模型...")
    
    # 保存 LoRA 适配器
    model.save_pretrained(f"{OUTPUT_DIR}/final_lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora_adapter")
    
    print(f"✓ LoRA 适配器已保存到: {OUTPUT_DIR}/final_lora_adapter")
    
    # 保存训练日志
    import json
    log_history = trainer.state.log_history
    with open(f"{OUTPUT_DIR}/training_log.json", 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"✓ 训练日志已保存到: {OUTPUT_DIR}/training_log.json")
    
    # ============= 10. 显示最终指标 =============
    print("\n10. 训练总结:")
    print("-" * 60)
    
    # 获取最终指标
    final_eval = trainer.evaluate()
    print(f"最终验证损失: {final_eval['eval_loss']:.4f}")
    
    # 从日志中提取训练损失趋势
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    if train_losses:
        print(f"初始训练损失: {train_losses[0]:.4f}")
        print(f"最终训练损失: {train_losses[-1]:.4f}")
        print(f"损失降低: {train_losses[0] - train_losses[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 步骤 2 完成！LoRA 微调完成。")
    print("=" * 60)
    print(f"\n模型保存在: {OUTPUT_DIR}/final_lora_adapter")
    print("下一步: 运行 3_full_finetuning.py 进行全量微调")

if __name__ == "__main__":
    main()
