"""
步骤 4: 评估和对比
比较基础模型、LoRA 微调模型和全量微调模型的表现
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime
import pandas as pd

def load_base_model(model_name):
    """加载基础模型"""
    print(f"加载基础模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_lora_model(base_model_name, lora_path):
    """加载 LoRA 微调模型"""
    print(f"加载 LoRA 模型: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()  # 合并 LoRA 权重
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_full_model(model_path):
    """加载全量微调模型"""
    print(f"加载全量微调模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    """生成回答"""
    # 格式化为 ChatML
    formatted_prompt = (
        "<|im_start|>system\n"
        "You are a helpful financial assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 的回答
    try:
        assistant_response = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
    except:
        assistant_response = response
    
    return assistant_response

def calculate_perplexity(model, tokenizer, text):
    """计算困惑度"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

def main():
    print("=" * 80)
    print("步骤 4: 模型评估和对比")
    print("=" * 80)
    
    # ============= 配置 =============
    BASE_MODEL_NAME = "microsoft/phi-2"
    LORA_MODEL_PATH = "./outputs/lora_model/final_lora_adapter"
    FULL_MODEL_PATH = "./outputs/full_model/final_model"
    TEST_PROMPTS_PATH = "./data/test_prompts.json"
    OUTPUT_PATH = "./outputs/evaluation_results.json"
    
    # ============= 1. 加载测试提示 =============
    print("\n1. 加载测试提示...")
    with open(TEST_PROMPTS_PATH, 'r', encoding='utf-8') as f:
        test_prompts = json.load(f)
    
    print(f"✓ 加载了 {len(test_prompts)} 个测试提示")
    
    # ============= 2. 加载模型 =============
    print("\n2. 加载所有模型...")
    
    models = {}
    
    # 基础模型
    try:
        base_model, base_tokenizer = load_base_model(BASE_MODEL_NAME)
        models['base'] = (base_model, base_tokenizer)
        print("✓ 基础模型加载成功")
    except Exception as e:
        print(f"✗ 基础模型加载失败: {e}")
        models['base'] = None
    
    # LoRA 模型
    try:
        lora_model, lora_tokenizer = load_lora_model(BASE_MODEL_NAME, LORA_MODEL_PATH)
        models['lora'] = (lora_model, lora_tokenizer)
        print("✓ LoRA 模型加载成功")
    except Exception as e:
        print(f"✗ LoRA 模型加载失败: {e}")
        models['lora'] = None
    
    # 全量微调模型
    try:
        full_model, full_tokenizer = load_full_model(FULL_MODEL_PATH)
        models['full'] = (full_model, full_tokenizer)
        print("✓ 全量微调模型加载成功")
    except Exception as e:
        print(f"✗ 全量微调模型加载失败: {e}")
        models['full'] = None
    
    # ============= 3. 生成回答 =============
    print("\n3. 生成回答...")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n测试案例 {i}/{len(test_prompts)}")
        print("-" * 80)
        
        # 构建提示
        if test_case['input']:
            prompt = f"{test_case['instruction']}\n\nContext: {test_case['input']}"
        else:
            prompt = test_case['instruction']
        
        print(f"提示: {prompt[:100]}...")
        
        result = {
            'prompt': prompt,
            'instruction': test_case['instruction'],
            'input': test_case['input'],
            'responses': {}
        }
        
        # 对每个模型生成回答
        for model_name, model_tuple in models.items():
            if model_tuple is None:
                print(f"  [{model_name.upper()}] 模型未加载，跳过")
                continue
            
            model, tokenizer = model_tuple
            print(f"  [{model_name.upper()}] 生成中...")
            
            try:
                response = generate_response(model, tokenizer, prompt)
                result['responses'][model_name] = response
                print(f"  [{model_name.upper()}] ✓ 生成完成 ({len(response)} 字符)")
                print(f"  回答预览: {response[:100]}...")
            except Exception as e:
                print(f"  [{model_name.upper()}] ✗ 生成失败: {e}")
                result['responses'][model_name] = f"Error: {str(e)}"
        
        results.append(result)
    
    # ============= 4. 计算困惑度 =============
    print("\n" + "=" * 80)
    print("4. 计算困惑度...")
    print("-" * 80)
    
    # 准备测试文本（使用前 100 个样本）
    from datasets import load_from_disk
    eval_dataset = load_from_disk("./data/eval")
    test_texts = [ex['text'] for ex in eval_dataset.select(range(min(100, len(eval_dataset))))]
    
    perplexities = {}
    
    for model_name, model_tuple in models.items():
        if model_tuple is None:
            continue
        
        model, tokenizer = model_tuple
        print(f"\n[{model_name.upper()}] 计算困惑度...")
        
        ppls = []
        for text in test_texts[:10]:  # 只用前 10 个样本以节省时间
            try:
                ppl = calculate_perplexity(model, tokenizer, text)
                ppls.append(ppl)
            except:
                continue
        
        avg_ppl = sum(ppls) / len(ppls) if ppls else float('inf')
        perplexities[model_name] = avg_ppl
        print(f"  平均困惑度: {avg_ppl:.2f}")
    
    # ============= 5. 保存结果 =============
    print("\n" + "=" * 80)
    print("5. 保存结果...")
    
    evaluation_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_model': BASE_MODEL_NAME,
        'test_cases': results,
        'perplexities': perplexities,
        'summary': {
            'num_test_cases': len(test_prompts),
            'models_evaluated': list(models.keys())
        }
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 评估结果已保存到: {OUTPUT_PATH}")
    
    # ============= 6. 生成对比报告 =============
    print("\n" + "=" * 80)
    print("6. 生成对比报告")
    print("=" * 80)
    
    # 创建报告
    report_lines = []
    report_lines.append("# 模型评估对比报告\n")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"基础模型: {BASE_MODEL_NAME}\n\n")
    
    # 困惑度对比
    report_lines.append("## 1. 困惑度对比\n")
    report_lines.append("| 模型 | 困惑度 |\n")
    report_lines.append("|------|--------|\n")
    for model_name, ppl in perplexities.items():
        report_lines.append(f"| {model_name.upper()} | {ppl:.2f} |\n")
    report_lines.append("\n")
    
    # 测试案例对比
    report_lines.append("## 2. 测试案例回答对比\n\n")
    
    for i, result in enumerate(results, 1):
        report_lines.append(f"### 测试案例 {i}\n\n")
        report_lines.append(f"**提示**: {result['prompt']}\n\n")
        
        for model_name in ['base', 'lora', 'full']:
            if model_name in result['responses']:
                response = result['responses'][model_name]
                report_lines.append(f"**{model_name.upper()} 模型回答**:\n")
                report_lines.append(f"{response}\n\n")
        
        report_lines.append("---\n\n")
    
    # 保存报告
    report_path = "./outputs/evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"✓ 评估报告已保存到: {report_path}")
    
    # ============= 7. 打印总结 =============
    print("\n" + "=" * 80)
    print("7. 评估总结")
    print("=" * 80)
    
    print("\n困惑度对比 (越低越好):")
    for model_name, ppl in sorted(perplexities.items(), key=lambda x: x[1]):
        print(f"  {model_name.upper():10s}: {ppl:8.2f}")
    
    if len(perplexities) >= 2:
        ppls_list = list(perplexities.values())
        improvement = ((max(ppls_list) - min(ppls_list)) / max(ppls_list)) * 100
        print(f"\n最大改进: {improvement:.1f}%")
    
    print("\n" + "=" * 80)
    print("✓ 步骤 4 完成！评估和对比完成。")
    print("=" * 80)
    
    print("\n查看详细结果:")
    print(f"  - JSON 格式: {OUTPUT_PATH}")
    print(f"  - Markdown 报告: {report_path}")
    
    print("\n建议的后续步骤:")
    print("  1. 查看评估报告，比较不同模型的表现")
    print("  2. 基于困惑度和回答质量选择最佳模型")
    print("  3. 如需要，调整超参数重新训练")
    print("  4. 在更多测试案例上验证模型性能")

if __name__ == "__main__":
    main()
