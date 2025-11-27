"""
一键运行所有训练和评估步骤
可以选择性地运行某些步骤
"""
import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """运行 Python 脚本"""
    print("\n" + "=" * 80)
    print(f"开始: {description}")
    print(f"脚本: {script_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print("\n" + "!" * 80)
        print(f"错误: {script_name} 执行失败 (返回码: {result.returncode})")
        print("!" * 80)
        return False
    
    print("\n" + "=" * 80)
    print(f"✓ 完成: {description}")
    print("=" * 80 + "\n")
    return True

def main():
    print("=" * 80)
    print("SFT 项目 - 完整训练流程")
    print("=" * 80)
    
    # 检查必要的文件
    required_files = [
        "1_prepare_dataset.py",
        "2_lora_finetuning.py",
        "3_full_finetuning.py",
        "4_evaluate_compare.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n错误: 缺少以下文件:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请确保所有脚本文件都在当前目录中。")
        return
    
    # 询问要运行哪些步骤
    print("\n选择要运行的步骤:")
    print("1. 准备数据集")
    print("2. LoRA 微调 (30min-1hr)")
    print("3. 全量微调 (2+ hrs)")
    print("4. 评估和对比")
    print("5. 运行全部")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    steps = []
    
    if choice == "1":
        steps = [
            ("1_prepare_dataset.py", "步骤 1: 准备数据集")
        ]
    elif choice == "2":
        steps = [
            ("1_prepare_dataset.py", "步骤 1: 准备数据集"),
            ("2_lora_finetuning.py", "步骤 2: LoRA 微调")
        ]
    elif choice == "3":
        steps = [
            ("1_prepare_dataset.py", "步骤 1: 准备数据集"),
            ("3_full_finetuning.py", "步骤 3: 全量微调")
        ]
    elif choice == "4":
        steps = [
            ("4_evaluate_compare.py", "步骤 4: 评估和对比")
        ]
    elif choice == "5":
        steps = [
            ("1_prepare_dataset.py", "步骤 1: 准备数据集"),
            ("2_lora_finetuning.py", "步骤 2: LoRA 微调"),
            ("3_full_finetuning.py", "步骤 3: 全量微调"),
            ("4_evaluate_compare.py", "步骤 4: 评估和对比")
        ]
    else:
        print("无效的选项！")
        return
    
    # 确认
    print("\n将要运行的步骤:")
    for i, (script, desc) in enumerate(steps, 1):
        print(f"  {i}. {desc}")
    
    confirm = input("\n确认运行? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n总流程开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行步骤
    success_count = 0
    for script, description in steps:
        success = run_script(script, description)
        if success:
            success_count += 1
        else:
            print(f"\n停止执行，因为 {script} 失败。")
            break
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("流程完成总结")
    print("=" * 80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print(f"成功步骤: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("\n✓ 所有步骤执行成功！")
        print("\n查看结果:")
        print("  - 数据: ./data/")
        print("  - LoRA 模型: ./outputs/lora_model/")
        print("  - 全量模型: ./outputs/full_model/")
        print("  - 评估结果: ./outputs/evaluation_results.json")
        print("  - 评估报告: ./outputs/evaluation_report.md")
    else:
        print("\n✗ 部分步骤执行失败。")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
