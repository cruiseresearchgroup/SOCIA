#!/usr/bin/env python3
"""
三校准器对比分析脚本：LogitHead vs RandomSearch vs SBI
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

# 添加路径以导入SBI校准器
sys.path.append('output/test_mask_patch')

def load_test_results(calibrator_name: str) -> Dict[str, Any]:
    """加载校准器的测试结果"""
    
    if calibrator_name == "sbi":
        test_dir = "data_fitting/mask_adoption_data/test_outputs_sbi"
    else:
        test_dir = f"data_fitting/mask_adoption_data/test_outputs_{calibrator_name}"
    
    results = {}
    
    # 加载测试指标
    metrics_path = os.path.join(test_dir, "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
    
    # 加载配置
    config_path = os.path.join(test_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # 加载SBI特有的后验分析
    if calibrator_name == "sbi":
        posterior_path = os.path.join(test_dir, "posterior_analysis.json")
        if os.path.exists(posterior_path):
            with open(posterior_path, 'r') as f:
                results['posterior_analysis'] = json.load(f)
    
    return results

def run_sbi_test_with_real_pytorch():
    """使用真正的PyTorch运行SBI测试"""
    
    print("=== 使用真正的PyTorch重新运行SBI测试 ===\n")
    
    try:
        # 导入必要的库
        from simulation_code_using_calibration_template_test import sbi_simulation_test
        import torch
        import sbi
        
        print(f"✅ PyTorch {torch.__version__} 和 SBI {sbi.__version__} 已加载")
        print("🚀 运行SBI双重Monte Carlo测试...")
        
        # 运行SBI测试
        sbi_simulation_test()
        
        print("✅ SBI测试完成！")
        
    except Exception as e:
        print(f"❌ SBI测试失败: {e}")
        return False
    
    return True

def compare_calibrators():
    """比较三种校准器的性能"""
    
    print("\n" + "="*80)
    print("三校准器性能对比分析")
    print("="*80)
    
    calibrators = ["logit_head", "random_search", "sbi"]
    results = {}
    
    # 加载所有结果
    for cal in calibrators:
        print(f"\n📊 加载 {cal.upper()} 校准器结果...")
        results[cal] = load_test_results(cal)
        
        if 'metrics' in results[cal]:
            print(f"   ✅ 测试指标已加载")
        else:
            print(f"   ❌ 测试指标缺失")
    
    # 创建对比表格
    print(f"\n{'='*100}")
    print("性能指标对比")
    print(f"{'='*100}")
    
    # 表头
    print(f"{'指标':<15} {'LogitHead':<20} {'RandomSearch':<20} {'SBI (双重MC)':<25} {'最佳':<10}")
    print(f"{'-'*100}")
    
    # 比较指标
    metrics_to_compare = [
        ("RMSE", "RMSE_aggregate_mean", "RMSE_aggregate_CI95"),
        ("MAE", "MAE_aggregate_mean", "MAE_aggregate_CI95"),
        ("Brier", "Brier_mean", "Brier_CI95"),
        ("TransitionFit", "TransitionFit_mean", "TransitionFit_CI95")
    ]
    
    best_counts = {"logit_head": 0, "random_search": 0, "sbi": 0}
    
    for metric_name, mean_key, ci_key in metrics_to_compare:
        values = {}
        
        for cal in calibrators:
            if 'metrics' in results[cal] and mean_key in results[cal]['metrics']:
                mean_val = results[cal]['metrics'][mean_key]
                ci_val = results[cal]['metrics'][ci_key]
                values[cal] = (mean_val, ci_val)
            else:
                values[cal] = (None, None)
        
        # 找到最佳值（最小值）
        valid_values = {k: v[0] for k, v in values.items() if v[0] is not None}
        if valid_values:
            best_cal = min(valid_values.keys(), key=lambda x: valid_values[x])
            best_counts[best_cal] += 1
        else:
            best_cal = "N/A"
        
        # 打印行
        row = f"{metric_name:<15}"
        for cal in calibrators:
            if values[cal][0] is not None:
                mean_val, ci_val = values[cal]
                if cal == best_cal:
                    row += f" 🏆{mean_val:.4f}±{ci_val:.4f}"
                else:
                    row += f"   {mean_val:.4f}±{ci_val:.4f}"
                row += f"{'':>5}"
            else:
                row += f" {'N/A':<20}"
        
        row += f" {best_cal:<10}"
        print(row)
    
    # 总体最佳校准器
    print(f"{'-'*100}")
    overall_best = max(best_counts.keys(), key=lambda x: best_counts[x])
    print(f"{'总体最佳':<15} {overall_best.upper():<20} (获胜 {best_counts[overall_best]}/4 项指标)")
    
    # 详细分析
    print(f"\n{'='*80}")
    print("详细分析")
    print(f"{'='*80}")
    
    for cal in calibrators:
        if 'metrics' not in results[cal]:
            continue
            
        metrics = results[cal]['metrics']
        print(f"\n📈 {cal.upper()} 校准器:")
        print(f"   仿真次数: {metrics.get('k_runs', 'N/A')}")
        
        if cal == "sbi" and 'sbi_info' in metrics:
            sbi_info = metrics['sbi_info']
            print(f"   后验样本: {sbi_info.get('n_posterior_samples', 'N/A')}")
            print(f"   每样本运行: {sbi_info.get('k_runs_per_sample', 'N/A')}")
            print(f"   方法: {sbi_info.get('method', 'N/A')}")
            print(f"   不确定性量化: {sbi_info.get('parameter_uncertainty_quantified', 'N/A')}")
            
            # 显示参数不确定性信息
            if cal in results and 'posterior_analysis' in results[cal]:
                post_analysis = results[cal]['posterior_analysis']
                print(f"   参数不确定性:")
                print(f"     RMSE std: {post_analysis['parameter_uncertainty']['rmse_std']:.4f}")
                print(f"     最佳参数RMSE: {post_analysis['posterior_parameter_stats']['rmse_range'][0]:.4f}")
                print(f"     最差参数RMSE: {post_analysis['posterior_parameter_stats']['rmse_range'][1]:.4f}")
        
        # 显示性能特点
        rmse = metrics.get('RMSE_aggregate_mean', 0)
        if rmse < 0.25:
            performance = "🟢 优秀"
        elif rmse < 0.35:
            performance = "🟡 良好"
        else:
            performance = "🔴 需改进"
        
        print(f"   性能评级: {performance}")
    
    # 方法特点分析
    print(f"\n{'='*80}")
    print("方法特点分析")
    print(f"{'='*80}")
    
    print(f"🎯 LogitHead校准器:")
    print(f"   ✅ 快速训练（逻辑回归）")
    print(f"   ✅ 可解释性强")
    print(f"   ❌ 只拟合决策层，信息传播参数固定")
    print(f"   ❌ 无不确定性量化")
    
    print(f"\n🎲 RandomSearch校准器:")
    print(f"   ✅ 全参数空间搜索")
    print(f"   ✅ 性能表现最佳")
    print(f"   ❌ 计算量大，无理论保证")
    print(f"   ❌ 无不确定性量化")
    
    print(f"\n🧠 SBI校准器:")
    print(f"   ✅ 理论严谨的贝叶斯推理")
    print(f"   ✅ 完整的不确定性量化")
    print(f"   ✅ 参数分布而非点估计")
    print(f"   ✅ 可重用的神经后验估计器")
    print(f"   ❌ 计算复杂度高")
    print(f"   ❌ 需要大量训练数据")
    
    # 使用建议
    print(f"\n{'='*80}")
    print("使用建议")
    print(f"{'='*80}")
    
    print(f"🚀 快速原型和探索: LogitHead")
    print(f"🎯 追求最佳性能: RandomSearch") 
    print(f"🔬 科学研究和不确定性分析: SBI")
    print(f"💡 生产部署: 根据具体需求选择")

def main():
    """主函数"""
    print("🔬 三校准器对比分析")
    print("="*50)
    
    # 检查是否需要重新运行SBI测试
    sbi_results_exist = os.path.exists("data_fitting/mask_adoption_data/test_outputs_sbi/test_metrics.json")
    
    if not sbi_results_exist:
        print("⚠️ SBI测试结果不存在，正在运行SBI测试...")
        if not run_sbi_test_with_real_pytorch():
            print("❌ SBI测试失败，将使用现有结果进行对比")
    else:
        print("✅ 发现现有SBI测试结果")
        
        # 询问是否重新运行（通过检查checkpoint是否更新）
        checkpoint_dir = "data_fitting/mask_adoption_data/outputs_SBICalibrator_K5"
        if os.path.exists(os.path.join(checkpoint_dir, "posterior_estimator.pt")):
            print("🧠 发现真正的神经后验估计器，重新运行SBI测试以获得更准确结果...")
            run_sbi_test_with_real_pytorch()
    
    # 进行对比分析
    compare_calibrators()

if __name__ == "__main__":
    main()
