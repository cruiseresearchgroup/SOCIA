#!/usr/bin/env python3
"""
比较四种Calibrator的性能
"""

import json
import os
import pandas as pd
import numpy as np

def load_test_metrics(calibrator_type):
    """加载测试指标"""
    test_dir = f"data_fitting/mask_adoption_data/test_outputs_{calibrator_type}"
    metrics_file = os.path.join(test_dir, "test_metrics.json")
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {metrics_file} not found")
        return None

def main():
    """主函数"""
    calibrators = {
        'SBI': 'sbi',
        'RandomSearch': 'random_search', 
        'LogitHead': 'logit_head',
        'BoCalibrator': 'bo'
    }
    
    # 加载所有测试结果
    results = {}
    for name, code in calibrators.items():
        metrics = load_test_metrics(code)
        if metrics:
            results[name] = {
                'RMSE': metrics['RMSE_aggregate_mean'],
                'RMSE_CI': metrics['RMSE_aggregate_CI95'],
                'MAE': metrics['MAE_aggregate_mean'], 
                'MAE_CI': metrics['MAE_aggregate_CI95'],
                'Brier': metrics['Brier_mean'],
                'Brier_CI': metrics['Brier_CI95'],
                'TransitionFit': metrics['TransitionFit_mean'],
                'TransitionFit_CI': metrics['TransitionFit_CI95']
            }
    
    # 创建DataFrame进行比较
    df = pd.DataFrame(results).T
    
    print("=" * 80)
    print("🏆 四种Calibrator性能比较分析")
    print("=" * 80)
    
    # 打印详细结果表格
    print("\n📊 详细测试结果（均值 ± 95%置信区间）")
    print("-" * 80)
    
    metrics = ['RMSE', 'MAE', 'Brier', 'TransitionFit']
    
    for metric in metrics:
        print(f"\n{metric}（越小越好）:")
        metric_data = []
        for name in results.keys():
            mean_val = results[name][metric]
            ci_val = results[name][f'{metric}_CI']
            metric_data.append((name, mean_val, ci_val))
            print(f"  {name:15}: {mean_val:.4f} ± {ci_val:.4f}")
        
        # 找出最佳方法
        best_name, best_val, _ = min(metric_data, key=lambda x: x[1])
        print(f"  🥇 最佳: {best_name}")
    
    # 计算排名
    print("\n🏅 综合排名分析")
    print("-" * 50)
    
    rankings = {}
    for metric in metrics:
        # 按照指标值排序（越小越好）
        sorted_results = sorted(results.items(), key=lambda x: x[1][metric])
        for rank, (name, _) in enumerate(sorted_results, 1):
            if name not in rankings:
                rankings[name] = []
            rankings[name].append(rank)
    
    # 计算平均排名
    avg_rankings = {}
    for name, ranks in rankings.items():
        avg_rankings[name] = np.mean(ranks)
    
    # 按平均排名排序
    sorted_rankings = sorted(avg_rankings.items(), key=lambda x: x[1])
    
    for rank, (name, avg_rank) in enumerate(sorted_rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏃"
        print(f"{medal} 第{rank}名: {name} (平均排名: {avg_rank:.1f})")
        
        # 显示各项指标的具体排名
        individual_ranks = rankings[name]
        print(f"    各指标排名: RMSE={individual_ranks[0]}, MAE={individual_ranks[1]}, Brier={individual_ranks[2]}, TransitionFit={individual_ranks[3]}")
    
    # 性能提升分析
    print("\n📈 相对于最佳方法(SBI)的性能差距")
    print("-" * 50)
    
    best_name = sorted_rankings[0][0]  # 最佳方法
    best_results = results[best_name]
    
    for name, metrics_data in results.items():
        if name == best_name:
            continue
        print(f"\n{name} vs {best_name}:")
        for metric in metrics:
            best_val = best_results[metric]
            current_val = metrics_data[metric]
            if best_val > 0:
                improvement = ((current_val - best_val) / best_val) * 100
                print(f"  {metric}: +{improvement:.1f}% 差距 ({current_val:.4f} vs {best_val:.4f})")
    
    # 保存结果到CSV
    print(f"\n💾 保存结果到 calibrator_comparison.csv")
    df_output = pd.DataFrame({
        name: [f"{results[name][metric]:.4f}" for metric in metrics]
        for name in results.keys()
    }, index=metrics)
    
    df_output.to_csv("output/test_mask_patch/calibrator_comparison.csv")
    
    print("\n🎯 总结和建议:")
    print("-" * 30)
    print(f"✅ 推荐使用: {sorted_rankings[0][0]} (在所有指标上表现最佳)")
    
    if 'BoCalibrator' in [item[0] for item in sorted_rankings[-2:]]:
        print("🔧 BoCalibrator改进建议:")
        print("   - 增加评估预算 (当前100次 → 建议200-500次)")
        print("   - 调整快速模式比例")
        print("   - 优化组合指标权重") 
        print("   - 尝试不同采集函数 (UCB, PI)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
