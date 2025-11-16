#!/usr/bin/env python3
"""
统一运行所有校准器的测试，使用test_data.csv作为测试窗口。
确保所有方法使用相同的测试窗口配置和k_runs。
"""

import sys
import os

# Add parent directory to path to import simulation modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_code_using_calibration_template_test_SBI import simulation_test, sbi_simulation_test

def run_all_calibrators_test():
    """运行所有校准器的测试，使用统一的测试窗口配置"""
    
    calibrators = [
        "logit_head",
        "random_search", 
        "bo_TuRBO",
        "bo_TuRBO_llm_guide",
        "bo_vanilla",
        "evo",
        "sbi"
    ]
    
    print("=" * 80)
    print("🚀 开始运行所有校准器的统一测试")
    print("=" * 80)
    print(f"测试窗口: test_data.csv (days 30-39)")
    print(f"k_runs: 20 (统一配置)")
    print(f"共 {len(calibrators)} 个校准器")
    print("=" * 80)
    
    results = {}
    
    for i, calibrator_type in enumerate(calibrators, 1):
        print(f"\n[{i}/{len(calibrators)}] 测试 {calibrator_type}...")
        print("-" * 80)
        
        try:
            if calibrator_type == "sbi":
                sbi_simulation_test()
            else:
                simulation_test(calibrator_type)
            results[calibrator_type] = "✅ 成功"
            print(f"✅ {calibrator_type} 测试完成")
        except Exception as e:
            results[calibrator_type] = f"❌ 失败: {str(e)}"
            print(f"❌ {calibrator_type} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    for calibrator_type, status in results.items():
        print(f"{calibrator_type:<25}: {status}")
    
    success_count = sum(1 for s in results.values() if "✅" in s)
    print(f"\n成功: {success_count}/{len(calibrators)}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)
    print("\n提示: 运行 compare_specific_calibrators.py 来比较结果")

if __name__ == "__main__":
    run_all_calibrators_test()

