"""
测试和优化功能使用示例
展示SOCIA集成测试、模块化SBI测试、性能优化、端到端测试等功能
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from comprehensive_test_runner import ComprehensiveTestRunner, TestAndOptimizationManager

def example_socia_integration_tests():
    """SOCIA集成测试示例"""
    print("=== SOCIA集成测试示例 ===")
    
    # 设置测试数据路径
    test_data_dir = "output/mask_adoption_calibrasim_debug_run3"
    
    try:
        # 创建综合测试运行器
        print("1. 创建综合测试运行器...")
        test_runner = ComprehensiveTestRunner(test_data_dir)
        
        # 运行SOCIA集成测试
        print("2. 运行SOCIA集成测试...")
        socia_results = test_runner.test_runners['socia_integration'].run_integration_tests()
        
        print("SOCIA集成测试结果:")
        print(f"  总测试数: {socia_results['summary']['total_tests']}")
        print(f"  通过测试数: {socia_results['summary']['passed_tests']}")
        print(f"  失败测试数: {socia_results['summary']['failed_tests']}")
        print(f"  成功率: {socia_results['summary']['success_rate']:.2%}")
        
        # 显示详细结果
        for test_name, result in socia_results['detailed_results'].items():
            status = "✓" if result['status'] == 'passed' else "✗"
            print(f"  {status} {test_name}: {result['message']}")
        
        print("SOCIA集成测试示例完成!")
        
    except Exception as e:
        print(f"SOCIA集成测试示例失败: {e}")

def example_modular_sbi_tests():
    """模块化SBI测试示例"""
    print("\n=== 模块化SBI测试示例 ===")
    
    # 设置测试数据路径
    test_data_dir = "output/mask_adoption_calibrasim_debug_run3"
    
    try:
        # 创建综合测试运行器
        print("1. 创建综合测试运行器...")
        test_runner = ComprehensiveTestRunner(test_data_dir)
        
        # 运行模块化SBI测试
        print("2. 运行模块化SBI测试...")
        modular_results = test_runner.test_runners['modular_sbi'].run_modular_sbi_tests()
        
        print("模块化SBI测试结果:")
        print(f"  总测试数: {modular_results['summary']['total_tests']}")
        print(f"  通过测试数: {modular_results['summary']['passed_tests']}")
        print(f"  失败测试数: {modular_results['summary']['failed_tests']}")
        print(f"  成功率: {modular_results['summary']['success_rate']:.2%}")
        
        # 显示详细结果
        for test_name, result in modular_results['detailed_results'].items():
            status = "✓" if result['status'] == 'passed' else "✗"
            print(f"  {status} {test_name}: {result['message']}")
        
        print("模块化SBI测试示例完成!")
        
    except Exception as e:
        print(f"模块化SBI测试示例失败: {e}")

def example_end_to_end_tests():
    """端到端测试示例"""
    print("\n=== 端到端测试示例 ===")
    
    # 设置测试数据路径
    test_data_dir = "output/mask_adoption_calibrasim_debug_run3"
    
    try:
        # 创建综合测试运行器
        print("1. 创建综合测试运行器...")
        test_runner = ComprehensiveTestRunner(test_data_dir)
        
        # 运行端到端测试
        print("2. 运行端到端测试...")
        e2e_results = test_runner.test_runners['end_to_end'].run_end_to_end_tests()
        
        print("端到端测试结果:")
        print(f"  总测试数: {e2e_results['summary']['total_tests']}")
        print(f"  通过测试数: {e2e_results['summary']['passed_tests']}")
        print(f"  失败测试数: {e2e_results['summary']['failed_tests']}")
        print(f"  成功率: {e2e_results['summary']['success_rate']:.2%}")
        
        # 显示性能基准
        if 'performance_benchmarks' in e2e_results:
            print("\n性能基准:")
            for test_name, benchmarks in e2e_results['performance_benchmarks'].items():
                print(f"  {test_name}:")
                print(f"    总时间: {benchmarks.get('total_time', 0):.2f}秒")
                print(f"    配置加载: {benchmarks.get('config_load_time', 0):.2f}秒")
                print(f"    数据加载: {benchmarks.get('data_load_time', 0):.2f}秒")
                print(f"    校准时间: {benchmarks.get('calibration_time', 0):.2f}秒")
        
        # 显示详细结果
        for test_name, result in e2e_results['detailed_results'].items():
            status = "✓" if result['status'] == 'passed' else "✗"
            print(f"  {status} {test_name}: {result['message']}")
        
        print("端到端测试示例完成!")
        
    except Exception as e:
        print(f"端到端测试示例失败: {e}")

def example_performance_optimization():
    """性能优化示例"""
    print("\n=== 性能优化示例 ===")
    
    try:
        # 创建测试和优化管理器
        print("1. 创建测试和优化管理器...")
        test_data_dir = "output/mask_adoption_calibrasim_debug_run3"
        manager = TestAndOptimizationManager(test_data_dir)
        
        # 运行性能优化
        print("2. 运行性能优化...")
        optimization_results = manager._run_performance_optimization()
        
        print("性能优化结果:")
        
        # 内存优化
        memory_opt = optimization_results.get('memory_optimization', {})
        if memory_opt:
            print(f"  内存优化: {'成功' if memory_opt.get('optimization_successful', False) else '失败'}")
            print(f"  释放内存: {memory_opt.get('memory_freed', 0) / 1024 / 1024:.2f} MB")
            print(f"  优化操作: {', '.join(memory_opt.get('actions_taken', []))}")
        
        # 执行优化
        execution_opt = optimization_results.get('execution_optimization', {})
        if execution_opt:
            print(f"  执行优化: 改进 {execution_opt.get('performance_improvement', 0):.1%}")
            print(f"  应用优化: {', '.join(execution_opt.get('optimizations_applied', []))}")
        
        # 错误处理
        error_handling = optimization_results.get('error_handling', {})
        if error_handling:
            print(f"  错误统计: {error_handling.get('total_errors', 0)} 个错误")
            print(f"  最常见错误: {error_handling.get('most_common_error', '无')}")
        
        # 优化建议
        recommendations = optimization_results.get('recommendations', [])
        if recommendations:
            print("\n优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("性能优化示例完成!")
        
    except Exception as e:
        print(f"性能优化示例失败: {e}")

def example_comprehensive_testing():
    """综合测试示例"""
    print("\n=== 综合测试示例 ===")
    
    # 设置测试数据路径
    test_data_dir = "output/mask_adoption_calibrasim_debug_run3"
    
    try:
        # 创建综合测试运行器
        print("1. 创建综合测试运行器...")
        test_runner = ComprehensiveTestRunner(test_data_dir)
        
        # 运行所有测试
        print("2. 运行所有测试...")
        comprehensive_results = test_runner.run_all_tests()
        
        print("综合测试结果:")
        print(f"  总执行时间: {comprehensive_results.get('report_info', {}).get('total_execution_time', 0):.2f}秒")
        
        # 整体摘要
        overall_summary = comprehensive_results.get('overall_summary', {})
        print(f"  总测试数: {overall_summary.get('total_tests', 0)}")
        print(f"  通过测试数: {overall_summary.get('total_passed', 0)}")
        print(f"  失败测试数: {overall_summary.get('total_failed', 0)}")
        print(f"  整体成功率: {overall_summary.get('overall_success_rate', 0):.2%}")
        
        # 各类型测试结果
        test_results = comprehensive_results.get('test_results', {})
        for test_type, results in test_results.items():
            summary = results.get('summary', {})
            print(f"\n{test_type} 测试:")
            print(f"  测试数: {summary.get('total_tests', 0)}")
            print(f"  通过数: {summary.get('passed_tests', 0)}")
            print(f"  成功率: {summary.get('success_rate', 0):.2%}")
        
        # 性能摘要
        performance_summary = comprehensive_results.get('performance_summary', {})
        if performance_summary:
            print("\n性能摘要:")
            memory_status = performance_summary.get('memory_status', {})
            if memory_status:
                print(f"  内存使用率: {memory_status.get('memory_percent', 0):.1f}%")
                print(f"  可用内存: {memory_status.get('available_memory', 0) / 1024 / 1024 / 1024:.2f} GB")
        
        # 建议
        recommendations = comprehensive_results.get('recommendations', [])
        if recommendations:
            print("\n建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # 下一步
        next_steps = comprehensive_results.get('next_steps', [])
        if next_steps:
            print("\n下一步:")
            for i, step in enumerate(next_steps, 1):
                print(f"  {i}. {step}")
        
        print("综合测试示例完成!")
        
    except Exception as e:
        print(f"综合测试示例失败: {e}")

def example_regression_testing():
    """回归测试示例"""
    print("\n=== 回归测试示例 ===")
    
    # 设置测试数据路径
    test_data_dir = "output/mask_adoption_calibrasim_debug_run3"
    
    try:
        # 创建综合测试运行器
        print("1. 创建综合测试运行器...")
        test_runner = ComprehensiveTestRunner(test_data_dir)
        
        # 运行基线测试
        print("2. 运行基线测试...")
        baseline_results = test_runner.run_all_tests()
        
        # 运行回归测试
        print("3. 运行回归测试...")
        regression_results = test_runner.run_regression_tests(baseline_results)
        
        print("回归测试结果:")
        
        # 回归分析
        regression_analysis = regression_results.get('regression_analysis', {})
        print(f"  整体状态: {regression_analysis.get('overall_status', 'unknown')}")
        
        # 性能回归
        performance_regression = regression_analysis.get('performance_regression', {})
        if performance_regression:
            print("\n性能回归:")
            for metric, data in performance_regression.items():
                print(f"  {metric}:")
                print(f"    基线: {data.get('baseline', 0):.2f}")
                print(f"    当前: {data.get('current', 0):.2f}")
                print(f"    回归: {'是' if data.get('regression', False) else '否'}")
        
        # 功能回归
        functionality_regression = regression_analysis.get('functionality_regression', {})
        if functionality_regression:
            print("\n功能回归:")
            for test_type, data in functionality_regression.items():
                print(f"  {test_type}:")
                print(f"    基线成功率: {data.get('baseline_success_rate', 0):.2%}")
                print(f"    当前成功率: {data.get('current_success_rate', 0):.2%}")
                print(f"    回归: {'是' if data.get('regression', False) else '否'}")
        
        print("回归测试示例完成!")
        
    except Exception as e:
        print(f"回归测试示例失败: {e}")

def main():
    """主函数"""
    print("测试和优化功能使用示例")
    print("=" * 60)
    
    # 运行示例
    example_socia_integration_tests()
    example_modular_sbi_tests()
    example_end_to_end_tests()
    example_performance_optimization()
    example_comprehensive_testing()
    example_regression_testing()
    
    print("\n所有测试和优化示例完成!")

if __name__ == "__main__":
    main()





