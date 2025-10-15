"""
SBI Agent使用示例
展示如何使用SBI Agent进行SOCIA模块校准
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from simple_sbi_agent import SimpleSBIAgent

def example_basic_usage():
    """基础使用示例"""
    print("=== SBI Agent基础使用示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 查看摘要信息
        print("2. 查看摘要信息...")
        summary = agent.get_summary()
        print(f"输出目录: {summary['output_dir']}")
        print(f"数据目录: {summary['data_dir']}")
        print(f"参数数量: {summary['parameter_count']}")
        print(f"目标数据形状: {summary['target_data_shape']}")
        
        # 加载参数
        print("3. 加载参数...")
        agent.load_parameters()
        print(f"当前参数: {list(agent.current_params.keys())[:5]}...")  # 显示前5个参数
        
        # 加载目标数据
        print("4. 加载目标数据...")
        agent.load_target_data()
        print(f"目标数据形状: {agent.target_data.shape}")
        
        print("基础使用示例完成!")
        
    except Exception as e:
        print(f"基础使用示例失败: {e}")

def example_calibration():
    """校准示例"""
    print("\n=== SBI Agent校准示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 执行校准
        print("2. 执行SBI校准...")
        results = agent.calibrate()
        
        # 查看结果
        print("3. 查看校准结果...")
        for module_name, module_results in results['calibration_results'].items():
            print(f"模块 {module_name}:")
            print(f"  收敛状态: {module_results.get('convergence_achieved', False)}")
            print(f"  最终指标: {module_results.get('final_metrics', {})}")
        
        print("校准示例完成!")
        
    except Exception as e:
        print(f"校准示例失败: {e}")

def example_module_calibration():
    """单模块校准示例"""
    print("\n=== 单模块校准示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 获取校准策略
        print("2. 获取校准策略...")
        strategy = agent.calibration_strategy
        print(f"校准顺序: {strategy.get('calibration_order', [])}")
        print(f"可校准模块: {strategy.get('calibration_modules', [])}")
        
        # 校准单个模块
        if strategy.get('calibration_modules'):
            module_name = strategy['calibration_modules'][0]
            print(f"3. 校准模块: {module_name}")
            module_results = agent.calibrate_module(module_name)
            print(f"模块校准结果: {module_results}")
        
        print("单模块校准示例完成!")
        
    except Exception as e:
        print(f"单模块校准示例失败: {e}")

def example_data_analysis():
    """数据分析示例"""
    print("\n=== 数据分析示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载目标数据
        print("2. 加载目标数据...")
        agent.load_target_data()
        
        # 分析数据质量
        print("3. 分析数据质量...")
        quality_report = agent.data_processor.validate_data_quality(agent.target_data)
        print(f"数据质量分数: {quality_report['quality_score']:.3f}")
        print(f"缺失值: {quality_report['missing_values']}")
        
        # 计算日率
        print("4. 计算日率...")
        daily_rates = agent.data_processor.calculate_daily_rates(agent.target_data)
        print(f"日率数据形状: {daily_rates.shape}")
        print(f"采纳率范围: {daily_rates['adoption_rate'].min():.3f} - {daily_rates['adoption_rate'].max():.3f}")
        print(f"信息率范围: {daily_rates['info_rate'].min():.3f} - {daily_rates['info_rate'].max():.3f}")
        
        print("数据分析示例完成!")
        
    except Exception as e:
        print(f"数据分析示例失败: {e}")

def main():
    """主函数"""
    print("SBI Agent使用示例")
    print("=" * 50)
    
    # 运行示例
    example_basic_usage()
    example_data_analysis()
    example_module_calibration()
    example_calibration()
    
    print("\n所有示例完成!")

if __name__ == "__main__":
    main()





