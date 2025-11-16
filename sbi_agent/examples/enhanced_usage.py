"""
增强的SBI Agent使用示例
展示增强的数据加载和参数管理功能
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from simple_sbi_agent import SimpleSBIAgent

def example_enhanced_data_loading():
    """增强数据加载示例"""
    print("=== 增强数据加载示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载增强配置
        print("2. 加载增强配置...")
        agent.load_socia_configs()
        
        # 获取模块依赖关系
        print("3. 分析模块依赖关系...")
        calibration_order = agent.get_calibration_order()
        print(f"校准顺序: {calibration_order}")
        
        for module in calibration_order:
            dependencies = agent.get_module_dependencies(module)
            print(f"模块 {module} 的依赖: {dependencies}")
        
        # 加载目标数据
        print("4. 加载目标数据...")
        agent.load_target_data()
        
        # 获取目标信号
        target_signals = agent.get_target_signals()
        print(f"目标信号: {list(target_signals.keys())}")
        for signal_name, signal_data in target_signals.items():
            print(f"  {signal_name}: {signal_data.shape}")
        
        print("增强数据加载示例完成!")
        
    except Exception as e:
        print(f"增强数据加载示例失败: {e}")

def example_enhanced_parameter_management():
    """增强参数管理示例"""
    print("\n=== 增强参数管理示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载增强配置
        print("2. 加载增强配置...")
        agent.load_socia_configs()
        
        # 获取校准顺序
        calibration_order = agent.get_calibration_order()
        print(f"校准顺序: {calibration_order}")
        
        # 分析每个模块的参数空间
        print("3. 分析模块参数空间...")
        for module_name in calibration_order:
            print(f"\n模块: {module_name}")
            
            # 获取参数空间
            param_space = agent.get_parameter_space_for_module(module_name)
            print(f"  参数数量: {param_space.get('parameter_count', 0)}")
            print(f"  参数名称: {param_space.get('parameter_names', [])}")
            
            # 获取参数边界
            param_bounds = agent.get_parameter_bounds_for_module(module_name)
            print(f"  参数边界: {len(param_bounds)} 个参数有边界约束")
            
            # 获取摘要统计设计
            summary_design = agent.get_summary_statistics_design(module_name)
            print(f"  摘要统计: {list(summary_design.keys())}")
        
        print("增强参数管理示例完成!")
        
    except Exception as e:
        print(f"增强参数管理示例失败: {e}")

def example_enhanced_calibration_strategy():
    """增强校准策略示例"""
    print("\n=== 增强校准策略示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载增强配置
        print("2. 加载增强配置...")
        agent.load_socia_configs()
        
        # 获取增强校准策略
        print("3. 获取增强校准策略...")
        enhanced_strategy = agent.get_enhanced_calibration_strategy()
        
        print("依赖关系分析:")
        dep_analysis = enhanced_strategy['dependency_analysis']
        print(f"  校准模块: {dep_analysis.get('calibration_modules', [])}")
        print(f"  校准顺序: {dep_analysis.get('calibration_order', [])}")
        
        print("\n摘要统计设计:")
        summary_stats = enhanced_strategy['summary_statistics']
        for module, stats in summary_stats.items():
            print(f"  {module}: {list(stats.keys())}")
        
        print("\n参数空间:")
        param_spaces = enhanced_strategy['parameter_spaces']
        for module, space in param_spaces.items():
            print(f"  {module}: {space.get('parameter_count', 0)} 个参数")
        
        print("增强校准策略示例完成!")
        
    except Exception as e:
        print(f"增强校准策略示例失败: {e}")

def example_parameter_validation():
    """参数验证示例"""
    print("\n=== 参数验证示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载增强配置
        print("2. 加载增强配置...")
        agent.load_socia_configs()
        
        # 获取校准顺序
        calibration_order = agent.get_calibration_order()
        
        # 测试参数验证
        print("3. 测试参数验证...")
        for module_name in calibration_order[:1]:  # 只测试第一个模块
            print(f"\n测试模块: {module_name}")
            
            # 获取参数边界
            param_bounds = agent.get_parameter_bounds_for_module(module_name)
            print(f"参数边界: {param_bounds}")
            
            # 创建测试参数
            test_params = {}
            for param_name, bounds in param_bounds.items():
                # 测试有效参数
                test_params[param_name] = (bounds[0] + bounds[1]) / 2
            
            # 验证参数
            is_valid, errors = agent.validate_module_parameters(test_params, module_name)
            print(f"参数验证结果: {'通过' if is_valid else '失败'}")
            if errors:
                print(f"错误信息: {errors}")
            
            # 测试无效参数
            invalid_params = test_params.copy()
            if param_bounds:
                first_param = list(param_bounds.keys())[0]
                invalid_params[first_param] = param_bounds[first_param][1] + 1  # 超出边界
            
            is_valid, errors = agent.validate_module_parameters(invalid_params, module_name)
            print(f"无效参数验证结果: {'通过' if is_valid else '失败'}")
            if errors:
                print(f"错误信息: {errors}")
        
        print("参数验证示例完成!")
        
    except Exception as e:
        print(f"参数验证示例失败: {e}")

def example_data_quality_analysis():
    """数据质量分析示例"""
    print("\n=== 数据质量分析示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载增强配置
        print("2. 加载增强配置...")
        agent.load_socia_configs()
        
        # 加载目标数据
        print("3. 加载目标数据...")
        agent.load_target_data()
        
        # 获取数据摘要
        print("4. 分析数据质量...")
        data_summary = agent.enhanced_data_processor.get_data_summary()
        
        print(f"目标数据形状: {data_summary['target_data_shape']}")
        print(f"日率数据形状: {data_summary['daily_rates_shape']}")
        
        quality_report = data_summary['data_quality']
        print(f"数据质量分数: {quality_report['quality_score']:.3f}")
        print(f"总行数: {quality_report['total_rows']}")
        print(f"缺失值: {quality_report['missing_values']}")
        
        # 获取目标信号
        target_signals = agent.get_target_signals()
        for signal_name, signal_data in target_signals.items():
            print(f"{signal_name}: 范围 [{signal_data.min():.3f}, {signal_data.max():.3f}], 均值 {signal_data.mean():.3f}")
        
        print("数据质量分析示例完成!")
        
    except Exception as e:
        print(f"数据质量分析示例失败: {e}")

def main():
    """主函数"""
    print("增强的SBI Agent使用示例")
    print("=" * 60)
    
    # 运行示例
    example_enhanced_data_loading()
    example_enhanced_parameter_management()
    example_enhanced_calibration_strategy()
    example_parameter_validation()
    example_data_quality_analysis()
    
    print("\n所有增强示例完成!")

if __name__ == "__main__":
    main()





