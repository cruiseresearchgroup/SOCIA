"""
仿真包装和SBI执行功能使用示例
展示SOCIA工作流集成、智能SBI策略、仿真包装、收敛监控等功能
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from simple_sbi_agent import SimpleSBIAgent

def example_socia_workflow_integration():
    """SOCIA工作流集成示例"""
    print("=== SOCIA工作流集成示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载SOCIA配置
        print("2. 加载SOCIA配置...")
        agent.load_socia_configs()
        
        # 获取工作流约束
        print("3. 获取工作流约束...")
        calibration_order = agent.get_calibration_order()
        print(f"校准顺序: {calibration_order}")
        
        for module_name in calibration_order:
            constraints = agent.get_workflow_constraints(module_name)
            print(f"模块 {module_name} 约束:")
            print(f"  参数约束: {len(constraints['parameter_constraints'])} 个")
            print(f"  收敛要求: {len(constraints['convergence_requirements'])} 个")
        
        # 开始迭代
        print("4. 开始迭代...")
        agent.start_iteration(1)
        
        # 模拟校准结果
        mock_results = {
            'calibration_results': {
                module: {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.1, 'mae': 0.05, 'r2': 0.9}
                }
                for module in calibration_order
            }
        }
        
        # 更新迭代
        agent.update_iteration(mock_results)
        
        # 生成反馈
        feedback = agent.generate_workflow_feedback(mock_results)
        print(f"反馈生成: {len(feedback.get('suggestions', []))} 个建议")
        
        print("SOCIA工作流集成示例完成!")
        
    except Exception as e:
        print(f"SOCIA工作流集成示例失败: {e}")

def example_intelligent_sbi_strategy():
    """智能SBI策略示例"""
    print("\n=== 智能SBI策略示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载SOCIA配置
        print("2. 加载SOCIA配置...")
        agent.load_socia_configs()
        
        # 获取校准模块
        calibration_order = agent.get_calibration_order()
        print(f"校准模块: {calibration_order}")
        
        # 设计SBI策略
        print("3. 设计SBI策略...")
        strategy = agent.design_sbi_strategy(calibration_order, data_complexity='medium')
        
        print("SBI方法选择:")
        for module_name, method_info in strategy['sbi_methods'].items():
            print(f"  {module_name}: {method_info['method']}")
            print(f"    参数: {list(method_info['parameters'].keys())}")
        
        print("\n联合校准策略:")
        for phase in strategy['calibration_phases']:
            print(f"  阶段: {phase['phase_type']}")
            print(f"    模块: {phase['modules']}")
            print(f"    描述: {phase['description']}")
        
        print("智能SBI策略示例完成!")
        
    except Exception as e:
        print(f"智能SBI策略示例失败: {e}")

def example_enhanced_simulation():
    """增强仿真示例"""
    print("\n=== 增强仿真示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载SOCIA配置
        print("2. 加载SOCIA配置...")
        agent.load_socia_configs()
        
        # 初始化仿真包装器
        print("3. 初始化仿真包装器...")
        agent.initialize_simulation_wrapper()
        
        # 运行单模块仿真
        print("4. 运行单模块仿真...")
        calibration_order = agent.get_calibration_order()
        if calibration_order:
            module_name = calibration_order[0]
            test_params = {
                'base_adoption_rate': 0.1,
                'beta_neighbors': 0.5,
                'gamma_info': 0.3
            }
            
            try:
                result = agent.run_enhanced_simulation(test_params, module_name)
                print(f"仿真结果: {list(result.keys())}")
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
            except Exception as e:
                print(f"仿真执行失败: {e}")
        
        # 运行多模块仿真
        print("5. 运行多模块仿真...")
        module_parameters = {
            module: {
                'base_adoption_rate': 0.1,
                'beta_neighbors': 0.5,
                'gamma_info': 0.3
            }
            for module in calibration_order[:2]  # 只测试前两个模块
        }
        
        try:
            results = agent.run_multi_module_simulation(
                module_parameters, calibration_order[:2], parallel=False
            )
            print(f"多模块仿真结果: {len(results)} 个模块")
            for module_name, result in results.items():
                print(f"  {module_name}: {'成功' if 'error' not in result else '失败'}")
        except Exception as e:
            print(f"多模块仿真失败: {e}")
        
        print("增强仿真示例完成!")
        
    except Exception as e:
        print(f"增强仿真示例失败: {e}")

def example_convergence_monitoring():
    """收敛监控示例"""
    print("\n=== 收敛监控示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载SOCIA配置
        print("2. 加载SOCIA配置...")
        agent.load_socia_configs()
        
        # 开始收敛监控
        print("3. 开始收敛监控...")
        calibration_order = agent.get_calibration_order()
        if calibration_order:
            module_name = calibration_order[0]
            agent.start_convergence_monitoring(module_name)
            
            # 模拟指标更新
            print("4. 模拟指标更新...")
            import numpy as np
            
            for i in range(10):
                metrics = {
                    'rmse': 0.5 * np.exp(-i * 0.1) + 0.1,
                    'mae': 0.3 * np.exp(-i * 0.1) + 0.05,
                    'r2': 0.5 + 0.4 * (1 - np.exp(-i * 0.1))
                }
                
                agent.update_convergence_metrics(metrics)
                
                convergence_status = agent.get_convergence_status()
                print(f"  迭代 {i+1}: RMSE={metrics['rmse']:.4f}, 收敛={convergence_status['convergence_summary']['overall_converged']}")
                
                if convergence_status['convergence_summary']['overall_converged']:
                    print("  收敛达成!")
                    break
        
        # 测试参数调整
        print("5. 测试参数调整...")
        current_bounds = {
            'base_adoption_rate': (0.0, 1.0),
            'beta_neighbors': (0.0, 2.0),
            'gamma_info': (0.0, 1.0)
        }
        
        calibration_results = {
            'convergence_achieved': False,
            'final_metrics': {'rmse': 0.3, 'mae': 0.15, 'r2': 0.7},
            'parameter_values': {
                'base_adoption_rate': 0.1,
                'beta_neighbors': 0.5,
                'gamma_info': 0.3
            }
        }
        
        should_adjust, adjustment = agent.should_adjust_parameters(
            module_name, current_bounds, calibration_results
        )
        
        if should_adjust:
            print(f"需要调整参数: {adjustment['strategy']}")
            print(f"调整原因: {adjustment['reason']}")
        else:
            print("不需要调整参数")
        
        print("收敛监控示例完成!")
        
    except Exception as e:
        print(f"收敛监控示例失败: {e}")

def example_restart_mechanism():
    """重启机制示例"""
    print("\n=== 重启机制示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载SOCIA配置
        print("2. 加载SOCIA配置...")
        agent.load_socia_configs()
        
        # 测试重启决策
        print("3. 测试重启决策...")
        failure_reasons = [
            "convergence_failure",
            "parameter_cancellation",
            "simulation_timeout",
            "memory_error",
            "other_error"
        ]
        
        calibration_results = {
            'convergence_achieved': False,
            'final_metrics': {'rmse': 0.5, 'mae': 0.25, 'r2': 0.6}
        }
        
        for reason in failure_reasons:
            should_restart, strategy = agent.should_restart_calibration(reason, calibration_results)
            print(f"失败原因: {reason}")
            print(f"  应该重启: {should_restart}")
            if should_restart:
                print(f"  重启策略: {strategy['restart_count']} 次重启")
                print(f"  参数调整: {list(strategy['parameter_adjustments'].keys())}")
                print(f"  方法变更: {list(strategy['method_changes'].keys())}")
            print()
        
        print("重启机制示例完成!")
        
    except Exception as e:
        print(f"重启机制示例失败: {e}")

def example_comprehensive_calibration():
    """综合校准示例"""
    print("\n=== 综合校准示例 ===")
    
    # 设置路径
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        print("1. 创建SBI Agent...")
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 加载SOCIA配置
        print("2. 加载SOCIA配置...")
        agent.load_socia_configs()
        
        # 获取校准策略
        print("3. 获取校准策略...")
        enhanced_strategy = agent.get_enhanced_calibration_strategy()
        print(f"增强校准策略包含: {list(enhanced_strategy.keys())}")
        
        # 开始迭代
        print("4. 开始迭代...")
        agent.start_iteration(1)
        
        # 模拟完整的校准流程
        print("5. 模拟完整校准流程...")
        calibration_order = agent.get_calibration_order()
        
        for i, module_name in enumerate(calibration_order[:2]):  # 只测试前两个模块
            print(f"\n校准模块 {i+1}: {module_name}")
            
            # 开始收敛监控
            agent.start_convergence_monitoring(module_name)
            
            # 获取参数空间
            param_space = agent.get_parameter_space_for_module(module_name)
            print(f"  参数空间: {param_space.get('parameter_count', 0)} 个参数")
            
            # 获取工作流约束
            constraints = agent.get_workflow_constraints(module_name)
            print(f"  工作流约束: {len(constraints['parameter_constraints'])} 个参数约束")
            
            # 模拟校准过程
            for iteration in range(5):
                metrics = {
                    'rmse': 0.5 * (0.9 ** iteration),
                    'mae': 0.25 * (0.9 ** iteration),
                    'r2': 0.6 + 0.3 * (1 - 0.9 ** iteration)
                }
                
                agent.update_convergence_metrics(metrics)
                
                convergence_status = agent.get_convergence_status()
                if convergence_status['convergence_summary']['overall_converged']:
                    print(f"    收敛达成 (迭代 {iteration+1})")
                    break
            
            # 模拟校准结果
            module_results = {
                'convergence_achieved': True,
                'final_metrics': {'rmse': 0.1, 'mae': 0.05, 'r2': 0.9},
                'calibrated_parameters': {
                    'base_adoption_rate': 0.15,
                    'beta_neighbors': 0.6,
                    'gamma_info': 0.4
                }
            }
            
            # 更新SBI策略
            agent.sbi_strategy.update_calibration_history(module_name, module_results)
        
        # 生成最终反馈
        print("6. 生成最终反馈...")
        final_results = {
            'calibration_results': {
                module: {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.1, 'mae': 0.05, 'r2': 0.9}
                }
                for module in calibration_order[:2]
            }
        }
        
        agent.update_iteration(final_results)
        feedback = agent.generate_workflow_feedback(final_results)
        
        print(f"反馈生成: {len(feedback.get('suggestions', []))} 个建议")
        print(f"下一步: {len(feedback.get('next_steps', []))} 个步骤")
        
        # 格式化并保存结果
        print("7. 格式化并保存结果...")
        agent.format_and_save_results(final_results)
        
        print("综合校准示例完成!")
        
    except Exception as e:
        print(f"综合校准示例失败: {e}")

def main():
    """主函数"""
    print("仿真包装和SBI执行功能使用示例")
    print("=" * 60)
    
    # 运行示例
    example_socia_workflow_integration()
    example_intelligent_sbi_strategy()
    example_enhanced_simulation()
    example_convergence_monitoring()
    example_restart_mechanism()
    example_comprehensive_calibration()
    
    print("\n所有仿真包装和SBI执行示例完成!")

if __name__ == "__main__":
    main()





