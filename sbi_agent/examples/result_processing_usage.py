"""
结果处理和验证功能使用示例
展示SOCIA结果格式集成、智能校准验证、结果保存和可视化、性能分析等功能
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from simple_sbi_agent import SimpleSBIAgent

def example_socia_result_integration():
    """SOCIA结果格式集成示例"""
    print("=== SOCIA结果格式集成示例 ===")
    
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
        
        # 模拟校准结果
        print("3. 模拟校准结果...")
        mock_calibration_results = {
            'calibration_results': {
                'InformationDiffusion': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85},
                    'calibrated_parameters': {
                        'base_info_rate': 0.12,
                        'info_decay_rate': 0.05,
                        'info_spread_probability': 0.3
                    }
                },
                'PolicyAndMessaging': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88},
                    'calibrated_parameters': {
                        'policy_effect_size': 0.25,
                        'messaging_frequency': 0.4,
                        'compliance_rate': 0.7
                    }
                },
                'SocialInfluenceAdoption': {
                    'convergence_achieved': False,
                    'final_metrics': {'rmse': 0.35, 'mae': 0.18, 'r2': 0.65},
                    'calibrated_parameters': {
                        'base_adoption_rate': 0.08,
                        'beta_neighbors': 0.6,
                        'gamma_info': 0.4
                    }
                }
            },
            'calibration_time': 1800
        }
        
        # 保存校准参数
        print("4. 保存校准参数...")
        params_path = agent.save_calibrated_parameters(mock_calibration_results)
        print(f"校准参数保存完成: {params_path}")
        
        # 保存校准报告
        print("5. 保存校准报告...")
        report_path = agent.save_calibration_report(mock_calibration_results)
        print(f"校准报告保存完成: {report_path}")
        
        # 保存反馈
        print("6. 保存反馈...")
        feedback_path = agent.save_feedback(mock_calibration_results, iteration_id=1)
        print(f"反馈保存完成: {feedback_path}")
        
        # 保存验证结果
        print("7. 保存验证结果...")
        verification_path = agent.save_verification_results(mock_calibration_results)
        print(f"验证结果保存完成: {verification_path}")
        
        print("SOCIA结果格式集成示例完成!")
        
    except Exception as e:
        print(f"SOCIA结果格式集成示例失败: {e}")

def example_intelligent_calibration_validation():
    """智能校准验证示例"""
    print("\n=== 智能校准验证示例 ===")
    
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
        
        # 模拟校准结果
        print("3. 模拟校准结果...")
        mock_calibration_results = {
            'calibration_results': {
                'InformationDiffusion': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85}
                },
                'PolicyAndMessaging': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88}
                },
                'SocialInfluenceAdoption': {
                    'convergence_achieved': False,
                    'final_metrics': {'rmse': 0.35, 'mae': 0.18, 'r2': 0.65}
                }
            }
        }
        
        # 验证校准结果
        print("4. 验证校准结果...")
        validation_output = agent.validate_calibration_results(mock_calibration_results)
        
        print("验证结果:")
        print(f"  整体状态: {validation_output['validation_results']['overall_status']}")
        print(f"  质量指标: {validation_output['quality_assessment']['overall_score']:.2f}")
        
        # 处理校准失败
        print("5. 处理校准失败...")
        failure_handling = agent.handle_calibration_failure(
            'convergence_failure', 
            mock_calibration_results,
            {'error_message': 'Module SocialInfluenceAdoption failed to converge'}
        )
        
        print("失败处理建议:")
        for suggestion in failure_handling.get('recommendations', []):
            print(f"  - {suggestion}")
        
        print("智能校准验证示例完成!")
        
    except Exception as e:
        print(f"智能校准验证示例失败: {e}")

def example_result_visualization():
    """结果可视化示例"""
    print("\n=== 结果可视化示例 ===")
    
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
        
        # 模拟校准结果
        print("3. 模拟校准结果...")
        mock_calibration_results = {
            'calibration_results': {
                'InformationDiffusion': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85}
                },
                'PolicyAndMessaging': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88}
                },
                'SocialInfluenceAdoption': {
                    'convergence_achieved': False,
                    'final_metrics': {'rmse': 0.35, 'mae': 0.18, 'r2': 0.65}
                }
            }
        }
        
        # 创建目标数据
        print("4. 创建目标数据...")
        target_data = pd.DataFrame({
            'day': range(30),
            'wearing_mask': np.random.beta(2, 5, 30),
            'received_info': np.random.beta(3, 2, 30)
        })
        
        # 创建校准可视化
        print("5. 创建校准可视化...")
        visualization_paths = agent.create_calibration_visualizations(
            mock_calibration_results, target_data
        )
        print(f"校准可视化创建完成: {len(visualization_paths)} 个图片")
        
        # 创建参数分析图
        print("6. 创建参数分析图...")
        parameter_analysis_paths = agent.create_parameter_analysis_plots(mock_calibration_results)
        print(f"参数分析图创建完成: {len(parameter_analysis_paths)} 个图片")
        
        print("结果可视化示例完成!")
        
    except Exception as e:
        print(f"结果可视化示例失败: {e}")

def example_performance_analysis():
    """性能分析示例"""
    print("\n=== 性能分析示例 ===")
    
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
        
        # 模拟校准结果
        print("3. 模拟校准结果...")
        mock_calibration_results = {
            'calibration_results': {
                'InformationDiffusion': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85}
                },
                'PolicyAndMessaging': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88}
                },
                'SocialInfluenceAdoption': {
                    'convergence_achieved': False,
                    'final_metrics': {'rmse': 0.35, 'mae': 0.18, 'r2': 0.65}
                }
            }
        }
        
        # 分析校准性能
        print("4. 分析校准性能...")
        performance_analysis = agent.analyze_calibration_performance(mock_calibration_results)
        
        print("性能分析结果:")
        print(f"  整体性能分数: {performance_analysis['overall_performance']['score']:.2f}")
        print(f"  性能等级: {performance_analysis['overall_performance']['level']}")
        print(f"  收敛率: {performance_analysis['overall_performance']['convergence_rate']:.2f}")
        
        # 生成优化建议
        print("5. 生成优化建议...")
        optimization_advice = agent.generate_optimization_advice(mock_calibration_results, performance_analysis)
        
        print("优化建议:")
        print(f"  整体优化策略: {optimization_advice['overall_optimization']['strategy']}")
        print(f"  重点领域: {optimization_advice['overall_optimization']['focus_areas']}")
        
        # 生成优化报告
        print("6. 生成优化报告...")
        optimization_report_path = agent.generate_optimization_report(performance_analysis, optimization_advice)
        print(f"优化报告生成完成: {optimization_report_path}")
        
        print("性能分析示例完成!")
        
    except Exception as e:
        print(f"性能分析示例失败: {e}")

def example_comprehensive_result_processing():
    """综合结果处理示例"""
    print("\n=== 综合结果处理示例 ===")
    
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
        
        # 模拟校准结果
        print("3. 模拟校准结果...")
        mock_calibration_results = {
            'calibration_results': {
                'InformationDiffusion': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85},
                    'calibrated_parameters': {
                        'base_info_rate': 0.12,
                        'info_decay_rate': 0.05,
                        'info_spread_probability': 0.3
                    }
                },
                'PolicyAndMessaging': {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88},
                    'calibrated_parameters': {
                        'policy_effect_size': 0.25,
                        'messaging_frequency': 0.4,
                        'compliance_rate': 0.7
                    }
                },
                'SocialInfluenceAdoption': {
                    'convergence_achieved': False,
                    'final_metrics': {'rmse': 0.35, 'mae': 0.18, 'r2': 0.65},
                    'calibrated_parameters': {
                        'base_adoption_rate': 0.08,
                        'beta_neighbors': 0.6,
                        'gamma_info': 0.4
                    }
                }
            },
            'calibration_time': 1800
        }
        
        # 创建目标数据
        print("4. 创建目标数据...")
        target_data = pd.DataFrame({
            'day': range(30),
            'wearing_mask': np.random.beta(2, 5, 30),
            'received_info': np.random.beta(3, 2, 30)
        })
        
        # 综合结果处理
        print("5. 综合结果处理...")
        processing_results = agent.process_and_validate_results(mock_calibration_results, target_data)
        
        print("处理结果摘要:")
        print(f"  验证输出: {len(processing_results['validation_output'])} 个组件")
        print(f"  可视化路径: {len(processing_results['visualization_paths'])} 个图片")
        print(f"  性能分析: {len(processing_results['performance_analysis'])} 个指标")
        print(f"  优化建议: {len(processing_results['optimization_advice'])} 个建议")
        print(f"  保存路径: {len(processing_results['saved_paths'])} 个文件")
        
        # 显示保存的文件
        print("\n保存的文件:")
        for file_type, file_path in processing_results['saved_paths'].items():
            print(f"  {file_type}: {file_path}")
        
        # 显示可视化文件
        print("\n可视化文件:")
        for i, viz_path in enumerate(processing_results['visualization_paths']):
            print(f"  图片 {i+1}: {viz_path}")
        
        print("综合结果处理示例完成!")
        
    except Exception as e:
        print(f"综合结果处理示例失败: {e}")

def main():
    """主函数"""
    print("结果处理和验证功能使用示例")
    print("=" * 60)
    
    # 运行示例
    example_socia_result_integration()
    example_intelligent_calibration_validation()
    example_result_visualization()
    example_performance_analysis()
    example_comprehensive_result_processing()
    
    print("\n所有结果处理和验证示例完成!")

if __name__ == "__main__":
    main()





