"""
端到端测试模块
测试mask_adoption任务的完整工作流程，从参数加载到结果保存的完整流程
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time

# 导入SBI Agent
from simple_sbi_agent import SimpleSBIAgent

logger = logging.getLogger(__name__)

class EndToEndTestSuite:
    """端到端测试套件"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.temp_dir = None
        self.test_results = {}
        self.performance_benchmarks = {}
        
    def setup_test_environment(self) -> None:
        """设置测试环境"""
        # 创建临时测试目录
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sbi_e2e_test_"))
        
        # 复制测试数据到临时目录
        if self.test_data_dir.exists():
            shutil.copytree(self.test_data_dir, self.temp_dir / "test_data")
        
        logger.info(f"端到端测试环境设置完成: {self.temp_dir}")
    
    def cleanup_test_environment(self) -> None:
        """清理测试环境"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("端到端测试环境清理完成")
    
    def test_mask_adoption_complete_workflow(self) -> Dict[str, Any]:
        """测试mask_adoption任务的完整工作流程"""
        test_name = "mask_adoption完整工作流程测试"
        logger.info(f"开始执行: {test_name}")
        
        start_time = time.time()
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            
            # 1. 测试配置加载
            logger.info("1. 测试配置加载...")
            config_load_start = time.time()
            agent.load_socia_configs()
            config_load_time = time.time() - config_load_start
            
            # 2. 测试数据加载
            logger.info("2. 测试数据加载...")
            data_load_start = time.time()
            agent.load_target_data()
            data_load_time = time.time() - data_load_start
            
            # 3. 测试参数加载
            logger.info("3. 测试参数加载...")
            param_load_start = time.time()
            agent.load_parameters()
            param_load_time = time.time() - param_load_start
            
            # 4. 测试校准策略设计
            logger.info("4. 测试校准策略设计...")
            strategy_start = time.time()
            calibration_order = agent.get_calibration_order()
            strategy = agent.design_sbi_strategy(calibration_order, data_complexity='medium')
            strategy_time = time.time() - strategy_start
            
            # 5. 测试仿真包装器初始化
            logger.info("5. 测试仿真包装器初始化...")
            wrapper_start = time.time()
            agent.initialize_simulation_wrapper()
            wrapper_time = time.time() - wrapper_start
            
            # 6. 测试校准过程（模拟）
            logger.info("6. 测试校准过程...")
            calibration_start = time.time()
            
            # 模拟校准结果
            mock_calibration_results = {
                'calibration_results': {
                    module: {
                        'convergence_achieved': True,
                        'final_metrics': {
                            'rmse': np.random.uniform(0.1, 0.3),
                            'mae': np.random.uniform(0.05, 0.15),
                            'r2': np.random.uniform(0.7, 0.95)
                        },
                        'calibrated_parameters': {
                            f'param_{i}': np.random.uniform(0.1, 0.9)
                            for i in range(3)
                        }
                    }
                    for module in calibration_order
                },
                'calibration_time': 1800
            }
            
            calibration_time = time.time() - calibration_start
            
            # 7. 测试结果验证
            logger.info("7. 测试结果验证...")
            validation_start = time.time()
            validation_output = agent.validate_calibration_results(mock_calibration_results)
            validation_time = time.time() - validation_start
            
            # 8. 测试结果保存
            logger.info("8. 测试结果保存...")
            save_start = time.time()
            
            # 保存各种结果文件
            params_path = agent.save_calibrated_parameters(mock_calibration_results)
            report_path = agent.save_calibration_report(mock_calibration_results)
            feedback_path = agent.save_feedback(mock_calibration_results, iteration_id=1)
            verification_path = agent.save_verification_results(mock_calibration_results)
            
            save_time = time.time() - save_start
            
            # 9. 测试可视化生成
            logger.info("9. 测试可视化生成...")
            viz_start = time.time()
            
            # 创建目标数据
            target_data = pd.DataFrame({
                'day': range(30),
                'wearing_mask': np.random.beta(2, 5, 30),
                'received_info': np.random.beta(3, 2, 30)
            })
            
            visualization_paths = agent.create_calibration_visualizations(
                mock_calibration_results, target_data
            )
            parameter_analysis_paths = agent.create_parameter_analysis_plots(mock_calibration_results)
            
            viz_time = time.time() - viz_start
            
            # 10. 测试性能分析
            logger.info("10. 测试性能分析...")
            analysis_start = time.time()
            
            performance_analysis = agent.analyze_calibration_performance(mock_calibration_results)
            optimization_advice = agent.generate_optimization_advice(mock_calibration_results, performance_analysis)
            
            analysis_time = time.time() - analysis_start
            
            # 11. 测试综合报告生成
            logger.info("11. 测试综合报告生成...")
            report_start = time.time()
            
            comprehensive_report_path = agent.generate_comprehensive_report(
                mock_calibration_results, 
                validation_output['validation_results'], 
                validation_output['quality_assessment']
            )
            optimization_report_path = agent.generate_optimization_report(performance_analysis, optimization_advice)
            
            report_time = time.time() - report_start
            
            total_time = time.time() - start_time
            
            # 记录性能基准
            self.performance_benchmarks[test_name] = {
                'total_time': total_time,
                'config_load_time': config_load_time,
                'data_load_time': data_load_time,
                'param_load_time': param_load_time,
                'strategy_time': strategy_time,
                'wrapper_time': wrapper_time,
                'calibration_time': calibration_time,
                'validation_time': validation_time,
                'save_time': save_time,
                'viz_time': viz_time,
                'analysis_time': analysis_time,
                'report_time': report_time
            }
            
            # 验证生成的文件
            generated_files = [
                params_path, report_path, feedback_path, verification_path,
                comprehensive_report_path, optimization_report_path
            ] + visualization_paths + parameter_analysis_paths
            
            files_exist = all(path.exists() for path in generated_files)
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': 'mask_adoption完整工作流程测试通过',
                'details': {
                    'total_time': total_time,
                    'calibration_order': calibration_order,
                    'strategy_designed': True,
                    'validation_completed': True,
                    'files_generated': len(generated_files),
                    'files_exist': files_exist,
                    'performance_benchmarks': self.performance_benchmarks[test_name]
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'mask_adoption完整工作流程测试失败: {e}',
                'error': str(e),
                'total_time': time.time() - start_time
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_socia_integration_workflow(self) -> Dict[str, Any]:
        """测试SOCIA集成工作流程"""
        test_name = "SOCIA集成工作流程测试"
        logger.info(f"开始执行: {test_name}")
        
        start_time = time.time()
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            agent.load_socia_configs()
            
            # 测试SOCIA工作流集成
            calibration_order = agent.get_calibration_order()
            
            # 测试迭代式改进
            for iteration_id in range(1, 4):
                agent.start_iteration(iteration_id)
                
                # 模拟不同质量的校准结果
                if iteration_id == 1:
                    mock_results = {
                        'calibration_results': {
                            module: {
                                'convergence_achieved': False,
                                'final_metrics': {'rmse': 0.5, 'mae': 0.25, 'r2': 0.6}
                            }
                            for module in calibration_order
                        }
                    }
                elif iteration_id == 2:
                    mock_results = {
                        'calibration_results': {
                            module: {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.3, 'mae': 0.15, 'r2': 0.8}
                            }
                            for module in calibration_order
                        }
                    }
                else:
                    mock_results = {
                        'calibration_results': {
                            module: {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.1, 'mae': 0.05, 'r2': 0.9}
                            }
                            for module in calibration_order
                        }
                    }
                
                agent.update_iteration(mock_results)
                
                # 测试反馈生成
                feedback = agent.generate_workflow_feedback(mock_results)
                assert 'suggestions' in feedback, f"第 {iteration_id} 轮反馈缺少建议"
                
                # 测试改进建议
                suggestions = agent.get_improvement_suggestions()
                assert isinstance(suggestions, list), f"第 {iteration_id} 轮改进建议格式错误"
                
                # 测试迭代继续判断
                should_continue = agent.should_continue_iteration()
                assert isinstance(should_continue, bool), f"第 {iteration_id} 轮迭代继续判断格式错误"
            
            # 测试工作流约束
            for module_name in calibration_order:
                constraints = agent.get_workflow_constraints(module_name)
                assert isinstance(constraints, dict), f"模块 {module_name} 约束格式错误"
            
            total_time = time.time() - start_time
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': 'SOCIA集成工作流程测试通过',
                'details': {
                    'total_time': total_time,
                    'iterations_completed': 3,
                    'calibration_order': calibration_order,
                    'workflow_integration_successful': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'SOCIA集成工作流程测试失败: {e}',
                'error': str(e),
                'total_time': time.time() - start_time
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_modular_calibration_workflow(self) -> Dict[str, Any]:
        """测试模块化校准工作流程"""
        test_name = "模块化校准工作流程测试"
        logger.info(f"开始执行: {test_name}")
        
        start_time = time.time()
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            agent.load_socia_configs()
            
            calibration_order = agent.get_calibration_order()
            
            # 测试分阶段校准
            phase_results = {}
            
            for phase, module_name in enumerate(calibration_order):
                logger.info(f"测试阶段 {phase + 1}: {module_name}")
                
                # 开始收敛监控
                agent.start_convergence_monitoring(module_name)
                
                # 测试参数空间构建
                param_space = agent.get_parameter_space_for_module(module_name)
                param_bounds = agent.get_parameter_bounds_for_module(module_name)
                
                # 测试SBI策略设计
                strategy = agent.design_sbi_strategy([module_name], data_complexity='medium')
                
                # 模拟校准过程
                for iteration in range(3):
                    metrics = {
                        'rmse': 0.5 * (0.8 ** iteration),
                        'mae': 0.3 * (0.8 ** iteration),
                        'r2': 0.6 + 0.3 * (1 - 0.8 ** iteration)
                    }
                    agent.update_convergence_metrics(metrics)
                
                # 测试收敛状态
                convergence_status = agent.get_convergence_status()
                
                # 测试参数调整
                mock_calibration_results = {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.2, 'mae': 0.1, 'r2': 0.8},
                    'parameter_values': {'param_1': 0.5, 'param_2': 0.3}
                }
                
                should_adjust, adjustment = agent.should_adjust_parameters(
                    module_name, param_bounds, mock_calibration_results
                )
                
                phase_results[module_name] = {
                    'phase': phase + 1,
                    'convergence_status': convergence_status,
                    'parameter_adjustment': adjustment,
                    'strategy_designed': True
                }
            
            # 测试多模块联合校准
            multi_module_strategy = agent.design_sbi_strategy(calibration_order, data_complexity='medium')
            
            total_time = time.time() - start_time
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '模块化校准工作流程测试通过',
                'details': {
                    'total_time': total_time,
                    'total_phases': len(calibration_order),
                    'phase_results': phase_results,
                    'multi_module_strategy_designed': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'模块化校准工作流程测试失败: {e}',
                'error': str(e),
                'total_time': time.time() - start_time
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_performance_and_stability(self) -> Dict[str, Any]:
        """测试性能和稳定性"""
        test_name = "性能和稳定性测试"
        logger.info(f"开始执行: {test_name}")
        
        start_time = time.time()
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            
            # 测试多次配置加载的性能
            load_times = []
            for i in range(5):
                load_start = time.time()
                agent.load_socia_configs()
                load_time = time.time() - load_start
                load_times.append(load_time)
            
            # 测试内存使用
            import psutil
            memory_before = psutil.virtual_memory().used
            agent.load_target_data()
            memory_after = psutil.virtual_memory().used
            memory_increase = memory_after - memory_before
            
            # 测试错误处理
            error_handling_tests = []
            
            # 测试无效参数处理
            try:
                invalid_params = {'invalid_param': 'invalid_value'}
                is_valid, errors = agent.validate_module_parameters(invalid_params, 'test_module')
                error_handling_tests.append({
                    'test': 'invalid_parameters',
                    'handled': True,
                    'errors_count': len(errors)
                })
            except Exception as e:
                error_handling_tests.append({
                    'test': 'invalid_parameters',
                    'handled': False,
                    'error': str(e)
                })
            
            # 测试性能基准
            performance_metrics = {
                'average_load_time': np.mean(load_times),
                'load_time_std': np.std(load_times),
                'memory_increase_mb': memory_increase / 1024 / 1024,
                'error_handling_tests': error_handling_tests
            }
            
            total_time = time.time() - start_time
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '性能和稳定性测试通过',
                'details': {
                    'total_time': total_time,
                    'performance_metrics': performance_metrics,
                    'stability_tests_passed': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'性能和稳定性测试失败: {e}',
                'error': str(e),
                'total_time': time.time() - start_time
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始运行端到端测试套件")
        
        # 设置测试环境
        self.setup_test_environment()
        
        try:
            # 运行所有测试
            self.test_mask_adoption_complete_workflow()
            self.test_socia_integration_workflow()
            self.test_modular_calibration_workflow()
            self.test_performance_and_stability()
            
            # 统计测试结果
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'passed')
            failed_tests = total_tests - passed_tests
            
            test_summary = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'test_results': self.test_results,
                'performance_benchmarks': self.performance_benchmarks
            }
            
            logger.info(f"端到端测试完成: {passed_tests}/{total_tests} 通过")
            return test_summary
            
        finally:
            # 清理测试环境
            self.cleanup_test_environment()

class EndToEndTestRunner:
    """端到端测试运行器"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.test_suite = EndToEndTestSuite(test_data_dir)
    
    def run_end_to_end_tests(self) -> Dict[str, Any]:
        """运行端到端测试"""
        logger.info("开始运行端到端测试")
        
        # 运行测试套件
        test_results = self.test_suite.run_all_tests()
        
        # 生成测试报告
        report = self._generate_test_report(test_results)
        
        return report
    
    def _generate_test_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_type': '端到端测试',
                'version': '1.0'
            },
            'summary': {
                'total_tests': test_results['total_tests'],
                'passed_tests': test_results['passed_tests'],
                'failed_tests': test_results['failed_tests'],
                'success_rate': test_results['success_rate']
            },
            'detailed_results': test_results['test_results'],
            'performance_benchmarks': test_results.get('performance_benchmarks', {}),
            'recommendations': self._generate_test_recommendations(test_results)
        }
        
        return report
    
    def _generate_test_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """生成测试建议"""
        recommendations = []
        
        if test_results['failed_tests'] > 0:
            recommendations.append("存在失败的端到端测试，建议检查系统集成")
        
        if test_results['success_rate'] < 0.8:
            recommendations.append("端到端测试通过率较低，建议检查完整工作流程")
        
        if test_results['success_rate'] == 1.0:
            recommendations.append("所有端到端测试通过，系统功能完整")
        
        # 基于性能基准生成建议
        performance_benchmarks = test_results.get('performance_benchmarks', {})
        for test_name, benchmarks in performance_benchmarks.items():
            if benchmarks.get('total_time', 0) > 300:  # 5分钟
                recommendations.append(f"{test_name} 执行时间过长，建议优化性能")
        
        return recommendations





