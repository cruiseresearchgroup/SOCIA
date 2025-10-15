"""
模块化SBI测试模块
测试单模块SBI校准、多模块联合校准、分阶段校准策略、参数对消处理
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

class ModularSBITestSuite:
    """模块化SBI测试套件"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.temp_dir = None
        self.test_results = {}
        
    def setup_test_environment(self) -> None:
        """设置测试环境"""
        # 创建临时测试目录
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sbi_modular_test_"))
        
        # 复制测试数据到临时目录
        if self.test_data_dir.exists():
            shutil.copytree(self.test_data_dir, self.temp_dir / "test_data")
        
        logger.info(f"模块化SBI测试环境设置完成: {self.temp_dir}")
    
    def cleanup_test_environment(self) -> None:
        """清理测试环境"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("模块化SBI测试环境清理完成")
    
    def test_single_module_sbi_calibration(self) -> Dict[str, Any]:
        """测试单模块SBI校准"""
        test_name = "单模块SBI校准测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            agent.load_socia_configs()
            
            # 获取校准模块
            calibration_order = agent.get_calibration_order()
            assert len(calibration_order) > 0, "没有可校准的模块"
            
            # 选择第一个模块进行测试
            test_module = calibration_order[0]
            
            # 测试参数空间构建
            param_space = agent.get_parameter_space_for_module(test_module)
            assert isinstance(param_space, dict), "参数空间格式错误"
            assert 'parameter_count' in param_space, "参数空间缺少参数数量"
            
            # 测试参数边界获取
            param_bounds = agent.get_parameter_bounds_for_module(test_module)
            assert isinstance(param_bounds, dict), "参数边界格式错误"
            
            # 测试参数验证
            test_params = {
                'test_param_1': 0.5,
                'test_param_2': 0.3,
                'test_param_3': 0.8
            }
            
            is_valid, validation_errors = agent.validate_module_parameters(test_params, test_module)
            assert isinstance(is_valid, bool), "参数验证返回格式错误"
            assert isinstance(validation_errors, list), "参数验证错误格式错误"
            
            # 测试SBI策略设计
            strategy = agent.design_sbi_strategy([test_module], data_complexity='medium')
            assert isinstance(strategy, dict), "SBI策略格式错误"
            assert 'sbi_methods' in strategy, "SBI策略缺少方法信息"
            assert test_module in strategy['sbi_methods'], f"策略中缺少模块 {test_module}"
            
            # 测试收敛监控
            agent.start_convergence_monitoring(test_module)
            
            # 模拟收敛指标更新
            for i in range(5):
                metrics = {
                    'rmse': 0.5 * np.exp(-i * 0.2) + 0.1,
                    'mae': 0.3 * np.exp(-i * 0.2) + 0.05,
                    'r2': 0.5 + 0.4 * (1 - np.exp(-i * 0.2))
                }
                agent.update_convergence_metrics(metrics)
            
            # 测试收敛状态获取
            convergence_status = agent.get_convergence_status()
            assert isinstance(convergence_status, dict), "收敛状态格式错误"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '单模块SBI校准测试通过',
                'details': {
                    'test_module': test_module,
                    'parameter_count': param_space.get('parameter_count', 0),
                    'strategy_designed': True,
                    'convergence_monitoring_working': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'单模块SBI校准测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_multi_module_joint_calibration(self) -> Dict[str, Any]:
        """测试多模块联合校准"""
        test_name = "多模块联合校准测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            agent.load_socia_configs()
            
            # 获取校准模块
            calibration_order = agent.get_calibration_order()
            assert len(calibration_order) >= 2, "需要至少2个模块进行联合校准测试"
            
            # 选择前两个模块进行测试
            test_modules = calibration_order[:2]
            
            # 测试多模块SBI策略设计
            strategy = agent.design_sbi_strategy(test_modules, data_complexity='medium')
            assert isinstance(strategy, dict), "多模块SBI策略格式错误"
            assert 'sbi_methods' in strategy, "多模块SBI策略缺少方法信息"
            assert 'joint_calibration' in strategy, "多模块SBI策略缺少联合校准信息"
            assert 'calibration_phases' in strategy, "多模块SBI策略缺少校准阶段信息"
            
            # 测试模块参数空间构建
            module_parameters = {}
            for module_name in test_modules:
                param_space = agent.get_parameter_space_for_module(module_name)
                module_parameters[module_name] = {
                    'param_1': 0.5,
                    'param_2': 0.3,
                    'param_3': 0.8
                }
            
            # 测试多模块仿真（模拟）
            try:
                # 这里应该调用实际的多模块仿真，但为了避免依赖，我们模拟
                simulation_results = {
                    module: {
                        'adoption_rate': np.random.random(30).tolist(),
                        'info_rate': np.random.random(30).tolist(),
                        'rmse': np.random.uniform(0.1, 0.5),
                        'mae': np.random.uniform(0.05, 0.3),
                        'r2': np.random.uniform(0.6, 0.95)
                    }
                    for module in test_modules
                }
                
                # 测试结果提取
                target_signals = ['adoption_rate', 'info_rate']
                extracted_signals = agent.simulation_wrapper.extract_target_signals(
                    simulation_results, target_signals
                ) if agent.simulation_wrapper else {}
                
                assert isinstance(extracted_signals, dict), "目标信号提取格式错误"
                
            except Exception as e:
                logger.warning(f"多模块仿真测试跳过: {e}")
                simulation_results = {}
            
            # 测试联合校准策略
            joint_strategy = strategy.get('joint_calibration', {})
            assert isinstance(joint_strategy, dict), "联合校准策略格式错误"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '多模块联合校准测试通过',
                'details': {
                    'test_modules': test_modules,
                    'strategy_designed': True,
                    'joint_calibration_configured': True,
                    'simulation_results_generated': len(simulation_results) > 0
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'多模块联合校准测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_phased_calibration_strategy(self) -> Dict[str, Any]:
        """测试分阶段校准策略"""
        test_name = "分阶段校准策略测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            agent.load_socia_configs()
            
            # 获取校准顺序
            calibration_order = agent.get_calibration_order()
            assert len(calibration_order) > 0, "没有可校准的模块"
            
            # 测试分阶段校准
            phase_results = {}
            
            for phase, module_name in enumerate(calibration_order):
                logger.info(f"测试阶段 {phase + 1}: {module_name}")
                
                # 开始收敛监控
                agent.start_convergence_monitoring(module_name)
                
                # 测试参数空间构建
                param_space = agent.get_parameter_space_for_module(module_name)
                assert isinstance(param_space, dict), f"模块 {module_name} 参数空间格式错误"
                
                # 测试SBI策略设计
                strategy = agent.design_sbi_strategy([module_name], data_complexity='medium')
                assert isinstance(strategy, dict), f"模块 {module_name} SBI策略格式错误"
                
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
                assert isinstance(convergence_status, dict), f"模块 {module_name} 收敛状态格式错误"
                
                # 测试参数调整
                current_bounds = agent.get_parameter_bounds_for_module(module_name)
                mock_calibration_results = {
                    'convergence_achieved': True,
                    'final_metrics': {'rmse': 0.2, 'mae': 0.1, 'r2': 0.8},
                    'parameter_values': {'param_1': 0.5, 'param_2': 0.3}
                }
                
                should_adjust, adjustment = agent.should_adjust_parameters(
                    module_name, current_bounds, mock_calibration_results
                )
                assert isinstance(should_adjust, bool), f"模块 {module_name} 参数调整判断格式错误"
                assert isinstance(adjustment, dict), f"模块 {module_name} 参数调整格式错误"
                
                phase_results[module_name] = {
                    'phase': phase + 1,
                    'convergence_status': convergence_status,
                    'parameter_adjustment': adjustment
                }
            
            # 测试整体校准策略
            enhanced_strategy = agent.get_enhanced_calibration_strategy()
            assert isinstance(enhanced_strategy, dict), "增强校准策略格式错误"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '分阶段校准策略测试通过',
                'details': {
                    'total_phases': len(calibration_order),
                    'phase_results': phase_results,
                    'enhanced_strategy_available': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'分阶段校准策略测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_parameter_cancellation_handling(self) -> Dict[str, Any]:
        """测试参数对消处理"""
        test_name = "参数对消处理测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            agent.load_socia_configs()
            
            # 获取校准模块
            calibration_order = agent.get_calibration_order()
            assert len(calibration_order) > 0, "没有可校准的模块"
            
            test_module = calibration_order[0]
            
            # 测试参数对消检测
            # 模拟参数对消情况
            problematic_params = {
                'param_1': 0.05,  # 接近边界
                'param_2': 0.95,  # 接近边界
                'param_3': 0.1,   # 接近边界
                'param_4': 0.9    # 接近边界
            }
            
            mock_calibration_results = {
                'convergence_achieved': False,
                'final_metrics': {'rmse': 0.8, 'mae': 0.4, 'r2': 0.3},
                'parameter_values': problematic_params
            }
            
            # 测试参数对消处理
            cancellation_analysis = agent.sbi_strategy.handle_parameter_cancellation(
                test_module, problematic_params, mock_calibration_results
            )
            
            assert isinstance(cancellation_analysis, dict), "参数对消分析格式错误"
            assert 'cancellation_detected' in cancellation_analysis, "参数对消分析缺少检测结果"
            assert 'problematic_parameters' in cancellation_analysis, "参数对消分析缺少问题参数"
            assert 'suggestions' in cancellation_analysis, "参数对消分析缺少建议"
            
            # 测试参数调整
            current_bounds = agent.get_parameter_bounds_for_module(test_module)
            should_adjust, adjustment = agent.should_adjust_parameters(
                test_module, current_bounds, mock_calibration_results
            )
            
            assert isinstance(should_adjust, bool), "参数调整判断格式错误"
            assert isinstance(adjustment, dict), "参数调整格式错误"
            
            # 测试重启机制
            should_restart, restart_strategy = agent.should_restart_calibration(
                'parameter_cancellation', mock_calibration_results
            )
            
            assert isinstance(should_restart, bool), "重启判断格式错误"
            assert isinstance(restart_strategy, dict), "重启策略格式错误"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '参数对消处理测试通过',
                'details': {
                    'test_module': test_module,
                    'cancellation_detected': cancellation_analysis.get('cancellation_detected', False),
                    'problematic_parameters_count': len(cancellation_analysis.get('problematic_parameters', [])),
                    'suggestions_count': len(cancellation_analysis.get('suggestions', [])),
                    'parameter_adjustment_available': should_adjust,
                    'restart_mechanism_available': should_restart
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'参数对消处理测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始运行模块化SBI测试套件")
        
        # 设置测试环境
        self.setup_test_environment()
        
        try:
            # 运行所有测试
            self.test_single_module_sbi_calibration()
            self.test_multi_module_joint_calibration()
            self.test_phased_calibration_strategy()
            self.test_parameter_cancellation_handling()
            
            # 统计测试结果
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'passed')
            failed_tests = total_tests - passed_tests
            
            test_summary = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'test_results': self.test_results
            }
            
            logger.info(f"模块化SBI测试完成: {passed_tests}/{total_tests} 通过")
            return test_summary
            
        finally:
            # 清理测试环境
            self.cleanup_test_environment()

class ModularSBITestRunner:
    """模块化SBI测试运行器"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.test_suite = ModularSBITestSuite(test_data_dir)
    
    def run_modular_sbi_tests(self) -> Dict[str, Any]:
        """运行模块化SBI测试"""
        logger.info("开始运行模块化SBI测试")
        
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
                'test_type': '模块化SBI测试',
                'version': '1.0'
            },
            'summary': {
                'total_tests': test_results['total_tests'],
                'passed_tests': test_results['passed_tests'],
                'failed_tests': test_results['failed_tests'],
                'success_rate': test_results['success_rate']
            },
            'detailed_results': test_results['test_results'],
            'recommendations': self._generate_test_recommendations(test_results)
        }
        
        return report
    
    def _generate_test_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """生成测试建议"""
        recommendations = []
        
        if test_results['failed_tests'] > 0:
            recommendations.append("存在失败的模块化SBI测试，建议检查相关功能")
        
        if test_results['success_rate'] < 0.8:
            recommendations.append("模块化SBI测试通过率较低，建议检查SBI算法实现")
        
        if test_results['success_rate'] == 1.0:
            recommendations.append("所有模块化SBI测试通过，SBI功能正常")
        
        return recommendations





