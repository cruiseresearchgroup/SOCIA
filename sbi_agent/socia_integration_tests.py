"""
SOCIA集成测试模块
测试SOCIA工作流集成功能、模块依赖关系处理、SOCIA格式结果生成、迭代式改进机制
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

# 导入SBI Agent
from simple_sbi_agent import SimpleSBIAgent

logger = logging.getLogger(__name__)

class SOCIAIntegrationTestSuite:
    """SOCIA集成测试套件"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.temp_dir = None
        self.test_results = {}
        
    def setup_test_environment(self) -> None:
        """设置测试环境"""
        # 创建临时测试目录
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sbi_test_"))
        
        # 复制测试数据到临时目录
        if self.test_data_dir.exists():
            shutil.copytree(self.test_data_dir, self.temp_dir / "test_data")
        
        logger.info(f"测试环境设置完成: {self.temp_dir}")
    
    def cleanup_test_environment(self) -> None:
        """清理测试环境"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("测试环境清理完成")
    
    def test_socia_workflow_integration(self) -> Dict[str, Any]:
        """测试SOCIA工作流集成"""
        test_name = "SOCIA工作流集成测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            
            # 测试配置加载
            agent.load_socia_configs()
            
            # 测试工作流约束获取
            calibration_order = agent.get_calibration_order()
            assert len(calibration_order) > 0, "校准顺序不能为空"
            
            # 测试模块约束获取
            for module_name in calibration_order:
                constraints = agent.get_workflow_constraints(module_name)
                assert isinstance(constraints, dict), f"模块 {module_name} 约束格式错误"
            
            # 测试迭代管理
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
            
            agent.update_iteration(mock_results)
            
            # 测试反馈生成
            feedback = agent.generate_workflow_feedback(mock_results)
            assert 'suggestions' in feedback, "反馈中缺少建议"
            assert 'next_steps' in feedback, "反馈中缺少下一步"
            
            # 测试改进建议
            suggestions = agent.get_improvement_suggestions()
            assert isinstance(suggestions, list), "改进建议格式错误"
            
            # 测试迭代继续判断
            should_continue = agent.should_continue_iteration()
            assert isinstance(should_continue, bool), "迭代继续判断格式错误"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': 'SOCIA工作流集成测试通过',
                'details': {
                    'calibration_order': calibration_order,
                    'feedback_generated': len(feedback.get('suggestions', [])),
                    'improvement_suggestions': len(suggestions)
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'SOCIA工作流集成测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_module_dependency_processing(self) -> Dict[str, Any]:
        """测试模块依赖关系处理"""
        test_name = "模块依赖关系处理测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            
            # 测试依赖关系分析
            agent.load_socia_configs()
            
            # 测试模块依赖获取
            calibration_order = agent.get_calibration_order()
            
            for module_name in calibration_order:
                dependencies = agent.get_module_dependencies(module_name)
                assert isinstance(dependencies, list), f"模块 {module_name} 依赖关系格式错误"
            
            # 测试校准顺序
            assert len(calibration_order) > 0, "校准顺序不能为空"
            
            # 测试参数空间构建
            for module_name in calibration_order:
                param_space = agent.get_parameter_space_for_module(module_name)
                assert isinstance(param_space, dict), f"模块 {module_name} 参数空间格式错误"
                
                param_bounds = agent.get_parameter_bounds_for_module(module_name)
                assert isinstance(param_bounds, dict), f"模块 {module_name} 参数边界格式错误"
            
            # 测试摘要统计设计
            for module_name in calibration_order:
                summary_design = agent.get_summary_statistics_design(module_name)
                assert isinstance(summary_design, dict), f"模块 {module_name} 摘要统计设计格式错误"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '模块依赖关系处理测试通过',
                'details': {
                    'calibration_order': calibration_order,
                    'total_modules': len(calibration_order),
                    'dependency_analysis_completed': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'模块依赖关系处理测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_socia_format_result_generation(self) -> Dict[str, Any]:
        """测试SOCIA格式结果生成"""
        test_name = "SOCIA格式结果生成测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            
            # 模拟校准结果
            mock_calibration_results = {
                'calibration_results': {
                    'InformationDiffusion': {
                        'convergence_achieved': True,
                        'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85},
                        'calibrated_parameters': {
                            'base_info_rate': 0.12,
                            'info_decay_rate': 0.05
                        }
                    },
                    'PolicyAndMessaging': {
                        'convergence_achieved': True,
                        'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88},
                        'calibrated_parameters': {
                            'policy_effect_size': 0.25,
                            'messaging_frequency': 0.4
                        }
                    }
                },
                'calibration_time': 1800
            }
            
            # 测试结果保存
            params_path = agent.save_calibrated_parameters(mock_calibration_results)
            assert params_path.exists(), "校准参数文件未生成"
            
            report_path = agent.save_calibration_report(mock_calibration_results)
            assert report_path.exists(), "校准报告文件未生成"
            
            feedback_path = agent.save_feedback(mock_calibration_results, iteration_id=1)
            assert feedback_path.exists(), "反馈文件未生成"
            
            verification_path = agent.save_verification_results(mock_calibration_results)
            assert verification_path.exists(), "验证结果文件未生成"
            
            # 验证文件内容
            with open(params_path, 'r', encoding='utf-8') as f:
                params_data = json.load(f)
                assert 'calibration_info' in params_data, "校准参数文件缺少校准信息"
                assert 'parameters' in params_data, "校准参数文件缺少参数"
                assert 'module_results' in params_data, "校准参数文件缺少模块结果"
            
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
                assert 'feedback_info' in feedback_data, "反馈文件缺少反馈信息"
                assert 'suggestions' in feedback_data, "反馈文件缺少建议"
                assert 'next_steps' in feedback_data, "反馈文件缺少下一步"
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': 'SOCIA格式结果生成测试通过',
                'details': {
                    'params_file': str(params_path),
                    'report_file': str(report_path),
                    'feedback_file': str(feedback_path),
                    'verification_file': str(verification_path)
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'SOCIA格式结果生成测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def test_iterative_improvement_mechanism(self) -> Dict[str, Any]:
        """测试迭代式改进机制"""
        test_name = "迭代式改进机制测试"
        logger.info(f"开始执行: {test_name}")
        
        try:
            # 创建SBI Agent
            agent = SimpleSBIAgent(self.temp_dir / "test_data")
            
            # 测试多轮迭代
            for iteration_id in range(1, 4):
                agent.start_iteration(iteration_id)
                
                # 模拟不同质量的校准结果
                if iteration_id == 1:
                    # 第一轮：部分收敛
                    mock_results = {
                        'calibration_results': {
                            'InformationDiffusion': {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.2, 'mae': 0.1, 'r2': 0.8}
                            },
                            'PolicyAndMessaging': {
                                'convergence_achieved': False,
                                'final_metrics': {'rmse': 0.4, 'mae': 0.2, 'r2': 0.6}
                            }
                        }
                    }
                elif iteration_id == 2:
                    # 第二轮：改进
                    mock_results = {
                        'calibration_results': {
                            'InformationDiffusion': {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.15, 'mae': 0.08, 'r2': 0.85}
                            },
                            'PolicyAndMessaging': {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.2, 'mae': 0.1, 'r2': 0.8}
                            }
                        }
                    }
                else:
                    # 第三轮：完全收敛
                    mock_results = {
                        'calibration_results': {
                            'InformationDiffusion': {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.1, 'mae': 0.05, 'r2': 0.9}
                            },
                            'PolicyAndMessaging': {
                                'convergence_achieved': True,
                                'final_metrics': {'rmse': 0.12, 'mae': 0.06, 'r2': 0.88}
                            }
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
            
            self.test_results[test_name] = {
                'status': 'passed',
                'message': '迭代式改进机制测试通过',
                'details': {
                    'iterations_completed': 3,
                    'improvement_mechanism_working': True
                }
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'failed',
                'message': f'迭代式改进机制测试失败: {e}',
                'error': str(e)
            }
            logger.error(f"{test_name} 失败: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始运行SOCIA集成测试套件")
        
        # 设置测试环境
        self.setup_test_environment()
        
        try:
            # 运行所有测试
            self.test_socia_workflow_integration()
            self.test_module_dependency_processing()
            self.test_socia_format_result_generation()
            self.test_iterative_improvement_mechanism()
            
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
            
            logger.info(f"SOCIA集成测试完成: {passed_tests}/{total_tests} 通过")
            return test_summary
            
        finally:
            # 清理测试环境
            self.cleanup_test_environment()

class SOCIAIntegrationTestRunner:
    """SOCIA集成测试运行器"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.test_suite = SOCIAIntegrationTestSuite(test_data_dir)
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        logger.info("开始运行SOCIA集成测试")
        
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
                'test_type': 'SOCIA集成测试',
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
            recommendations.append("存在失败的测试，建议检查相关功能")
        
        if test_results['success_rate'] < 0.8:
            recommendations.append("测试通过率较低，建议全面检查系统功能")
        
        if test_results['success_rate'] == 1.0:
            recommendations.append("所有测试通过，系统功能正常")
        
        return recommendations





