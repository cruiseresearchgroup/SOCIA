"""
综合测试运行器
整合所有测试模块，提供完整的测试和优化功能
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time

# 导入测试模块
from .socia_integration_tests import SOCIAIntegrationTestRunner
from .modular_sbi_tests import ModularSBITestRunner
from .end_to_end_tests import EndToEndTestRunner
from .performance_optimization import PerformanceOptimizer

logger = logging.getLogger(__name__)

class ComprehensiveTestRunner:
    """综合测试运行器"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.test_runners = {
            'socia_integration': SOCIAIntegrationTestRunner(test_data_dir),
            'modular_sbi': ModularSBITestRunner(test_data_dir),
            'end_to_end': EndToEndTestRunner(test_data_dir)
        }
        self.performance_optimizer = PerformanceOptimizer()
        self.test_results = {}
        self.optimization_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始运行综合测试套件")
        
        start_time = time.time()
        
        try:
            # 运行SOCIA集成测试
            logger.info("运行SOCIA集成测试...")
            socia_results = self.test_runners['socia_integration'].run_integration_tests()
            self.test_results['socia_integration'] = socia_results
            
            # 运行模块化SBI测试
            logger.info("运行模块化SBI测试...")
            modular_results = self.test_runners['modular_sbi'].run_modular_sbi_tests()
            self.test_results['modular_sbi'] = modular_results
            
            # 运行端到端测试
            logger.info("运行端到端测试...")
            e2e_results = self.test_runners['end_to_end'].run_end_to_end_tests()
            self.test_results['end_to_end'] = e2e_results
            
            # 运行性能优化
            logger.info("运行性能优化...")
            optimization_results = self._run_performance_optimization()
            self.optimization_results = optimization_results
            
            # 生成综合报告
            total_time = time.time() - start_time
            comprehensive_report = self._generate_comprehensive_report(total_time)
            
            logger.info(f"综合测试完成，总耗时: {total_time:.2f}秒")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"综合测试失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_results': self.test_results,
                'optimization_results': self.optimization_results
            }
    
    def _run_performance_optimization(self) -> Dict[str, Any]:
        """运行性能优化"""
        optimization_results = {
            'memory_optimization': {},
            'execution_optimization': {},
            'error_handling': {},
            'recommendations': []
        }
        
        try:
            # 内存优化
            if self.performance_optimizer.memory_optimizer.should_optimize_memory():
                memory_result = self.performance_optimizer.memory_optimizer.optimize_memory()
                optimization_results['memory_optimization'] = memory_result
            
            # 执行优化
            execution_result = self.performance_optimizer.execution_optimizer.optimize_execution(
                'comprehensive_testing', {'applicable_strategies': ['batch_processing', 'caching']}
            )
            optimization_results['execution_optimization'] = execution_result
            
            # 错误处理测试
            error_stats = self.performance_optimizer.error_handler.get_error_statistics()
            optimization_results['error_handling'] = error_stats
            
            # 生成优化建议
            optimization_results['recommendations'] = self._generate_optimization_recommendations()
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        for test_type, results in self.test_results.items():
            if results.get('summary', {}).get('success_rate', 0) < 0.8:
                recommendations.append(f"{test_type} 测试通过率较低，建议优化相关功能")
        
        # 基于性能指标生成建议
        performance_summary = self.performance_optimizer.get_performance_summary()
        
        memory_status = performance_summary.get('memory_status', {})
        if memory_status.get('memory_percent', 0) > 80:
            recommendations.append("内存使用率较高，建议启用内存优化")
        
        error_stats = performance_summary.get('error_stats', {})
        if error_stats.get('total_errors', 0) > 10:
            recommendations.append("错误数量较多，建议检查错误处理机制")
        
        return recommendations
    
    def _generate_comprehensive_report(self, total_time: float) -> Dict[str, Any]:
        """生成综合报告"""
        # 统计所有测试结果
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for test_type, results in self.test_results.items():
            summary = results.get('summary', {})
            total_tests += summary.get('total_tests', 0)
            total_passed += summary.get('passed_tests', 0)
            total_failed += summary.get('failed_tests', 0)
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # 生成综合报告
        comprehensive_report = {
            'report_info': {
                'timestamp': datetime.now().isoformat(),
                'test_type': '综合测试报告',
                'version': '1.0',
                'total_execution_time': total_time
            },
            'overall_summary': {
                'total_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'overall_success_rate': overall_success_rate
            },
            'test_results': self.test_results,
            'optimization_results': self.optimization_results,
            'performance_summary': self.performance_optimizer.get_performance_summary(),
            'recommendations': self._generate_comprehensive_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        
        return comprehensive_report
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 基于整体测试结果生成建议
        overall_success_rate = 0
        total_tests = 0
        
        for test_type, results in self.test_results.items():
            summary = results.get('summary', {})
            test_count = summary.get('total_tests', 0)
            success_rate = summary.get('success_rate', 0)
            
            total_tests += test_count
            overall_success_rate += success_rate * test_count
        
        if total_tests > 0:
            overall_success_rate /= total_tests
        
        if overall_success_rate < 0.8:
            recommendations.append("整体测试通过率较低，建议全面检查系统功能")
        elif overall_success_rate < 0.95:
            recommendations.append("测试通过率良好，建议优化失败的功能")
        else:
            recommendations.append("所有测试通过，系统功能正常")
        
        # 基于性能优化结果生成建议
        optimization_recommendations = self.optimization_results.get('recommendations', [])
        recommendations.extend(optimization_recommendations)
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """生成下一步建议"""
        next_steps = []
        
        # 基于测试结果生成下一步
        for test_type, results in self.test_results.items():
            summary = results.get('summary', {})
            if summary.get('failed_tests', 0) > 0:
                next_steps.append(f"修复 {test_type} 测试中的失败项")
        
        # 基于性能优化生成下一步
        if self.optimization_results.get('memory_optimization', {}).get('optimization_successful', True):
            next_steps.append("实施内存优化建议")
        
        if self.optimization_results.get('execution_optimization', {}).get('optimizations_applied'):
            next_steps.append("实施执行优化建议")
        
        # 通用下一步
        next_steps.extend([
            "监控系统性能",
            "定期运行测试套件",
            "根据测试结果持续改进"
        ])
        
        return next_steps
    
    def save_test_report(self, report: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """保存测试报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试报告保存完成: {output_path}")
    
    def run_regression_tests(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """运行回归测试"""
        logger.info("开始运行回归测试")
        
        # 运行当前测试
        current_results = self.run_all_tests()
        
        # 比较结果
        regression_analysis = self._analyze_regression(baseline_results, current_results)
        
        return {
            'regression_analysis': regression_analysis,
            'current_results': current_results,
            'baseline_results': baseline_results
        }
    
    def _analyze_regression(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """分析回归"""
        analysis = {
            'performance_regression': {},
            'functionality_regression': {},
            'overall_status': 'stable'
        }
        
        # 比较性能指标
        baseline_performance = baseline.get('performance_summary', {})
        current_performance = current.get('performance_summary', {})
        
        if baseline_performance and current_performance:
            # 比较内存使用
            baseline_memory = baseline_performance.get('memory_status', {}).get('memory_percent', 0)
            current_memory = current_performance.get('memory_status', {}).get('memory_percent', 0)
            
            if current_memory > baseline_memory * 1.2:  # 20%增长
                analysis['performance_regression']['memory'] = {
                    'baseline': baseline_memory,
                    'current': current_memory,
                    'regression': True
                }
                analysis['overall_status'] = 'regression_detected'
        
        # 比较功能测试结果
        baseline_tests = baseline.get('test_results', {})
        current_tests = current.get('test_results', {})
        
        for test_type in baseline_tests:
            if test_type in current_tests:
                baseline_success_rate = baseline_tests[test_type].get('summary', {}).get('success_rate', 0)
                current_success_rate = current_tests[test_type].get('summary', {}).get('success_rate', 0)
                
                if current_success_rate < baseline_success_rate:
                    analysis['functionality_regression'][test_type] = {
                        'baseline_success_rate': baseline_success_rate,
                        'current_success_rate': current_success_rate,
                        'regression': True
                    }
                    analysis['overall_status'] = 'regression_detected'
        
        return analysis

class TestAndOptimizationManager:
    """测试和优化管理器"""
    
    def __init__(self, test_data_dir: Union[str, Path]):
        self.test_data_dir = Path(test_data_dir)
        self.comprehensive_runner = ComprehensiveTestRunner(test_data_dir)
        self.performance_optimizer = PerformanceOptimizer()
        
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """运行完整测试套件"""
        logger.info("开始运行完整测试套件")
        
        # 运行综合测试
        test_results = self.comprehensive_runner.run_all_tests()
        
        # 运行性能优化
        optimization_results = self._run_performance_optimization()
        
        # 生成最终报告
        final_report = {
            'test_results': test_results,
            'optimization_results': optimization_results,
            'overall_status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        return final_report
    
    def _run_performance_optimization(self) -> Dict[str, Any]:
        """运行性能优化"""
        optimization_results = {
            'memory_optimization': {},
            'execution_optimization': {},
            'error_handling': {},
            'recommendations': []
        }
        
        try:
            # 内存优化
            if self.performance_optimizer.memory_optimizer.should_optimize_memory():
                memory_result = self.performance_optimizer.memory_optimizer.optimize_memory()
                optimization_results['memory_optimization'] = memory_result
            
            # 执行优化
            execution_result = self.performance_optimizer.execution_optimizer.optimize_execution(
                'comprehensive_testing', {'applicable_strategies': ['batch_processing', 'caching']}
            )
            optimization_results['execution_optimization'] = execution_result
            
            # 错误处理
            error_stats = self.performance_optimizer.error_handler.get_error_statistics()
            optimization_results['error_handling'] = error_stats
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def run_continuous_testing(self, interval: int = 3600) -> None:
        """运行持续测试"""
        logger.info(f"开始持续测试，间隔: {interval}秒")
        
        while True:
            try:
                # 运行测试
                test_results = self.comprehensive_runner.run_all_tests()
                
                # 保存结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.test_data_dir / f"continuous_test_results_{timestamp}.json"
                self.comprehensive_runner.save_test_report(test_results, output_path)
                
                logger.info(f"持续测试完成，结果保存到: {output_path}")
                
                # 等待下次测试
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("持续测试被用户中断")
                break
            except Exception as e:
                logger.error(f"持续测试失败: {e}")
                time.sleep(60)  # 错误后等待1分钟再重试
    
    def shutdown(self) -> None:
        """关闭测试和优化管理器"""
        self.performance_optimizer.shutdown()
        logger.info("测试和优化管理器已关闭")
