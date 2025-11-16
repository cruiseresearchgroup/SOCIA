"""
智能校准验证模块
基于SOCIA的verification_results进行验证，实现多维度校准质量评估
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class SOCIAVerificationValidator:
    """SOCIA验证验证器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.verification_results = {}
        self.validation_constraints = {}
        self.quality_thresholds = {}
    
    def load_verification_constraints(self, verification_path: Union[str, Path]) -> None:
        """加载验证约束"""
        try:
            with open(verification_path, 'r', encoding='utf-8') as f:
                self.verification_results = json.load(f)
            
            self._extract_validation_constraints()
            self._set_quality_thresholds()
            
            logger.info("验证约束加载完成")
        except Exception as e:
            logger.error(f"验证约束加载失败: {e}")
            raise
    
    def _extract_validation_constraints(self) -> None:
        """提取验证约束"""
        if 'issues' in self.verification_results:
            for issue in self.verification_results['issues']:
                if issue.get('type') == 'quality':
                    metric = issue.get('metric')
                    threshold = issue.get('threshold')
                    if metric and threshold:
                        self.quality_thresholds[metric] = threshold
                
                elif issue.get('type') == 'convergence':
                    module_name = issue.get('module')
                    requirements = issue.get('requirements')
                    if module_name and requirements:
                        self.validation_constraints[module_name] = requirements
    
    def _set_quality_thresholds(self) -> None:
        """设置质量阈值"""
        # 默认质量阈值
        default_thresholds = {
            'rmse': 0.3,
            'mae': 0.15,
            'r2': 0.7,
            'convergence_rate': 0.8,
            'calibration_time': 3600  # 1小时
        }
        
        # 合并用户定义的阈值
        for metric, threshold in default_thresholds.items():
            if metric not in self.quality_thresholds:
                self.quality_thresholds[metric] = threshold
    
    def validate_calibration_quality(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证校准质量"""
        validation_results = {
            'overall_status': 'passed',
            'module_validations': {},
            'quality_metrics': {},
            'constraint_violations': [],
            'recommendations': []
        }
        
        # 验证每个模块
        module_results = calibration_results.get('calibration_results', {})
        for module_name, module_results in module_results.items():
            module_validation = self._validate_module(module_name, module_results)
            validation_results['module_validations'][module_name] = module_validation
        
        # 计算整体质量指标
        validation_results['quality_metrics'] = self._calculate_quality_metrics(module_results)
        
        # 检查约束违反
        validation_results['constraint_violations'] = self._check_constraint_violations(module_results)
        
        # 生成建议
        validation_results['recommendations'] = self._generate_validation_recommendations(module_results)
        
        # 确定整体状态
        validation_results['overall_status'] = self._determine_overall_status(validation_results)
        
        return validation_results
    
    def _validate_module(self, module_name: str, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证单个模块"""
        validation = {
            'status': 'passed',
            'issues': [],
            'metrics': {},
            'constraint_checks': {}
        }
        
        # 检查收敛
        if not module_results.get('convergence_achieved', False):
            validation['status'] = 'failed'
            validation['issues'].append('convergence_failure')
        
        # 检查性能指标
        final_metrics = module_results.get('final_metrics', {})
        if final_metrics:
            validation['metrics'] = self._validate_metrics(final_metrics)
            
            # 检查阈值违反
            for metric, value in final_metrics.items():
                if metric in self.quality_thresholds:
                    threshold = self.quality_thresholds[metric]
                    if value > threshold:
                        validation['issues'].append(f'{metric}_threshold_violation')
                        validation['status'] = 'failed'
        
        # 检查模块特定约束
        if module_name in self.validation_constraints:
            constraint_checks = self._check_module_constraints(module_name, module_results)
            validation['constraint_checks'] = constraint_checks
            
            if any(not check['passed'] for check in constraint_checks.values()):
                validation['status'] = 'failed'
                validation['issues'].append('constraint_violation')
        
        return validation
    
    def _validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """验证指标"""
        validation = {}
        
        for metric, value in metrics.items():
            validation[metric] = {
                'value': value,
                'threshold': self.quality_thresholds.get(metric, float('inf')),
                'passed': value <= self.quality_thresholds.get(metric, float('inf')),
                'quality': self._assess_metric_quality(metric, value)
            }
        
        return validation
    
    def _assess_metric_quality(self, metric: str, value: float) -> str:
        """评估指标质量"""
        threshold = self.quality_thresholds.get(metric, float('inf'))
        
        if metric in ['rmse', 'mae']:
            if value <= threshold * 0.5:
                return 'excellent'
            elif value <= threshold:
                return 'good'
            elif value <= threshold * 1.5:
                return 'fair'
            else:
                return 'poor'
        elif metric in ['r2']:
            if value >= 0.9:
                return 'excellent'
            elif value >= 0.8:
                return 'good'
            elif value >= 0.7:
                return 'fair'
            else:
                return 'poor'
        else:
            return 'unknown'
    
    def _check_module_constraints(self, module_name: str, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """检查模块约束"""
        constraints = self.validation_constraints.get(module_name, {})
        constraint_checks = {}
        
        for constraint_name, constraint_value in constraints.items():
            constraint_checks[constraint_name] = {
                'constraint': constraint_value,
                'passed': True,  # 这里需要实现具体的约束检查逻辑
                'message': 'Constraint check not implemented'
            }
        
        return constraint_checks
    
    def _calculate_quality_metrics(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算质量指标"""
        if not module_results:
            return {}
        
        # 计算收敛率
        converged_count = sum(1 for r in module_results.values() if r.get('convergence_achieved', False))
        convergence_rate = converged_count / len(module_results)
        
        # 计算平均性能指标
        rmse_values = [r.get('final_metrics', {}).get('rmse', 0) for r in module_results.values()]
        mae_values = [r.get('final_metrics', {}).get('mae', 0) for r in module_results.values()]
        r2_values = [r.get('final_metrics', {}).get('r2', 0) for r in module_results.values()]
        
        return {
            'convergence_rate': convergence_rate,
            'average_rmse': np.mean(rmse_values) if rmse_values else 0,
            'average_mae': np.mean(mae_values) if mae_values else 0,
            'average_r2': np.mean(r2_values) if r2_values else 0,
            'rmse_std': np.std(rmse_values) if rmse_values else 0,
            'quality_score': self._calculate_overall_quality_score(module_results)
        }
    
    def _calculate_overall_quality_score(self, module_results: Dict[str, Any]) -> float:
        """计算整体质量分数"""
        if not module_results:
            return 0.0
        
        # 基于收敛率和性能指标计算质量分数
        converged_count = sum(1 for r in module_results.values() if r.get('convergence_achieved', False))
        convergence_rate = converged_count / len(module_results)
        
        # 计算平均RMSE
        rmse_values = [r.get('final_metrics', {}).get('rmse', 1.0) for r in module_results.values()]
        average_rmse = np.mean(rmse_values) if rmse_values else 1.0
        
        # 质量分数 = 收敛率 * (1 - 平均RMSE)
        quality_score = convergence_rate * (1 - min(average_rmse, 1.0))
        
        return quality_score
    
    def _check_constraint_violations(self, module_results: Dict[str, Any]) -> List[str]:
        """检查约束违反"""
        violations = []
        
        # 检查质量阈值违反
        for module_name, results in module_results.items():
            final_metrics = results.get('final_metrics', {})
            
            for metric, value in final_metrics.items():
                if metric in self.quality_thresholds:
                    threshold = self.quality_thresholds[metric]
                    if value > threshold:
                        violations.append(f"模块 {module_name} 的 {metric} ({value:.4f}) 超过阈值 ({threshold})")
        
        return violations
    
    def _generate_validation_recommendations(self, module_results: Dict[str, Any]) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        # 基于验证结果生成建议
        for module_name, results in module_results.items():
            if not results.get('convergence_achieved', False):
                recommendations.append(f"模块 {module_name} 需要重新校准以提高收敛性")
            
            final_metrics = results.get('final_metrics', {})
            if final_metrics:
                rmse = final_metrics.get('rmse', 1.0)
                if rmse > self.quality_thresholds.get('rmse', 0.3):
                    recommendations.append(f"模块 {module_name} 的RMSE ({rmse:.4f}) 需要改进")
                
                r2 = final_metrics.get('r2', 0.0)
                if r2 < self.quality_thresholds.get('r2', 0.7):
                    recommendations.append(f"模块 {module_name} 的R² ({r2:.4f}) 需要改进")
        
        return recommendations
    
    def _determine_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """确定整体状态"""
        # 检查是否有失败的模块
        failed_modules = [
            name for name, validation in validation_results['module_validations'].items()
            if validation['status'] == 'failed'
        ]
        
        if failed_modules:
            return 'failed'
        
        # 检查约束违反
        if validation_results['constraint_violations']:
            return 'warning'
        
        # 检查质量指标
        quality_metrics = validation_results['quality_metrics']
        if quality_metrics.get('quality_score', 0) < 0.5:
            return 'warning'
        
        return 'passed'

class MultiDimensionalQualityAssessor:
    """多维度质量评估器"""
    
    def __init__(self):
        self.assessment_dimensions = {
            'convergence': self._assess_convergence_quality,
            'performance': self._assess_performance_quality,
            'stability': self._assess_stability_quality,
            'efficiency': self._assess_efficiency_quality
        }
    
    def assess_calibration_quality(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估校准质量"""
        assessment = {
            'overall_score': 0.0,
            'dimension_scores': {},
            'dimension_details': {},
            'recommendations': []
        }
        
        # 评估各个维度
        for dimension, assessor in self.assessment_dimensions.items():
            dimension_result = assessor(calibration_results)
            assessment['dimension_scores'][dimension] = dimension_result['score']
            assessment['dimension_details'][dimension] = dimension_result
        
        # 计算整体分数
        assessment['overall_score'] = np.mean(list(assessment['dimension_scores'].values()))
        
        # 生成建议
        assessment['recommendations'] = self._generate_quality_recommendations(assessment)
        
        return assessment
    
    def _assess_convergence_quality(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估收敛质量"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return {'score': 0.0, 'details': 'No modules to assess'}
        
        # 计算收敛率
        converged_count = sum(1 for r in module_results.values() if r.get('convergence_achieved', False))
        convergence_rate = converged_count / len(module_results)
        
        # 计算收敛质量分数
        score = convergence_rate
        
        details = {
            'convergence_rate': convergence_rate,
            'converged_modules': converged_count,
            'total_modules': len(module_results),
            'non_converged_modules': [
                name for name, r in module_results.items()
                if not r.get('convergence_achieved', False)
            ]
        }
        
        return {'score': score, 'details': details}
    
    def _assess_performance_quality(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估性能质量"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return {'score': 0.0, 'details': 'No modules to assess'}
        
        # 计算性能指标
        rmse_values = [r.get('final_metrics', {}).get('rmse', 1.0) for r in module_results.values()]
        mae_values = [r.get('final_metrics', {}).get('mae', 1.0) for r in module_results.values()]
        r2_values = [r.get('final_metrics', {}).get('r2', 0.0) for r in module_results.values()]
        
        # 计算性能分数
        avg_rmse = np.mean(rmse_values) if rmse_values else 1.0
        avg_mae = np.mean(mae_values) if mae_values else 1.0
        avg_r2 = np.mean(r2_values) if r2_values else 0.0
        
        # 性能分数 = (1 - 平均RMSE) * 平均R²
        performance_score = (1 - min(avg_rmse, 1.0)) * max(avg_r2, 0.0)
        
        details = {
            'average_rmse': avg_rmse,
            'average_mae': avg_mae,
            'average_r2': avg_r2,
            'rmse_std': np.std(rmse_values) if rmse_values else 0,
            'performance_distribution': {
                'excellent': sum(1 for r in rmse_values if r < 0.1),
                'good': sum(1 for r in rmse_values if 0.1 <= r < 0.3),
                'fair': sum(1 for r in rmse_values if 0.3 <= r < 0.5),
                'poor': sum(1 for r in rmse_values if r >= 0.5)
            }
        }
        
        return {'score': performance_score, 'details': details}
    
    def _assess_stability_quality(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估稳定性质量"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return {'score': 0.0, 'details': 'No modules to assess'}
        
        # 计算性能指标的稳定性
        rmse_values = [r.get('final_metrics', {}).get('rmse', 1.0) for r in module_results.values()]
        
        if len(rmse_values) < 2:
            return {'score': 1.0, 'details': 'Insufficient data for stability assessment'}
        
        # 计算变异系数
        rmse_std = np.std(rmse_values)
        rmse_mean = np.mean(rmse_values)
        coefficient_of_variation = rmse_std / rmse_mean if rmse_mean > 0 else 0
        
        # 稳定性分数 = 1 - 变异系数
        stability_score = max(0, 1 - coefficient_of_variation)
        
        details = {
            'coefficient_of_variation': coefficient_of_variation,
            'rmse_std': rmse_std,
            'rmse_mean': rmse_mean,
            'stability_level': self._classify_stability(coefficient_of_variation)
        }
        
        return {'score': stability_score, 'details': details}
    
    def _assess_efficiency_quality(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估效率质量"""
        # 基于校准时间评估效率
        calibration_time = calibration_results.get('calibration_time', 0)
        
        # 效率分数 = 1 / (1 + 校准时间/3600)  # 以小时为单位
        efficiency_score = 1 / (1 + calibration_time / 3600)
        
        details = {
            'calibration_time': calibration_time,
            'efficiency_level': self._classify_efficiency(calibration_time)
        }
        
        return {'score': efficiency_score, 'details': details}
    
    def _classify_stability(self, coefficient_of_variation: float) -> str:
        """分类稳定性"""
        if coefficient_of_variation < 0.1:
            return 'excellent'
        elif coefficient_of_variation < 0.2:
            return 'good'
        elif coefficient_of_variation < 0.3:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_efficiency(self, calibration_time: float) -> str:
        """分类效率"""
        if calibration_time < 1800:  # 30分钟
            return 'excellent'
        elif calibration_time < 3600:  # 1小时
            return 'good'
        elif calibration_time < 7200:  # 2小时
            return 'fair'
        else:
            return 'poor'
    
    def _generate_quality_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """生成质量建议"""
        recommendations = []
        
        dimension_scores = assessment['dimension_scores']
        
        # 基于各维度分数生成建议
        for dimension, score in dimension_scores.items():
            if score < 0.5:
                if dimension == 'convergence':
                    recommendations.append("提高收敛性：增加迭代次数或调整参数空间")
                elif dimension == 'performance':
                    recommendations.append("提高性能：优化摘要统计设计或检查数据质量")
                elif dimension == 'stability':
                    recommendations.append("提高稳定性：减少参数空间变化或增加正则化")
                elif dimension == 'efficiency':
                    recommendations.append("提高效率：优化算法参数或使用并行计算")
        
        return recommendations

class CalibrationFailureHandler:
    """校准失败处理器"""
    
    def __init__(self):
        self.failure_patterns = {
            'convergence_failure': self._handle_convergence_failure,
            'parameter_cancellation': self._handle_parameter_cancellation,
            'simulation_timeout': self._handle_simulation_timeout,
            'memory_error': self._handle_memory_error,
            'data_quality_issue': self._handle_data_quality_issue
        }
    
    def handle_calibration_failure(self, failure_type: str, 
                                 calibration_results: Dict[str, Any],
                                 error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理校准失败"""
        if failure_type in self.failure_patterns:
            handler = self.failure_patterns[failure_type]
            return handler(calibration_results, error_details)
        else:
            return self._handle_unknown_failure(calibration_results, error_details)
    
    def _handle_convergence_failure(self, calibration_results: Dict[str, Any], 
                                  error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理收敛失败"""
        return {
            'recovery_strategy': 'parameter_adjustment',
            'adjustments': {
                'expand_parameter_bounds': True,
                'increase_iterations': True,
                'reduce_tolerance': True
            },
            'recommendations': [
                '扩大参数空间边界',
                '增加最大迭代次数',
                '降低收敛容差',
                '检查参数约束是否过于严格'
            ]
        }
    
    def _handle_parameter_cancellation(self, calibration_results: Dict[str, Any], 
                                     error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理参数对消"""
        return {
            'recovery_strategy': 'constraint_adjustment',
            'adjustments': {
                'add_parameter_constraints': True,
                'reduce_parameter_correlations': True,
                'use_regularization': True
            },
            'recommendations': [
                '添加参数独立性约束',
                '使用正则化技术',
                '分阶段校准参数',
                '检查参数间的相关性'
            ]
        }
    
    def _handle_simulation_timeout(self, calibration_results: Dict[str, Any], 
                                 error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理仿真超时"""
        return {
            'recovery_strategy': 'complexity_reduction',
            'adjustments': {
                'reduce_simulation_complexity': True,
                'use_simpler_sbi_method': True,
                'reduce_parameter_dimensions': True
            },
            'recommendations': [
                '简化仿真模型',
                '使用更简单的SBI方法',
                '减少参数维度',
                '增加超时时间'
            ]
        }
    
    def _handle_memory_error(self, calibration_results: Dict[str, Any], 
                           error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理内存错误"""
        return {
            'recovery_strategy': 'memory_optimization',
            'adjustments': {
                'reduce_batch_size': True,
                'use_gradient_checkpointing': True,
                'reduce_parameter_dimensions': True
            },
            'recommendations': [
                '减少批处理大小',
                '使用梯度检查点',
                '减少参数维度',
                '增加系统内存'
            ]
        }
    
    def _handle_data_quality_issue(self, calibration_results: Dict[str, Any], 
                                 error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据质量问题"""
        return {
            'recovery_strategy': 'data_improvement',
            'adjustments': {
                'improve_data_preprocessing': True,
                'add_data_validation': True,
                'use_robust_metrics': True
            },
            'recommendations': [
                '改进数据预处理',
                '添加数据验证步骤',
                '使用鲁棒性指标',
                '检查数据完整性'
            ]
        }
    
    def _handle_unknown_failure(self, calibration_results: Dict[str, Any], 
                              error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理未知失败"""
        return {
            'recovery_strategy': 'general_troubleshooting',
            'adjustments': {
                'reduce_complexity': True,
                'increase_robustness': True,
                'add_debugging': True
            },
            'recommendations': [
                '降低整体复杂度',
                '增加系统鲁棒性',
                '添加调试信息',
                '检查系统配置'
            ]
        }





