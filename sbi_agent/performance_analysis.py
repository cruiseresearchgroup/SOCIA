"""
性能分析和优化建议模块
分析校准性能和收敛性，提供参数优化建议，支持模块化性能分析
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

class CalibrationPerformanceAnalyzer:
    """校准性能分析器"""
    
    def __init__(self):
        self.performance_history = []
        self.benchmark_metrics = {
            'excellent_rmse': 0.1,
            'good_rmse': 0.3,
            'fair_rmse': 0.5,
            'excellent_r2': 0.9,
            'good_r2': 0.7,
            'fair_r2': 0.5
        }
    
    def analyze_calibration_performance(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析校准性能"""
        analysis = {
            'overall_performance': {},
            'module_performance': {},
            'performance_trends': {},
            'optimization_opportunities': [],
            'recommendations': []
        }
        
        # 分析整体性能
        analysis['overall_performance'] = self._analyze_overall_performance(calibration_results)
        
        # 分析模块性能
        analysis['module_performance'] = self._analyze_module_performance(calibration_results)
        
        # 分析性能趋势
        analysis['performance_trends'] = self._analyze_performance_trends(calibration_results)
        
        # 识别优化机会
        analysis['optimization_opportunities'] = self._identify_optimization_opportunities(calibration_results)
        
        # 生成建议
        analysis['recommendations'] = self._generate_performance_recommendations(calibration_results)
        
        return analysis
    
    def _analyze_overall_performance(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析整体性能"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return {'status': 'no_data', 'score': 0.0}
        
        # 计算整体指标
        rmse_values = [r.get('final_metrics', {}).get('rmse', 1.0) for r in module_results.values()]
        mae_values = [r.get('final_metrics', {}).get('mae', 1.0) for r in module_results.values()]
        r2_values = [r.get('final_metrics', {}).get('r2', 0.0) for r in module_results.values()]
        
        avg_rmse = np.mean(rmse_values)
        avg_mae = np.mean(mae_values)
        avg_r2 = np.mean(r2_values)
        
        # 计算性能分数
        performance_score = self._calculate_performance_score(avg_rmse, avg_mae, avg_r2)
        
        # 确定性能等级
        performance_level = self._classify_performance_level(avg_rmse, avg_r2)
        
        return {
            'status': 'completed',
            'score': performance_score,
            'level': performance_level,
            'metrics': {
                'average_rmse': avg_rmse,
                'average_mae': avg_mae,
                'average_r2': avg_r2,
                'rmse_std': np.std(rmse_values),
                'r2_std': np.std(r2_values)
            },
            'convergence_rate': sum(1 for r in module_results.values() if r.get('convergence_achieved', False)) / len(module_results)
        }
    
    def _analyze_module_performance(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析模块性能"""
        module_results = calibration_results.get('calibration_results', {})
        module_analysis = {}
        
        for module_name, results in module_results.items():
            final_metrics = results.get('final_metrics', {})
            
            if not final_metrics:
                module_analysis[module_name] = {
                    'status': 'no_metrics',
                    'score': 0.0,
                    'issues': ['no_metrics_available']
                }
                continue
            
            rmse = final_metrics.get('rmse', 1.0)
            mae = final_metrics.get('mae', 1.0)
            r2 = final_metrics.get('r2', 0.0)
            
            # 计算模块性能分数
            module_score = self._calculate_performance_score(rmse, mae, r2)
            
            # 识别模块问题
            issues = self._identify_module_issues(rmse, mae, r2, results.get('convergence_achieved', False))
            
            # 评估模块质量
            quality_level = self._classify_performance_level(rmse, r2)
            
            module_analysis[module_name] = {
                'status': 'completed',
                'score': module_score,
                'level': quality_level,
                'metrics': final_metrics,
                'issues': issues,
                'convergence_achieved': results.get('convergence_achieved', False)
            }
        
        return module_analysis
    
    def _analyze_performance_trends(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能趋势"""
        # 这里需要历史数据，暂时返回模拟分析
        trends = {
            'rmse_trend': 'improving',
            'convergence_trend': 'stable',
            'efficiency_trend': 'stable',
            'stability_trend': 'improving'
        }
        
        return trends
    
    def _identify_optimization_opportunities(self, calibration_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别优化机会"""
        opportunities = []
        
        module_results = calibration_results.get('calibration_results', {})
        
        for module_name, results in module_results.items():
            final_metrics = results.get('final_metrics', {})
            
            if not final_metrics:
                continue
            
            rmse = final_metrics.get('rmse', 1.0)
            r2 = final_metrics.get('r2', 0.0)
            
            # 识别优化机会
            if rmse > self.benchmark_metrics['good_rmse']:
                opportunities.append({
                    'module': module_name,
                    'type': 'performance_improvement',
                    'priority': 'high' if rmse > self.benchmark_metrics['fair_rmse'] else 'medium',
                    'description': f'模块 {module_name} 的RMSE ({rmse:.4f}) 需要改进',
                    'suggestions': [
                        '调整参数空间',
                        '优化摘要统计设计',
                        '检查数据质量',
                        '增加迭代次数'
                    ]
                })
            
            if r2 < self.benchmark_metrics['good_r2']:
                opportunities.append({
                    'module': module_name,
                    'type': 'model_fit_improvement',
                    'priority': 'high' if r2 < self.benchmark_metrics['fair_r2'] else 'medium',
                    'description': f'模块 {module_name} 的R² ({r2:.4f}) 需要改进',
                    'suggestions': [
                        '优化模型结构',
                        '改进参数约束',
                        '使用更复杂的摘要统计',
                        '检查参数相关性'
                    ]
                })
            
            if not results.get('convergence_achieved', False):
                opportunities.append({
                    'module': module_name,
                    'type': 'convergence_improvement',
                    'priority': 'high',
                    'description': f'模块 {module_name} 未收敛',
                    'suggestions': [
                        '增加最大迭代次数',
                        '调整收敛容差',
                        '扩大参数空间',
                        '使用不同的SBI方法'
                    ]
                })
        
        return opportunities
    
    def _generate_performance_recommendations(self, calibration_results: Dict[str, Any]) -> List[str]:
        """生成性能建议"""
        recommendations = []
        
        # 基于整体性能生成建议
        overall_perf = self._analyze_overall_performance(calibration_results)
        
        if overall_perf['score'] < 0.5:
            recommendations.append("整体性能较低，建议检查数据质量和模型结构")
        
        if overall_perf['metrics']['average_rmse'] > 0.5:
            recommendations.append("平均RMSE较高，建议优化摘要统计设计")
        
        if overall_perf['convergence_rate'] < 0.8:
            recommendations.append("收敛率较低，建议调整SBI参数或方法")
        
        # 基于模块性能生成建议
        module_perf = self._analyze_module_performance(calibration_results)
        
        for module_name, module_analysis in module_perf.items():
            if module_analysis['score'] < 0.3:
                recommendations.append(f"模块 {module_name} 性能较差，需要重点优化")
            
            if 'high_rmse' in module_analysis['issues']:
                recommendations.append(f"模块 {module_name} RMSE过高，建议调整参数约束")
            
            if 'low_r2' in module_analysis['issues']:
                recommendations.append(f"模块 {module_name} R²过低，建议优化模型结构")
        
        return recommendations
    
    def _calculate_performance_score(self, rmse: float, mae: float, r2: float) -> float:
        """计算性能分数"""
        # 性能分数 = (1 - RMSE) * R²
        # 限制RMSE在[0, 1]范围内
        normalized_rmse = min(rmse, 1.0)
        performance_score = (1 - normalized_rmse) * max(r2, 0.0)
        
        return performance_score
    
    def _classify_performance_level(self, rmse: float, r2: float) -> str:
        """分类性能等级"""
        if rmse <= self.benchmark_metrics['excellent_rmse'] and r2 >= self.benchmark_metrics['excellent_r2']:
            return 'excellent'
        elif rmse <= self.benchmark_metrics['good_rmse'] and r2 >= self.benchmark_metrics['good_r2']:
            return 'good'
        elif rmse <= self.benchmark_metrics['fair_rmse'] and r2 >= self.benchmark_metrics['fair_r2']:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_module_issues(self, rmse: float, mae: float, r2: float, converged: bool) -> List[str]:
        """识别模块问题"""
        issues = []
        
        if not converged:
            issues.append('convergence_failure')
        
        if rmse > self.benchmark_metrics['good_rmse']:
            issues.append('high_rmse')
        
        if mae > 0.2:  # MAE阈值
            issues.append('high_mae')
        
        if r2 < self.benchmark_metrics['good_r2']:
            issues.append('low_r2')
        
        return issues

class ParameterOptimizationAdvisor:
    """参数优化建议器"""
    
    def __init__(self):
        self.optimization_strategies = {
            'parameter_space': self._optimize_parameter_space,
            'sbi_method': self._optimize_sbi_method,
            'summary_statistics': self._optimize_summary_statistics,
            'convergence_criteria': self._optimize_convergence_criteria
        }
    
    def generate_optimization_advice(self, calibration_results: Dict[str, Any],
                                   performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化建议"""
        advice = {
            'overall_optimization': {},
            'module_optimizations': {},
            'parameter_adjustments': {},
            'method_changes': {},
            'priority_actions': []
        }
        
        # 生成整体优化建议
        advice['overall_optimization'] = self._generate_overall_optimization(calibration_results, performance_analysis)
        
        # 生成模块优化建议
        advice['module_optimizations'] = self._generate_module_optimizations(calibration_results, performance_analysis)
        
        # 生成参数调整建议
        advice['parameter_adjustments'] = self._generate_parameter_adjustments(calibration_results, performance_analysis)
        
        # 生成方法变更建议
        advice['method_changes'] = self._generate_method_changes(calibration_results, performance_analysis)
        
        # 生成优先级行动
        advice['priority_actions'] = self._generate_priority_actions(calibration_results, performance_analysis)
        
        return advice
    
    def _generate_overall_optimization(self, calibration_results: Dict[str, Any],
                                     performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成整体优化建议"""
        overall_perf = performance_analysis.get('overall_performance', {})
        
        optimization = {
            'strategy': 'comprehensive_optimization',
            'focus_areas': [],
            'expected_improvements': {},
            'implementation_plan': []
        }
        
        # 基于性能分析确定重点领域
        if overall_perf.get('score', 0) < 0.5:
            optimization['focus_areas'].append('overall_performance')
            optimization['expected_improvements']['performance_score'] = '0.3-0.5 improvement'
        
        if overall_perf.get('convergence_rate', 0) < 0.8:
            optimization['focus_areas'].append('convergence_improvement')
            optimization['expected_improvements']['convergence_rate'] = '0.2-0.3 improvement'
        
        if overall_perf.get('metrics', {}).get('average_rmse', 1.0) > 0.3:
            optimization['focus_areas'].append('accuracy_improvement')
            optimization['expected_improvements']['rmse'] = '0.1-0.2 reduction'
        
        # 生成实施计划
        optimization['implementation_plan'] = [
            '1. 分析性能瓶颈',
            '2. 调整参数空间',
            '3. 优化SBI方法',
            '4. 改进摘要统计',
            '5. 验证优化效果'
        ]
        
        return optimization
    
    def _generate_module_optimizations(self, calibration_results: Dict[str, Any],
                                     performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成模块优化建议"""
        module_perf = performance_analysis.get('module_performance', {})
        module_optimizations = {}
        
        for module_name, module_analysis in module_perf.items():
            optimization = {
                'strategy': 'module_specific_optimization',
                'focus_areas': [],
                'parameter_adjustments': {},
                'method_changes': {},
                'expected_improvements': {}
            }
            
            # 基于模块问题确定优化重点
            issues = module_analysis.get('issues', [])
            
            if 'convergence_failure' in issues:
                optimization['focus_areas'].append('convergence')
                optimization['parameter_adjustments']['max_iterations'] = 'increase_by_50%'
                optimization['parameter_adjustments']['tolerance'] = 'reduce_by_20%'
            
            if 'high_rmse' in issues:
                optimization['focus_areas'].append('accuracy')
                optimization['method_changes']['summary_statistics'] = 'enhance_complexity'
                optimization['method_changes']['parameter_bounds'] = 'refine_based_on_data'
            
            if 'low_r2' in issues:
                optimization['focus_areas'].append('model_fit')
                optimization['method_changes']['sbi_method'] = 'consider_alternative'
                optimization['method_changes']['regularization'] = 'add_regularization'
            
            # 生成预期改进
            current_score = module_analysis.get('score', 0)
            if current_score < 0.5:
                optimization['expected_improvements']['performance_score'] = f'{current_score + 0.2:.2f}'
            
            module_optimizations[module_name] = optimization
        
        return module_optimizations
    
    def _generate_parameter_adjustments(self, calibration_results: Dict[str, Any],
                                      performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成参数调整建议"""
        adjustments = {
            'global_adjustments': {},
            'module_specific_adjustments': {},
            'constraint_adjustments': {}
        }
        
        # 全局参数调整
        overall_perf = performance_analysis.get('overall_performance', {})
        
        if overall_perf.get('convergence_rate', 0) < 0.8:
            adjustments['global_adjustments']['max_iterations'] = {
                'current': 'default',
                'recommended': 'increase_by_50%',
                'reason': 'low_convergence_rate'
            }
        
        if overall_perf.get('metrics', {}).get('average_rmse', 1.0) > 0.3:
            adjustments['global_adjustments']['tolerance'] = {
                'current': 'default',
                'recommended': 'reduce_by_30%',
                'reason': 'high_average_rmse'
            }
        
        # 模块特定调整
        module_perf = performance_analysis.get('module_performance', {})
        
        for module_name, module_analysis in module_perf.items():
            if module_analysis.get('score', 0) < 0.3:
                adjustments['module_specific_adjustments'][module_name] = {
                    'parameter_bounds': 'expand_by_20%',
                    'learning_rate': 'reduce_by_25%',
                    'batch_size': 'increase_by_50%'
                }
        
        return adjustments
    
    def _generate_method_changes(self, calibration_results: Dict[str, Any],
                               performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成方法变更建议"""
        changes = {
            'sbi_method_changes': {},
            'summary_statistics_changes': {},
            'convergence_criteria_changes': {}
        }
        
        # SBI方法变更
        overall_perf = performance_analysis.get('overall_performance', {})
        
        if overall_perf.get('convergence_rate', 0) < 0.7:
            changes['sbi_method_changes']['recommendation'] = 'consider_snle_or_abc'
            changes['sbi_method_changes']['reason'] = 'low_convergence_rate'
        
        if overall_perf.get('metrics', {}).get('average_rmse', 1.0) > 0.4:
            changes['sbi_method_changes']['recommendation'] = 'consider_snre'
            changes['sbi_method_changes']['reason'] = 'high_average_rmse'
        
        # 摘要统计变更
        changes['summary_statistics_changes']['recommendation'] = 'add_morphological_features'
        changes['summary_statistics_changes']['reason'] = 'improve_discriminative_power'
        
        # 收敛标准变更
        changes['convergence_criteria_changes']['recommendation'] = 'implement_early_stopping'
        changes['convergence_criteria_changes']['reason'] = 'improve_efficiency'
        
        return changes
    
    def _generate_priority_actions(self, calibration_results: Dict[str, Any],
                                 performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优先级行动"""
        actions = []
        
        # 基于性能分析确定优先级行动
        overall_perf = performance_analysis.get('overall_performance', {})
        module_perf = performance_analysis.get('module_performance', {})
        
        # 高优先级行动
        if overall_perf.get('convergence_rate', 0) < 0.5:
            actions.append({
                'priority': 'high',
                'action': 'improve_convergence',
                'description': '提高整体收敛率',
                'expected_impact': 'high',
                'implementation_effort': 'medium'
            })
        
        # 中优先级行动
        if overall_perf.get('score', 0) < 0.5:
            actions.append({
                'priority': 'medium',
                'action': 'improve_accuracy',
                'description': '提高整体准确性',
                'expected_impact': 'medium',
                'implementation_effort': 'high'
            })
        
        # 低优先级行动
        actions.append({
            'priority': 'low',
            'action': 'optimize_efficiency',
            'description': '优化计算效率',
            'expected_impact': 'low',
            'implementation_effort': 'low'
        })
        
        return actions
    
    def _optimize_parameter_space(self, module_name: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化参数空间"""
        # 实现参数空间优化逻辑
        return {'strategy': 'parameter_space_optimization'}
    
    def _optimize_sbi_method(self, module_name: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化SBI方法"""
        # 实现SBI方法优化逻辑
        return {'strategy': 'sbi_method_optimization'}
    
    def _optimize_summary_statistics(self, module_name: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化摘要统计"""
        # 实现摘要统计优化逻辑
        return {'strategy': 'summary_statistics_optimization'}
    
    def _optimize_convergence_criteria(self, module_name: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化收敛标准"""
        # 实现收敛标准优化逻辑
        return {'strategy': 'convergence_criteria_optimization'}

class SOCIAOptimizationReporter:
    """SOCIA优化报告生成器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.optimization_reports_dir = self.output_dir / "sbi_results" / "optimization_reports"
        self.optimization_reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_optimization_report(self, performance_analysis: Dict[str, Any],
                                   optimization_advice: Dict[str, Any]) -> Path:
        """生成优化报告"""
        report_data = {
            'report_info': {
                'title': 'SOCIA SBI优化建议报告',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            },
            'performance_summary': performance_analysis.get('overall_performance', {}),
            'module_analysis': performance_analysis.get('module_performance', {}),
            'optimization_opportunities': performance_analysis.get('optimization_opportunities', []),
            'optimization_advice': optimization_advice,
            'implementation_roadmap': self._generate_implementation_roadmap(optimization_advice)
        }
        
        # 生成HTML报告
        html_path = self.optimization_reports_dir / "optimization_report.html"
        self._generate_html_optimization_report(report_data, html_path)
        
        # 生成JSON报告
        json_path = self.optimization_reports_dir / "optimization_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化报告生成完成: {html_path}, {json_path}")
        return html_path
    
    def _generate_implementation_roadmap(self, optimization_advice: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成实施路线图"""
        roadmap = []
        
        # 基于优化建议生成路线图
        priority_actions = optimization_advice.get('priority_actions', [])
        
        for i, action in enumerate(priority_actions):
            roadmap.append({
                'phase': i + 1,
                'action': action['action'],
                'description': action['description'],
                'priority': action['priority'],
                'expected_impact': action['expected_impact'],
                'implementation_effort': action['implementation_effort'],
                'estimated_duration': self._estimate_duration(action['implementation_effort']),
                'dependencies': self._identify_dependencies(action['action'])
            })
        
        return roadmap
    
    def _estimate_duration(self, effort: str) -> str:
        """估计实施时间"""
        duration_map = {
            'low': '1-2 days',
            'medium': '3-5 days',
            'high': '1-2 weeks'
        }
        return duration_map.get(effort, 'unknown')
    
    def _identify_dependencies(self, action: str) -> List[str]:
        """识别依赖关系"""
        dependencies = {
            'improve_convergence': ['parameter_space_analysis', 'sbi_method_selection'],
            'improve_accuracy': ['data_quality_check', 'summary_statistics_design'],
            'optimize_efficiency': ['performance_profiling', 'resource_allocation']
        }
        return dependencies.get(action, [])
    
    def _generate_html_optimization_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """生成HTML优化报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report_info.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .summary { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #bdc3c7; border-radius: 5px; }
                .metric { display: inline-block; margin: 5px 10px; padding: 8px 12px; background-color: #3498db; color: white; border-radius: 3px; }
                .recommendation { background-color: #f39c12; color: white; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .high-priority { background-color: #e74c3c; }
                .medium-priority { background-color: #f39c12; }
                .low-priority { background-color: #27ae60; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report_info.title }}</h1>
                <p>生成时间: {{ report_info.timestamp }}</p>
                <p>版本: {{ report_info.version }}</p>
            </div>
            
            <div class="summary">
                <h2>性能摘要</h2>
                <div class="metric">性能分数: {{ "%.2f"|format(performance_summary.score) }}</div>
                <div class="metric">性能等级: {{ performance_summary.level }}</div>
                <div class="metric">收敛率: {{ "%.1f"|format(performance_summary.convergence_rate * 100) }}%</div>
            </div>
            
            <div class="section">
                <h2>优化机会</h2>
                {% for opportunity in optimization_opportunities %}
                <div class="recommendation {{ opportunity.priority }}-priority">
                    <strong>{{ opportunity.module }} - {{ opportunity.type }}</strong><br>
                    {{ opportunity.description }}<br>
                    <em>建议: {{ opportunity.suggestions|join(', ') }}</em>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>实施路线图</h2>
                {% for phase in implementation_roadmap %}
                <div class="recommendation {{ phase.priority }}-priority">
                    <strong>阶段 {{ phase.phase }}: {{ phase.action }}</strong><br>
                    描述: {{ phase.description }}<br>
                    优先级: {{ phase.priority }} | 预期影响: {{ phase.expected_impact }} | 实施难度: {{ phase.implementation_effort }}<br>
                    预计时间: {{ phase.estimated_duration }}<br>
                    依赖: {{ phase.dependencies|join(', ') }}
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        
        from jinja2 import Template
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)





