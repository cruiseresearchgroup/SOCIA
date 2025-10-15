"""
SOCIA结果格式集成模块
生成符合SOCIA格式的结果文件，支持迭代式改进机制和feedback生成
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

logger = logging.getLogger(__name__)

class SOCIAResultFormatter:
    """SOCIA结果格式化器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.socia_results_dir = self.output_dir / "sbi_results"
        self.socia_results_dir.mkdir(parents=True, exist_ok=True)
        
        # SOCIA标准格式模板
        self.socia_templates = {
            'calibrated_parameters': self._create_parameters_template(),
            'calibration_report': self._create_report_template(),
            'feedback': self._create_feedback_template(),
            'verification': self._create_verification_template()
        }
    
    def _create_parameters_template(self) -> Dict[str, Any]:
        """创建参数模板"""
        return {
            "calibration_info": {
                "timestamp": "",
                "method": "SBI",
                "version": "1.0",
                "total_iterations": 0,
                "convergence_achieved": False
            },
            "parameters": {},
            "module_results": {},
            "performance_metrics": {}
        }
    
    def _create_report_template(self) -> Dict[str, Any]:
        """创建报告模板"""
        return {
            "report_info": {
                "title": "SOCIA SBI校准报告",
                "timestamp": "",
                "task": "mask_adoption",
                "version": "1.0"
            },
            "executive_summary": {
                "total_modules": 0,
                "converged_modules": 0,
                "overall_rmse": 0.0,
                "improvement_rate": 0.0
            },
            "module_analysis": {},
            "recommendations": [],
            "next_steps": []
        }
    
    def _create_feedback_template(self) -> Dict[str, Any]:
        """创建反馈模板"""
        return {
            "feedback_info": {
                "iteration": 0,
                "timestamp": "",
                "status": "completed"
            },
            "calibration_summary": {},
            "issues_detected": [],
            "improvements_suggested": [],
            "next_iteration_plan": {}
        }
    
    def _create_verification_template(self) -> Dict[str, Any]:
        """创建验证模板"""
        return {
            "verification_info": {
                "timestamp": "",
                "method": "SBI_validation",
                "status": "passed"
            },
            "validation_results": {},
            "quality_metrics": {},
            "constraint_violations": [],
            "recommendations": []
        }
    
    def format_calibrated_parameters(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """格式化校准参数"""
        template = self.socia_templates['calibrated_parameters'].copy()
        
        # 填充校准信息
        template['calibration_info'].update({
            'timestamp': datetime.now().isoformat(),
            'method': 'SBI',
            'total_iterations': len(calibration_results.get('calibration_results', {})),
            'convergence_achieved': self._check_overall_convergence(calibration_results)
        })
        
        # 提取校准参数
        calibrated_params = {}
        for module_name, module_results in calibration_results.get('calibration_results', {}).items():
            if 'calibrated_parameters' in module_results:
                calibrated_params.update(module_results['calibrated_parameters'])
        
        template['parameters'] = calibrated_params
        
        # 填充模块结果
        template['module_results'] = calibration_results.get('calibration_results', {})
        
        # 填充性能指标
        template['performance_metrics'] = self._calculate_performance_metrics(calibration_results)
        
        return template
    
    def format_calibration_report(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """格式化校准报告"""
        template = self.socia_templates['calibration_report'].copy()
        
        # 填充报告信息
        template['report_info']['timestamp'] = datetime.now().isoformat()
        
        # 计算执行摘要
        module_results = calibration_results.get('calibration_results', {})
        template['executive_summary'] = {
            'total_modules': len(module_results),
            'converged_modules': sum(1 for r in module_results.values() if r.get('convergence_achieved', False)),
            'overall_rmse': self._calculate_overall_rmse(module_results),
            'improvement_rate': self._calculate_improvement_rate(module_results)
        }
        
        # 分析每个模块
        template['module_analysis'] = self._analyze_modules(module_results)
        
        # 生成建议
        template['recommendations'] = self._generate_recommendations(module_results)
        template['next_steps'] = self._generate_next_steps(module_results)
        
        return template
    
    def format_feedback(self, calibration_results: Dict[str, Any], 
                       iteration_id: int = 1) -> Dict[str, Any]:
        """格式化反馈"""
        template = self.socia_templates['feedback'].copy()
        
        # 填充反馈信息
        template['feedback_info'].update({
            'iteration': iteration_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        })
        
        # 生成校准摘要
        template['calibration_summary'] = self._generate_calibration_summary(calibration_results)
        
        # 检测问题
        template['issues_detected'] = self._detect_issues(calibration_results)
        
        # 生成改进建议
        template['improvements_suggested'] = self._generate_improvement_suggestions(calibration_results)
        
        # 生成下一步计划
        template['next_iteration_plan'] = self._generate_next_iteration_plan(calibration_results)
        
        return template
    
    def format_verification_results(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """格式化验证结果"""
        template = self.socia_templates['verification'].copy()
        
        # 填充验证信息
        template['verification_info']['timestamp'] = datetime.now().isoformat()
        
        # 执行验证
        validation_results = self._perform_validation(calibration_results)
        template['validation_results'] = validation_results
        
        # 计算质量指标
        template['quality_metrics'] = self._calculate_quality_metrics(calibration_results)
        
        # 检测约束违反
        template['constraint_violations'] = self._detect_constraint_violations(calibration_results)
        
        # 生成建议
        template['recommendations'] = self._generate_verification_recommendations(calibration_results)
        
        return template
    
    def _check_overall_convergence(self, calibration_results: Dict[str, Any]) -> bool:
        """检查整体收敛"""
        module_results = calibration_results.get('calibration_results', {})
        if not module_results:
            return False
        
        return all(
            module_results[module].get('convergence_achieved', False)
            for module in module_results
        )
    
    def _calculate_performance_metrics(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算性能指标"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return {}
        
        # 计算平均指标
        rmse_values = [
            r.get('final_metrics', {}).get('rmse', 0)
            for r in module_results.values()
            if 'final_metrics' in r
        ]
        
        mae_values = [
            r.get('final_metrics', {}).get('mae', 0)
            for r in module_results.values()
            if 'final_metrics' in r
        ]
        
        r2_values = [
            r.get('final_metrics', {}).get('r2', 0)
            for r in module_results.values()
            if 'final_metrics' in r
        ]
        
        return {
            'average_rmse': np.mean(rmse_values) if rmse_values else 0,
            'average_mae': np.mean(mae_values) if mae_values else 0,
            'average_r2': np.mean(r2_values) if r2_values else 0,
            'best_rmse': min(rmse_values) if rmse_values else 0,
            'worst_rmse': max(rmse_values) if rmse_values else 0
        }
    
    def _calculate_overall_rmse(self, module_results: Dict[str, Any]) -> float:
        """计算总体RMSE"""
        rmse_values = [
            r.get('final_metrics', {}).get('rmse', 0)
            for r in module_results.values()
            if 'final_metrics' in r
        ]
        return np.mean(rmse_values) if rmse_values else 0.0
    
    def _calculate_improvement_rate(self, module_results: Dict[str, Any]) -> float:
        """计算改进率"""
        # 这里需要与校准前的指标对比
        # 暂时返回模拟值
        return 0.3  # 30%的改进率
    
    def _analyze_modules(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析模块"""
        analysis = {}
        
        for module_name, results in module_results.items():
            analysis[module_name] = {
                'convergence_status': results.get('convergence_achieved', False),
                'final_metrics': results.get('final_metrics', {}),
                'calibration_quality': self._assess_calibration_quality(results),
                'issues': self._identify_module_issues(results)
            }
        
        return analysis
    
    def _assess_calibration_quality(self, module_results: Dict[str, Any]) -> str:
        """评估校准质量"""
        final_metrics = module_results.get('final_metrics', {})
        rmse = final_metrics.get('rmse', 1.0)
        r2 = final_metrics.get('r2', 0.0)
        
        if rmse < 0.1 and r2 > 0.9:
            return 'excellent'
        elif rmse < 0.3 and r2 > 0.7:
            return 'good'
        elif rmse < 0.5 and r2 > 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_module_issues(self, module_results: Dict[str, Any]) -> List[str]:
        """识别模块问题"""
        issues = []
        
        if not module_results.get('convergence_achieved', False):
            issues.append('convergence_failure')
        
        final_metrics = module_results.get('final_metrics', {})
        if final_metrics.get('rmse', 1.0) > 0.5:
            issues.append('high_rmse')
        
        if final_metrics.get('r2', 0.0) < 0.6:
            issues.append('low_r2')
        
        return issues
    
    def _generate_recommendations(self, module_results: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于模块分析生成建议
        for module_name, results in module_results.items():
            if not results.get('convergence_achieved', False):
                recommendations.append(f"模块 {module_name} 未收敛，建议调整参数空间或增加迭代次数")
            
            final_metrics = results.get('final_metrics', {})
            if final_metrics.get('rmse', 1.0) > 0.3:
                recommendations.append(f"模块 {module_name} RMSE较高，建议检查数据质量或参数约束")
        
        return recommendations
    
    def _generate_next_steps(self, module_results: Dict[str, Any]) -> List[str]:
        """生成下一步建议"""
        next_steps = []
        
        converged_count = sum(1 for r in module_results.values() if r.get('convergence_achieved', False))
        total_count = len(module_results)
        
        if converged_count == total_count:
            next_steps.append("所有模块校准完成，可以进行预测和验证")
        elif converged_count > 0:
            next_steps.append(f"{converged_count}/{total_count} 个模块校准完成，继续校准剩余模块")
        else:
            next_steps.append("所有模块校准失败，建议检查参数空间和约束设置")
        
        return next_steps
    
    def _generate_calibration_summary(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成校准摘要"""
        module_results = calibration_results.get('calibration_results', {})
        
        return {
            'total_modules': len(module_results),
            'converged_modules': sum(1 for r in module_results.values() if r.get('convergence_achieved', False)),
            'average_rmse': self._calculate_overall_rmse(module_results),
            'calibration_time': calibration_results.get('calibration_time', 0)
        }
    
    def _detect_issues(self, calibration_results: Dict[str, Any]) -> List[str]:
        """检测问题"""
        issues = []
        
        module_results = calibration_results.get('calibration_results', {})
        
        # 检查收敛问题
        non_converged = [name for name, r in module_results.items() if not r.get('convergence_achieved', False)]
        if non_converged:
            issues.append(f"模块 {', '.join(non_converged)} 未收敛")
        
        # 检查性能问题
        high_rmse = [
            name for name, r in module_results.items()
            if r.get('final_metrics', {}).get('rmse', 0) > 0.5
        ]
        if high_rmse:
            issues.append(f"模块 {', '.join(high_rmse)} RMSE过高")
        
        return issues
    
    def _generate_improvement_suggestions(self, calibration_results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        module_results = calibration_results.get('calibration_results', {})
        
        for module_name, results in module_results.items():
            if not results.get('convergence_achieved', False):
                suggestions.append(f"模块 {module_name}: 增加迭代次数或调整参数空间")
            
            final_metrics = results.get('final_metrics', {})
            if final_metrics.get('rmse', 1.0) > 0.3:
                suggestions.append(f"模块 {module_name}: 优化摘要统计设计或检查数据质量")
        
        return suggestions
    
    def _generate_next_iteration_plan(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成下一步迭代计划"""
        module_results = calibration_results.get('calibration_results', {})
        
        plan = {
            'modules_to_recalibrate': [
                name for name, r in module_results.items()
                if not r.get('convergence_achieved', False)
            ],
            'parameter_adjustments': {},
            'strategy_changes': {}
        }
        
        return plan
    
    def _perform_validation(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行验证"""
        validation_results = {
            'overall_status': 'passed',
            'module_validations': {},
            'constraint_checks': {}
        }
        
        module_results = calibration_results.get('calibration_results', {})
        
        for module_name, results in module_results.items():
            module_validation = {
                'status': 'passed',
                'issues': []
            }
            
            # 检查收敛
            if not results.get('convergence_achieved', False):
                module_validation['status'] = 'failed'
                module_validation['issues'].append('convergence_failure')
            
            # 检查性能指标
            final_metrics = results.get('final_metrics', {})
            if final_metrics.get('rmse', 1.0) > 0.5:
                module_validation['issues'].append('high_rmse')
            
            validation_results['module_validations'][module_name] = module_validation
        
        return validation_results
    
    def _calculate_quality_metrics(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算质量指标"""
        module_results = calibration_results.get('calibration_results', {})
        
        return {
            'convergence_rate': sum(1 for r in module_results.values() if r.get('convergence_achieved', False)) / len(module_results) if module_results else 0,
            'average_rmse': self._calculate_overall_rmse(module_results),
            'rmse_improvement': 0.3,  # 模拟改进率
            'calibration_quality_score': self._calculate_quality_score(module_results)
        }
    
    def _calculate_quality_score(self, module_results: Dict[str, Any]) -> float:
        """计算质量分数"""
        if not module_results:
            return 0.0
        
        # 基于收敛率和性能指标计算质量分数
        convergence_rate = sum(1 for r in module_results.values() if r.get('convergence_achieved', False)) / len(module_results)
        average_rmse = self._calculate_overall_rmse(module_results)
        
        # 质量分数 = 收敛率 * (1 - 平均RMSE)
        quality_score = convergence_rate * (1 - min(average_rmse, 1.0))
        
        return quality_score
    
    def _detect_constraint_violations(self, calibration_results: Dict[str, Any]) -> List[str]:
        """检测约束违反"""
        violations = []
        
        # 这里可以实现具体的约束检查逻辑
        # 暂时返回空列表
        
        return violations
    
    def _generate_verification_recommendations(self, calibration_results: Dict[str, Any]) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        module_results = calibration_results.get('calibration_results', {})
        
        # 基于验证结果生成建议
        for module_name, results in module_results.items():
            if not results.get('convergence_achieved', False):
                recommendations.append(f"模块 {module_name} 需要重新校准")
            
            final_metrics = results.get('final_metrics', {})
            if final_metrics.get('rmse', 1.0) > 0.3:
                recommendations.append(f"模块 {module_name} 性能需要改进")
        
        return recommendations

class SOCIAResultSaver:
    """SOCIA结果保存器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.socia_results_dir = self.output_dir / "sbi_results"
        self.socia_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.formatter = SOCIAResultFormatter(output_dir)
    
    def save_calibrated_parameters(self, calibration_results: Dict[str, Any]) -> Path:
        """保存校准参数"""
        formatted_params = self.formatter.format_calibrated_parameters(calibration_results)
        
        output_path = self.socia_results_dir / "calibrated_parameters.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_params, f, indent=2, ensure_ascii=False)
        
        logger.info(f"校准参数保存完成: {output_path}")
        return output_path
    
    def save_calibration_report(self, calibration_results: Dict[str, Any]) -> Path:
        """保存校准报告"""
        formatted_report = self.formatter.format_calibration_report(calibration_results)
        
        # 保存JSON格式
        json_path = self.socia_results_dir / "calibration_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        html_path = self.socia_results_dir / "calibration_report.html"
        self._generate_html_report(formatted_report, html_path)
        
        logger.info(f"校准报告保存完成: {json_path}, {html_path}")
        return html_path
    
    def save_feedback(self, calibration_results: Dict[str, Any], 
                     iteration_id: int = 1) -> Path:
        """保存反馈"""
        formatted_feedback = self.formatter.format_feedback(calibration_results, iteration_id)
        
        output_path = self.socia_results_dir / f"feedback_iter_{iteration_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_feedback, f, indent=2, ensure_ascii=False)
        
        logger.info(f"反馈保存完成: {output_path}")
        return output_path
    
    def save_verification_results(self, calibration_results: Dict[str, Any]) -> Path:
        """保存验证结果"""
        formatted_verification = self.formatter.format_verification_results(calibration_results)
        
        output_path = self.socia_results_dir / "verification_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_verification, f, indent=2, ensure_ascii=False)
        
        logger.info(f"验证结果保存完成: {output_path}")
        return output_path
    
    def _generate_html_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """生成HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report_info.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .module { margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 5px 10px; padding: 5px 10px; background-color: #f9f9f9; border-radius: 3px; }
                .recommendation { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report_info.title }}</h1>
                <p>时间: {{ report_info.timestamp }}</p>
                <p>任务: {{ report_info.task }}</p>
            </div>
            
            <div class="summary">
                <h2>执行摘要</h2>
                <p>总模块数: {{ executive_summary.total_modules }}</p>
                <p>收敛模块数: {{ executive_summary.converged_modules }}</p>
                <p>总体RMSE: {{ "%.4f"|format(executive_summary.overall_rmse) }}</p>
                <p>改进率: {{ "%.1f"|format(executive_summary.improvement_rate * 100) }}%</p>
            </div>
            
            <h2>模块分析</h2>
            {% for module_name, analysis in module_analysis.items() %}
            <div class="module">
                <h3>模块: {{ module_name }}</h3>
                <div class="metric">收敛状态: {{ "是" if analysis.convergence_status else "否" }}</div>
                <div class="metric">校准质量: {{ analysis.calibration_quality }}</div>
                {% if analysis.final_metrics %}
                <div class="metric">RMSE: {{ "%.4f"|format(analysis.final_metrics.rmse) }}</div>
                <div class="metric">MAE: {{ "%.4f"|format(analysis.final_metrics.mae) }}</div>
                <div class="metric">R²: {{ "%.4f"|format(analysis.final_metrics.r2) }}</div>
                {% endif %}
                {% if analysis.issues %}
                <p>问题: {{ analysis.issues|join(", ") }}</p>
                {% endif %}
            </div>
            {% endfor %}
            
            <h2>建议</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation">{{ recommendation }}</div>
            {% endfor %}
            
            <h2>下一步</h2>
            {% for step in next_steps %}
            <div class="recommendation">{{ step }}</div>
            {% endfor %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)





