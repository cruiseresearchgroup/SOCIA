"""
SOCIA工作流集成模块
支持SOCIA的verification_results约束设计、feedback机制、迭代式改进
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class SOCIAVerificationConstraints:
    """SOCIA验证约束处理器"""
    
    def __init__(self):
        self.verification_results = {}
        self.constraints = {}
        self.issues = []
    
    def load_verification_results(self, verification_path: Union[str, Path]) -> None:
        """加载验证结果"""
        try:
            with open(verification_path, 'r', encoding='utf-8') as f:
                self.verification_results = json.load(f)
            
            self._extract_constraints()
            self._generate_parameter_constraints()
            
            logger.info("验证结果加载完成")
        except Exception as e:
            logger.error(f"验证结果加载失败: {e}")
            raise
    
    def _extract_constraints(self) -> None:
        """提取约束"""
        if 'issues' in self.verification_results:
            self.issues = self.verification_results['issues']
        else:
            logger.warning("验证结果中没有找到问题列表")
    
    def _generate_parameter_constraints(self) -> None:
        """生成参数约束"""
        self.constraints = {
            'parameter_bounds': {},
            'convergence_requirements': {},
            'quality_thresholds': {},
            'implementation_constraints': {}
        }
        
        # 基于问题生成约束
        for issue in self.issues:
            if issue.get('type') == 'parameter_bound':
                param_name = issue.get('parameter')
                bounds = issue.get('bounds')
                if param_name and bounds:
                    self.constraints['parameter_bounds'][param_name] = bounds
            
            elif issue.get('type') == 'convergence':
                module_name = issue.get('module')
                requirements = issue.get('requirements')
                if module_name and requirements:
                    self.constraints['convergence_requirements'][module_name] = requirements
            
            elif issue.get('type') == 'quality':
                threshold = issue.get('threshold')
                metric = issue.get('metric')
                if threshold and metric:
                    self.constraints['quality_thresholds'][metric] = threshold
            
            elif issue.get('type') == 'implementation':
                constraint = {
                    'description': issue.get('description', ''),
                    'severity': issue.get('severity', 'medium'),
                    'solution': issue.get('solution', '')
                }
                self.constraints['implementation_constraints'][issue.get('location', 'unknown')] = constraint
    
    def get_parameter_constraints(self, module_name: str) -> Dict[str, Any]:
        """获取模块的参数约束"""
        module_constraints = {}
        
        for param_name, bounds in self.constraints['parameter_bounds'].items():
            if param_name.startswith(module_name) or module_name in param_name:
                module_constraints[param_name] = bounds
        
        return module_constraints
    
    def get_convergence_requirements(self, module_name: str) -> Dict[str, Any]:
        """获取模块的收敛要求"""
        return self.constraints['convergence_requirements'].get(module_name, {})
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """获取质量阈值"""
        return self.constraints['quality_thresholds']
    
    def get_implementation_constraints(self) -> Dict[str, Any]:
        """获取实现约束"""
        return self.constraints['implementation_constraints']

class SOCIAFeedbackManager:
    """SOCIA反馈管理器"""
    
    def __init__(self):
        self.feedback_history = []
        self.current_feedback = {}
        self.improvement_suggestions = []
    
    def load_feedback(self, feedback_path: Union[str, Path]) -> None:
        """加载反馈"""
        try:
            with open(feedback_path, 'r', encoding='utf-8') as f:
                self.current_feedback = json.load(f)
            
            self._extract_improvement_suggestions()
            logger.info("反馈加载完成")
        except Exception as e:
            logger.warning(f"反馈加载失败: {e}")
            self.current_feedback = {}
    
    def _extract_improvement_suggestions(self) -> None:
        """提取改进建议"""
        if 'suggestions' in self.current_feedback:
            self.improvement_suggestions = self.current_feedback['suggestions']
        else:
            logger.warning("反馈中没有找到改进建议")
    
    def generate_feedback(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成反馈"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'calibration_results': calibration_results,
            'suggestions': self._generate_suggestions(calibration_results),
            'next_steps': self._generate_next_steps(calibration_results)
        }
        
        self.feedback_history.append(feedback)
        return feedback
    
    def _generate_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """生成建议"""
        suggestions = []
        
        # 基于校准结果生成建议
        for module_name, module_results in results.get('calibration_results', {}).items():
            if not module_results.get('convergence_achieved', False):
                suggestions.append(f"模块 {module_name} 未收敛，建议调整参数空间或增加迭代次数")
            
            final_metrics = module_results.get('final_metrics', {})
            if final_metrics.get('rmse', 1.0) > 0.5:
                suggestions.append(f"模块 {module_name} 的RMSE较高，建议检查参数约束或数据质量")
            
            if final_metrics.get('r2', 0.0) < 0.7:
                suggestions.append(f"模块 {module_name} 的R²较低，建议优化摘要统计设计")
        
        return suggestions
    
    def _generate_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """生成下一步建议"""
        next_steps = []
        
        # 基于校准结果生成下一步
        convergence_count = sum(
            1 for module_results in results.get('calibration_results', {}).values()
            if module_results.get('convergence_achieved', False)
        )
        
        total_modules = len(results.get('calibration_results', {}))
        
        if convergence_count == total_modules:
            next_steps.append("所有模块校准完成，可以进行预测和验证")
        elif convergence_count > 0:
            next_steps.append(f"{convergence_count}/{total_modules} 个模块校准完成，继续校准剩余模块")
        else:
            next_steps.append("所有模块校准失败，建议检查参数空间和约束设置")
        
        return next_steps
    
    def save_feedback(self, feedback: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """保存反馈"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        logger.info(f"反馈保存完成: {output_path}")

class SOCIAIterativeImprovement:
    """SOCIA迭代式改进管理器"""
    
    def __init__(self):
        self.improvement_history = []
        self.current_iteration = 0
        self.best_results = {}
        self.improvement_trends = {}
    
    def start_iteration(self, iteration_id: int) -> None:
        """开始迭代"""
        self.current_iteration = iteration_id
        self.improvement_history.append({
            'iteration_id': iteration_id,
            'start_time': datetime.now().isoformat(),
            'status': 'started'
        })
        
        logger.info(f"开始迭代 {iteration_id}")
    
    def update_iteration(self, results: Dict[str, Any]) -> None:
        """更新迭代结果"""
        if self.improvement_history:
            self.improvement_history[-1].update({
                'results': results,
                'end_time': datetime.now().isoformat(),
                'status': 'completed'
            })
        
        # 更新最佳结果
        self._update_best_results(results)
        
        # 分析改进趋势
        self._analyze_improvement_trends()
    
    def _update_best_results(self, results: Dict[str, Any]) -> None:
        """更新最佳结果"""
        for module_name, module_results in results.get('calibration_results', {}).items():
            if module_name not in self.best_results:
                self.best_results[module_name] = module_results
            else:
                current_rmse = module_results.get('final_metrics', {}).get('rmse', float('inf'))
                best_rmse = self.best_results[module_name].get('final_metrics', {}).get('rmse', float('inf'))
                
                if current_rmse < best_rmse:
                    self.best_results[module_name] = module_results
    
    def _analyze_improvement_trends(self) -> None:
        """分析改进趋势"""
        if len(self.improvement_history) < 2:
            return
        
        # 分析RMSE趋势
        rmse_trends = {}
        for module_name in self.best_results.keys():
            rmse_values = []
            for iteration in self.improvement_history:
                if 'results' in iteration:
                    module_results = iteration['results'].get('calibration_results', {}).get(module_name, {})
                    rmse = module_results.get('final_metrics', {}).get('rmse', None)
                    if rmse is not None:
                        rmse_values.append(rmse)
            
            if len(rmse_values) >= 2:
                trend = 'improving' if rmse_values[-1] < rmse_values[0] else 'degrading'
                rmse_trends[module_name] = {
                    'trend': trend,
                    'values': rmse_values,
                    'improvement_rate': (rmse_values[0] - rmse_values[-1]) / rmse_values[0] if rmse_values[0] > 0 else 0
                }
        
        self.improvement_trends = rmse_trends
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        suggestions = []
        
        for module_name, trends in self.improvement_trends.items():
            if trends['trend'] == 'degrading':
                suggestions.append(f"模块 {module_name} 性能下降，建议调整参数空间或校准策略")
            elif trends['improvement_rate'] < 0.1:
                suggestions.append(f"模块 {module_name} 改进缓慢，建议增加迭代次数或调整摘要统计")
        
        return suggestions
    
    def should_continue_iteration(self) -> bool:
        """判断是否应该继续迭代"""
        if len(self.improvement_history) < 2:
            return True
        
        # 检查是否有改进
        recent_improvements = sum(
            1 for trends in self.improvement_trends.values()
            if trends['trend'] == 'improving'
        )
        
        return recent_improvements > 0
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """获取迭代摘要"""
        return {
            'current_iteration': self.current_iteration,
            'total_iterations': len(self.improvement_history),
            'best_results': self.best_results,
            'improvement_trends': self.improvement_trends,
            'should_continue': self.should_continue_iteration()
        }

class SOCIAResultFormatter:
    """SOCIA结果格式化器"""
    
    def __init__(self):
        self.result_templates = {}
        self.output_formats = ['json', 'csv', 'html']
    
    def format_calibration_results(self, results: Dict[str, Any], 
                                 format_type: str = 'json') -> Union[Dict, str, pd.DataFrame]:
        """格式化校准结果"""
        if format_type == 'json':
            return self._format_json_results(results)
        elif format_type == 'csv':
            return self._format_csv_results(results)
        elif format_type == 'html':
            return self._format_html_results(results)
        else:
            raise ValueError(f"不支持的结果格式: {format_type}")
    
    def _format_json_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """格式化JSON结果"""
        formatted_results = {
            'calibration_summary': {
                'total_modules': len(results.get('calibration_results', {})),
                'converged_modules': sum(
                    1 for module_results in results.get('calibration_results', {}).values()
                    if module_results.get('convergence_achieved', False)
                ),
                'overall_rmse': self._calculate_overall_rmse(results),
                'calibration_time': results.get('calibration_time', 0)
            },
            'module_results': results.get('calibration_results', {}),
            'performance_metrics': results.get('performance_summary', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        return formatted_results
    
    def _format_csv_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """格式化CSV结果"""
        rows = []
        
        for module_name, module_results in results.get('calibration_results', {}).items():
            row = {
                'module_name': module_name,
                'convergence_achieved': module_results.get('convergence_achieved', False),
                'rmse': module_results.get('final_metrics', {}).get('rmse', 0),
                'mae': module_results.get('final_metrics', {}).get('mae', 0),
                'r2': module_results.get('final_metrics', {}).get('r2', 0)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _format_html_results(self, results: Dict[str, Any]) -> str:
        """格式化HTML结果"""
        html = f"""
        <html>
        <head>
            <title>SOCIA SBI校准结果</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .module {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 5px 10px; }}
            </style>
        </head>
        <body>
            <h1>SOCIA SBI校准结果</h1>
            <div class="summary">
                <h2>校准摘要</h2>
                <p>总模块数: {len(results.get('calibration_results', {}))}</p>
                <p>收敛模块数: {sum(1 for module_results in results.get('calibration_results', {}).values() if module_results.get('convergence_achieved', False))}</p>
                <p>总体RMSE: {self._calculate_overall_rmse(results):.4f}</p>
            </div>
        """
        
        for module_name, module_results in results.get('calibration_results', {}).items():
            html += f"""
            <div class="module">
                <h3>模块: {module_name}</h3>
                <div class="metric">收敛状态: {'是' if module_results.get('convergence_achieved', False) else '否'}</div>
                <div class="metric">RMSE: {module_results.get('final_metrics', {}).get('rmse', 0):.4f}</div>
                <div class="metric">MAE: {module_results.get('final_metrics', {}).get('mae', 0):.4f}</div>
                <div class="metric">R²: {module_results.get('final_metrics', {}).get('r2', 0):.4f}</div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _calculate_overall_rmse(self, results: Dict[str, Any]) -> float:
        """计算总体RMSE"""
        rmse_values = []
        for module_results in results.get('calibration_results', {}).values():
            rmse = module_results.get('final_metrics', {}).get('rmse', 0)
            if rmse > 0:
                rmse_values.append(rmse)
        
        return np.mean(rmse_values) if rmse_values else 0.0
    
    def save_formatted_results(self, results: Dict[str, Any], 
                             output_dir: Union[str, Path],
                             formats: List[str] = ['json', 'csv', 'html']) -> None:
        """保存格式化结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for format_type in formats:
            if format_type == 'json':
                formatted_results = self._format_json_results(results)
                output_path = output_dir / "calibration_results.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_results, f, indent=2, ensure_ascii=False)
            
            elif format_type == 'csv':
                formatted_results = self._format_csv_results(results)
                output_path = output_dir / "calibration_results.csv"
                formatted_results.to_csv(output_path, index=False)
            
            elif format_type == 'html':
                formatted_results = self._format_html_results(results)
                output_path = output_dir / "calibration_results.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_results)
            
            logger.info(f"结果保存完成: {output_path}")

class SOCIAWorkflowIntegrator:
    """SOCIA工作流集成器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.verification_constraints = SOCIAVerificationConstraints()
        self.feedback_manager = SOCIAFeedbackManager()
        self.iterative_improvement = SOCIAIterativeImprovement()
        self.result_formatter = SOCIAResultFormatter()
    
    def load_socia_workflow_configs(self) -> None:
        """加载SOCIA工作流配置"""
        try:
            # 加载验证结果
            verification_path = self.output_dir / "verification_results_iter_1.json"
            if verification_path.exists():
                self.verification_constraints.load_verification_results(verification_path)
            
            # 加载反馈
            feedback_path = self.output_dir / "feedback_iter_1.json"
            if feedback_path.exists():
                self.feedback_manager.load_feedback(feedback_path)
            
            logger.info("SOCIA工作流配置加载完成")
        except Exception as e:
            logger.warning(f"SOCIA工作流配置加载失败: {e}")
    
    def get_parameter_constraints_for_module(self, module_name: str) -> Dict[str, Any]:
        """获取模块的参数约束"""
        return self.verification_constraints.get_parameter_constraints(module_name)
    
    def get_convergence_requirements_for_module(self, module_name: str) -> Dict[str, Any]:
        """获取模块的收敛要求"""
        return self.verification_constraints.get_convergence_requirements(module_name)
    
    def generate_feedback(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成反馈"""
        return self.feedback_manager.generate_feedback(calibration_results)
    
    def start_iteration(self, iteration_id: int) -> None:
        """开始迭代"""
        self.iterative_improvement.start_iteration(iteration_id)
    
    def update_iteration(self, results: Dict[str, Any]) -> None:
        """更新迭代"""
        self.iterative_improvement.update_iteration(results)
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        return self.iterative_improvement.get_improvement_suggestions()
    
    def should_continue_iteration(self) -> bool:
        """判断是否应该继续迭代"""
        return self.iterative_improvement.should_continue_iteration()
    
    def format_and_save_results(self, results: Dict[str, Any], 
                               output_dir: Optional[Union[str, Path]] = None) -> None:
        """格式化并保存结果"""
        if output_dir is None:
            output_dir = self.output_dir / "sbi_results"
        
        self.result_formatter.save_formatted_results(results, output_dir)
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """获取工作流摘要"""
        return {
            'verification_constraints': self.verification_constraints.constraints,
            'feedback_history': len(self.feedback_manager.feedback_history),
            'iteration_summary': self.iterative_improvement.get_iteration_summary()
        }





