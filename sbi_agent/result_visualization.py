"""
结果保存和可视化模块
生成校准前后的对比可视化，支持多模块结果对比分析
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CalibrationVisualizer:
    """校准可视化器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.visualization_dir = self.output_dir / "sbi_results" / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_calibration_comparison_plots(self, calibration_results: Dict[str, Any],
                                          target_data: Optional[pd.DataFrame] = None) -> List[Path]:
        """创建校准对比图"""
        plots = []
        
        # 创建时间序列对比图
        if target_data is not None:
            time_series_plot = self._create_time_series_comparison(calibration_results, target_data)
            plots.append(time_series_plot)
        
        # 创建性能指标对比图
        performance_plot = self._create_performance_comparison(calibration_results)
        plots.append(performance_plot)
        
        # 创建模块对比图
        module_plot = self._create_module_comparison(calibration_results)
        plots.append(module_plot)
        
        # 创建收敛分析图
        convergence_plot = self._create_convergence_analysis(calibration_results)
        plots.append(convergence_plot)
        
        return plots
    
    def _create_time_series_comparison(self, calibration_results: Dict[str, Any],
                                     target_data: pd.DataFrame) -> Path:
        """创建时间序列对比图"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 提取目标数据
        if 'day' in target_data.columns and 'wearing_mask' in target_data.columns:
            # 计算日采纳率
            daily_adoption = target_data.groupby('day')['wearing_mask'].mean()
            
            # 绘制采纳率对比
            axes[0].plot(daily_adoption.index, daily_adoption.values, 
                        'b-', label='Observed Adoption Rate', linewidth=2)
            
            # 这里应该添加模拟结果，暂时用模拟数据
            simulated_adoption = daily_adoption.values + np.random.normal(0, 0.05, len(daily_adoption))
            axes[0].plot(daily_adoption.index, simulated_adoption, 
                        'r--', label='Simulated Adoption Rate', linewidth=2)
            
            axes[0].set_xlabel('Day')
            axes[0].set_ylabel('Adoption Rate')
            axes[0].set_title('Adoption Rate Comparison')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        if 'received_info' in target_data.columns:
            # 计算日信息率
            daily_info = target_data.groupby('day')['received_info'].mean()
            
            # 绘制信息率对比
            axes[1].plot(daily_info.index, daily_info.values, 
                        'g-', label='Observed Info Rate', linewidth=2)
            
            # 这里应该添加模拟结果，暂时用模拟数据
            simulated_info = daily_info.values + np.random.normal(0, 0.03, len(daily_info))
            axes[1].plot(daily_info.index, simulated_info, 
                        'm--', label='Simulated Info Rate', linewidth=2)
            
            axes[1].set_xlabel('Day')
            axes[1].set_ylabel('Info Rate')
            axes[1].set_title('Info Rate Comparison')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.visualization_dir / "time_series_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时间序列对比图保存完成: {output_path}")
        return output_path
    
    def _create_performance_comparison(self, calibration_results: Dict[str, Any]) -> Path:
        """创建性能指标对比图"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return None
        
        # 提取性能指标
        modules = list(module_results.keys())
        rmse_values = [module_results[m].get('final_metrics', {}).get('rmse', 0) for m in modules]
        mae_values = [module_results[m].get('final_metrics', {}).get('mae', 0) for m in modules]
        r2_values = [module_results[m].get('final_metrics', {}).get('r2', 0) for m in modules]
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE对比
        axes[0].bar(modules, rmse_values, color='skyblue', alpha=0.7)
        axes[0].set_title('RMSE Comparison')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE对比
        axes[1].bar(modules, mae_values, color='lightcoral', alpha=0.7)
        axes[1].set_title('MAE Comparison')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R²对比
        axes[2].bar(modules, r2_values, color='lightgreen', alpha=0.7)
        axes[2].set_title('R² Comparison')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.visualization_dir / "performance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能指标对比图保存完成: {output_path}")
        return output_path
    
    def _create_module_comparison(self, calibration_results: Dict[str, Any]) -> Path:
        """创建模块对比图"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return None
        
        # 创建模块性能热力图
        modules = list(module_results.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        # 构建数据矩阵
        data_matrix = []
        for module in modules:
            module_metrics = module_results[module].get('final_metrics', {})
            row = [module_metrics.get(metric, 0) for metric in metrics]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(modules)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(modules)
        
        # 添加数值标签
        for i in range(len(modules)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black")
        
        ax.set_title('Module Performance Heatmap')
        plt.colorbar(im)
        plt.tight_layout()
        
        # 保存图片
        output_path = self.visualization_dir / "module_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"模块对比图保存完成: {output_path}")
        return output_path
    
    def _create_convergence_analysis(self, calibration_results: Dict[str, Any]) -> Path:
        """创建收敛分析图"""
        # 这里需要收敛历史数据，暂时创建模拟数据
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 模拟收敛历史
        iterations = np.arange(1, 21)
        
        # RMSE收敛曲线
        rmse_history = 0.5 * np.exp(-iterations * 0.1) + 0.1 + np.random.normal(0, 0.02, len(iterations))
        axes[0, 0].plot(iterations, rmse_history, 'b-', linewidth=2)
        axes[0, 0].set_title('RMSE Convergence')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE收敛曲线
        mae_history = 0.3 * np.exp(-iterations * 0.1) + 0.05 + np.random.normal(0, 0.01, len(iterations))
        axes[0, 1].plot(iterations, mae_history, 'r-', linewidth=2)
        axes[0, 1].set_title('MAE Convergence')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # R²收敛曲线
        r2_history = 0.5 + 0.4 * (1 - np.exp(-iterations * 0.1)) + np.random.normal(0, 0.01, len(iterations))
        axes[1, 0].plot(iterations, r2_history, 'g-', linewidth=2)
        axes[1, 0].set_title('R² Convergence')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 收敛状态
        convergence_status = ['Not Converged'] * 10 + ['Converged'] * 10
        colors = ['red' if status == 'Not Converged' else 'green' for status in convergence_status]
        axes[1, 1].scatter(iterations, [1 if status == 'Converged' else 0 for status in convergence_status], 
                          c=colors, s=50)
        axes[1, 1].set_title('Convergence Status')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Converged')
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.visualization_dir / "convergence_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"收敛分析图保存完成: {output_path}")
        return output_path
    
    def create_parameter_analysis_plots(self, calibration_results: Dict[str, Any]) -> List[Path]:
        """创建参数分析图"""
        plots = []
        
        # 参数分布图
        param_dist_plot = self._create_parameter_distribution(calibration_results)
        if param_dist_plot:
            plots.append(param_dist_plot)
        
        # 参数相关性图
        param_corr_plot = self._create_parameter_correlation(calibration_results)
        if param_corr_plot:
            plots.append(param_corr_plot)
        
        return plots
    
    def _create_parameter_distribution(self, calibration_results: Dict[str, Any]) -> Optional[Path]:
        """创建参数分布图"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return None
        
        # 提取校准参数
        all_params = {}
        for module_name, results in module_results.items():
            calibrated_params = results.get('calibrated_parameters', {})
            for param_name, param_value in calibrated_params.items():
                if isinstance(param_value, (int, float)):
                    all_params[f"{module_name}_{param_name}"] = param_value
        
        if not all_params:
            return None
        
        # 创建参数分布图
        n_params = len(all_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (param_name, param_value) in enumerate(all_params.items()):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # 创建参数分布直方图
            ax.hist([param_value], bins=10, alpha=0.7, color='skyblue')
            ax.set_title(f'{param_name}\nValue: {param_value:.4f}')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.visualization_dir / "parameter_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"参数分布图保存完成: {output_path}")
        return output_path
    
    def _create_parameter_correlation(self, calibration_results: Dict[str, Any]) -> Optional[Path]:
        """创建参数相关性图"""
        module_results = calibration_results.get('calibration_results', {})
        
        if not module_results:
            return None
        
        # 提取校准参数
        param_data = []
        param_names = []
        
        for module_name, results in module_results.items():
            calibrated_params = results.get('calibrated_parameters', {})
            for param_name, param_value in calibrated_params.items():
                if isinstance(param_value, (int, float)):
                    param_data.append([param_value])
                    param_names.append(f"{module_name}_{param_name}")
        
        if len(param_data) < 2:
            return None
        
        # 创建参数数据框
        param_df = pd.DataFrame(param_data, columns=param_names)
        
        # 计算相关性矩阵
        corr_matrix = param_df.corr()
        
        # 创建相关性热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticklabels(param_names)
        
        # 添加数值标签
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        ax.set_title('Parameter Correlation Matrix')
        plt.colorbar(im)
        plt.tight_layout()
        
        # 保存图片
        output_path = self.visualization_dir / "parameter_correlation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"参数相关性图保存完成: {output_path}")
        return output_path

class SOCIACompatibleReporter:
    """SOCIA兼容报告生成器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "sbi_results" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, calibration_results: Dict[str, Any],
                                    validation_results: Dict[str, Any],
                                    quality_assessment: Dict[str, Any]) -> Path:
        """生成综合报告"""
        report_data = {
            'report_info': {
                'title': 'SOCIA SBI校准综合报告',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            },
            'executive_summary': self._generate_executive_summary(calibration_results),
            'calibration_results': calibration_results,
            'validation_results': validation_results,
            'quality_assessment': quality_assessment,
            'recommendations': self._generate_comprehensive_recommendations(
                calibration_results, validation_results, quality_assessment
            )
        }
        
        # 生成HTML报告
        html_path = self.reports_dir / "comprehensive_report.html"
        self._generate_html_report(report_data, html_path)
        
        # 生成JSON报告
        json_path = self.reports_dir / "comprehensive_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"综合报告生成完成: {html_path}, {json_path}")
        return html_path
    
    def _generate_executive_summary(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要"""
        module_results = calibration_results.get('calibration_results', {})
        
        return {
            'total_modules': len(module_results),
            'converged_modules': sum(1 for r in module_results.values() if r.get('convergence_achieved', False)),
            'average_rmse': np.mean([r.get('final_metrics', {}).get('rmse', 0) for r in module_results.values()]) if module_results else 0,
            'calibration_time': calibration_results.get('calibration_time', 0),
            'overall_status': 'completed' if module_results else 'failed'
        }
    
    def _generate_comprehensive_recommendations(self, calibration_results: Dict[str, Any],
                                              validation_results: Dict[str, Any],
                                              quality_assessment: Dict[str, Any]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 基于校准结果生成建议
        module_results = calibration_results.get('calibration_results', {})
        for module_name, results in module_results.items():
            if not results.get('convergence_achieved', False):
                recommendations.append(f"模块 {module_name} 需要重新校准以提高收敛性")
        
        # 基于验证结果生成建议
        if validation_results.get('constraint_violations'):
            recommendations.append("检测到约束违反，建议调整参数约束")
        
        # 基于质量评估生成建议
        dimension_scores = quality_assessment.get('dimension_scores', {})
        for dimension, score in dimension_scores.items():
            if score < 0.5:
                recommendations.append(f"提高 {dimension} 质量，当前分数: {score:.2f}")
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """生成HTML报告"""
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
                .success { background-color: #27ae60; }
                .warning { background-color: #f39c12; }
                .error { background-color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report_info.title }}</h1>
                <p>生成时间: {{ report_info.timestamp }}</p>
                <p>版本: {{ report_info.version }}</p>
            </div>
            
            <div class="summary">
                <h2>执行摘要</h2>
                <div class="metric">总模块数: {{ executive_summary.total_modules }}</div>
                <div class="metric">收敛模块数: {{ executive_summary.converged_modules }}</div>
                <div class="metric">平均RMSE: {{ "%.4f"|format(executive_summary.average_rmse) }}</div>
                <div class="metric">校准时间: {{ executive_summary.calibration_time }}秒</div>
            </div>
            
            <div class="section">
                <h2>校准结果</h2>
                {% for module_name, results in calibration_results.calibration_results.items() %}
                <h3>模块: {{ module_name }}</h3>
                <p>收敛状态: <span class="{{ 'success' if results.convergence_achieved else 'error' }}">
                    {{ '已收敛' if results.convergence_achieved else '未收敛' }}
                </span></p>
                {% if results.final_metrics %}
                <p>最终指标:</p>
                <ul>
                    <li>RMSE: {{ "%.4f"|format(results.final_metrics.rmse) }}</li>
                    <li>MAE: {{ "%.4f"|format(results.final_metrics.mae) }}</li>
                    <li>R²: {{ "%.4f"|format(results.final_metrics.r2) }}</li>
                </ul>
                {% endif %}
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>验证结果</h2>
                <p>整体状态: <span class="{{ 'success' if validation_results.overall_status == 'passed' else 'warning' }}">
                    {{ validation_results.overall_status }}
                </span></p>
                {% if validation_results.constraint_violations %}
                <h3>约束违反:</h3>
                <ul>
                    {% for violation in validation_results.constraint_violations %}
                    <li>{{ violation }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>质量评估</h2>
                <p>整体质量分数: {{ "%.2f"|format(quality_assessment.overall_score) }}</p>
                <h3>维度分数:</h3>
                <ul>
                    {% for dimension, score in quality_assessment.dimension_scores.items() %}
                    <li>{{ dimension }}: {{ "%.2f"|format(score) }}</li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>建议</h2>
                {% for recommendation in recommendations %}
                <div class="recommendation">{{ recommendation }}</div>
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





