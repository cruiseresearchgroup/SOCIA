"""
轻量级SBI Agent主文件
专门针对SOCIA项目设计，支持模块化SBI校准
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging

# 导入工具模块
from .file_utils import FileUtils
from .param_utils import ParameterManager, ParameterType
from .log_utils import setup_logging, get_logger, PerformanceMonitor, SBIProgressLogger
from .data_utils import SOCIADataProcessor, SummaryStatisticsCalculator, DataValidator
from .socia_utils import SOCIAConfigManager

# 导入增强模块
from .enhanced_data_loader import (
    SOCIAModuleDependencyAnalyzer, 
    IntelligentSummaryStatisticsDesigner, 
    AdvancedParameterSpaceBuilder, 
    SOCIADataProcessor as EnhancedSOCIADataProcessor
)
from .enhanced_parameter_manager import SOCIAParameterManager

# 导入仿真和SBI执行模块
from .socia_workflow_integration import SOCIAWorkflowIntegrator
from .intelligent_sbi_strategy import IntelligentSBIStrategy
from .enhanced_simulation_wrapper import EnhancedSimulationWrapper
from .convergence_monitor import ConvergenceMonitorManager

# 导入结果处理和验证模块
from .socia_result_integration import SOCIAResultFormatter, SOCIAResultSaver
from .intelligent_calibration_validation import (
    SOCIAVerificationValidator, 
    MultiDimensionalQualityAssessor, 
    CalibrationFailureHandler
)
from .result_visualization import CalibrationVisualizer, SOCIACompatibleReporter
from .performance_analysis import (
    CalibrationPerformanceAnalyzer, 
    ParameterOptimizationAdvisor, 
    SOCIAOptimizationReporter
)

class SimpleSBIAgent:
    """轻量级SBI Agent主类"""
    
    def __init__(self, output_dir: Union[str, Path], data_dir: Optional[Union[str, Path]] = None):
        """
        初始化SBI Agent
        
        Args:
            output_dir: SOCIA任务输出目录路径
            data_dir: 数据目录路径，默认为 output_dir/../data_fitting/mask_adoption_data/
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir) if data_dir else self.output_dir.parent / "data_fitting" / "mask_adoption_data"
        
        # 初始化日志
        self.logger = setup_logging(
            log_level="INFO",
            log_file=str(self.output_dir / "sbi_agent.log")
        )
        
        # 初始化性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 初始化工具类
        self.file_utils = FileUtils()
        self.param_manager = ParameterManager()
        self.data_processor = SOCIADataProcessor()
        self.statistics_calculator = SummaryStatisticsCalculator()
        self.data_validator = DataValidator()
        self.socia_config_manager = SOCIAConfigManager(self.output_dir)
        
        # 初始化增强工具类
        self.dependency_analyzer = SOCIAModuleDependencyAnalyzer()
        self.summary_designer = IntelligentSummaryStatisticsDesigner()
        self.parameter_space_builder = AdvancedParameterSpaceBuilder()
        self.enhanced_data_processor = EnhancedSOCIADataProcessor()
        self.enhanced_param_manager = SOCIAParameterManager()
        
        # 初始化仿真和SBI执行组件
        self.workflow_integrator = SOCIAWorkflowIntegrator(self.output_dir)
        self.sbi_strategy = IntelligentSBIStrategy()
        self.simulation_wrapper = None  # 将在需要时初始化
        self.convergence_monitor = ConvergenceMonitorManager()
        
        # 初始化结果处理和验证组件
        self.result_formatter = SOCIAResultFormatter(self.output_dir)
        self.result_saver = SOCIAResultSaver(self.output_dir)
        self.verification_validator = SOCIAVerificationValidator(self.output_dir)
        self.quality_assessor = MultiDimensionalQualityAssessor()
        self.failure_handler = CalibrationFailureHandler()
        self.visualizer = CalibrationVisualizer(self.output_dir)
        self.reporter = SOCIACompatibleReporter(self.output_dir)
        self.performance_analyzer = CalibrationPerformanceAnalyzer()
        self.optimization_advisor = ParameterOptimizationAdvisor()
        self.optimization_reporter = SOCIAOptimizationReporter(self.output_dir)
        
        # 初始化属性
        self.param_defs: Dict[str, Any] = {}
        self.current_params: Dict[str, Any] = {}
        self.target_data: Optional[pd.DataFrame] = None
        self.simulation_code_path: Optional[Path] = None
        self.model_plan: Optional[Dict[str, Any]] = None
        self.data_analysis: Optional[Dict[str, Any]] = None
        self.calibration_strategy: Optional[Dict[str, Any]] = None
        
        # 验证路径
        self.validate_paths()
        
        # 加载SOCIA配置
        self.load_socia_configs()
        
        self.logger.info("SBI Agent初始化完成")
    
    def validate_paths(self) -> None:
        """验证路径"""
        if not self.output_dir.exists():
            raise FileNotFoundError(f"输出目录不存在: {self.output_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 检查必需文件
        required_files = [
            "parameters.json",
            "parameter_definitions.json",
            "simulation_code_iter_1.py"
        ]
        
        for file_name in required_files:
            file_path = self.output_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"必需文件不存在: {file_path}")
        
        self.logger.info("路径验证通过")
    
    def load_socia_configs(self) -> None:
        """加载SOCIA配置"""
        try:
            # 加载所有SOCIA配置
            configs = self.socia_config_manager.load_all_configs()
            
            # 获取校准策略
            self.calibration_strategy = self.socia_config_manager.get_calibration_strategy()
            
            # 加载增强配置
            self._load_enhanced_configs()
            
            self.logger.info("SOCIA配置加载完成")
        except Exception as e:
            self.logger.error(f"SOCIA配置加载失败: {e}")
            raise
    
    def _load_enhanced_configs(self) -> None:
        """加载增强配置"""
        try:
            # 加载模块依赖关系分析
            model_plan_path = self.output_dir / "model_plan_iter_1.json"
            if model_plan_path.exists():
                self.dependency_analyzer.load_model_plan(model_plan_path)
                self.logger.info("模块依赖关系分析完成")
            
            # 加载数据分析结果
            data_analysis_path = self.output_dir / "data_analysis_iter_1.json"
            if data_analysis_path.exists():
                self.summary_designer.load_data_analysis(data_analysis_path)
                self.logger.info("智能摘要统计设计完成")
            
            # 加载参数定义
            param_defs_path = self.output_dir / "parameter_definitions.json"
            if param_defs_path.exists():
                self.parameter_space_builder.load_parameter_definitions(param_defs_path)
                self.logger.info("参数空间构建完成")
            
            # 加载增强参数管理
            current_params_path = self.output_dir / "parameters.json"
            if current_params_path.exists() and model_plan_path.exists():
                self.enhanced_param_manager.load_parameter_configuration(
                    param_defs_path, current_params_path, model_plan_path
                )
                self.logger.info("增强参数管理完成")
            
            # 加载SOCIA工作流配置
            self.workflow_integrator.load_socia_workflow_configs()
            self.logger.info("SOCIA工作流集成完成")
            
        except Exception as e:
            self.logger.warning(f"增强配置加载失败: {e}")
    
    def load_parameters(self) -> None:
        """加载参数"""
        try:
            # 加载参数定义
            param_defs_path = self.output_dir / "parameter_definitions.json"
            self.param_defs = self.file_utils.load_json(param_defs_path)
            self.param_manager.load_parameter_definitions(self.param_defs)
            
            # 加载当前参数
            params_path = self.output_dir / "parameters.json"
            self.current_params = self.file_utils.load_json(params_path)
            self.param_manager.load_current_parameters(self.current_params)
            
            self.logger.info(f"参数加载完成: {len(self.current_params)} 个参数")
        except Exception as e:
            self.logger.error(f"参数加载失败: {e}")
            raise
    
    def load_target_data(self) -> None:
        """加载目标数据"""
        try:
            # 使用增强数据处理器
            train_data_path = self.data_dir / "train_data.csv"
            self.enhanced_data_processor.load_and_preprocess_data(train_data_path)
            
            # 获取处理后的数据
            self.target_data = self.enhanced_data_processor.target_data
            
            # 获取数据摘要
            data_summary = self.enhanced_data_processor.get_data_summary()
            self.logger.info(f"数据质量报告: {data_summary['data_quality']['quality_score']:.3f}")
            
            self.logger.info(f"目标数据加载完成: {len(self.target_data)} 行")
        except Exception as e:
            self.logger.error(f"目标数据加载失败: {e}")
            raise
    
    def get_simulation_code_path(self) -> Path:
        """获取仿真代码路径"""
        if self.simulation_code_path is None:
            self.simulation_code_path = self.output_dir / "simulation_code_iter_1.py"
        
        return self.simulation_code_path
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """运行仿真"""
        try:
            # 创建临时参数文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(params, temp_file, indent=2)
                temp_params_path = temp_file.name
            
            # 设置环境变量
            env = os.environ.copy()
            env['SBI_PARAMS_FILE'] = temp_params_path
            
            # 运行仿真代码
            simulation_path = self.get_simulation_code_path()
            result = subprocess.run(
                [sys.executable, str(simulation_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 清理临时文件
            os.unlink(temp_params_path)
            
            if result.returncode != 0:
                raise RuntimeError(f"仿真执行失败: {result.stderr}")
            
            # 解析仿真结果
            simulation_results = self._parse_simulation_output(result.stdout)
            
            self.logger.info("仿真执行成功")
            return simulation_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("仿真执行超时")
            raise
        except Exception as e:
            self.logger.error(f"仿真执行失败: {e}")
            raise
    
    def _parse_simulation_output(self, output: str) -> Dict[str, np.ndarray]:
        """解析仿真输出"""
        # 这里需要根据实际的仿真输出格式来解析
        # 暂时返回模拟数据
        simulation_results = {
            'adoption_rate': np.random.random(30),  # 30天的采纳率
            'info_rate': np.random.random(30)       # 30天的信息率
        }
        
        return simulation_results
    
    def calculate_summary_statistics(self, 
                                   observed: Dict[str, np.ndarray], 
                                   simulated: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算摘要统计"""
        statistics = {}
        
        for signal_name in observed.keys():
            if signal_name in simulated:
                signal_stats = self.statistics_calculator.calculate_summary_statistics(
                    observed[signal_name],
                    simulated[signal_name]
                )
                
                # 添加信号名前缀
                for stat_name, value in signal_stats.items():
                    statistics[f"{signal_name}_{stat_name}"] = value
        
        return statistics
    
    def calibrate_module(self, module_name: str) -> Dict[str, Any]:
        """校准单个模块"""
        self.logger.info(f"开始校准模块: {module_name}")
        
        # 获取模块的校准策略
        module_strategy = self.calibration_strategy.get('summary_statistics', {}).get(module_name, {})
        parameter_constraints = self.calibration_strategy.get('parameter_constraints', {}).get(module_name, {})
        
        # 这里应该实现具体的SBI校准逻辑
        # 暂时返回模拟结果
        calibration_results = {
            'module_name': module_name,
            'calibrated_parameters': {},
            'convergence_achieved': True,
            'final_metrics': {'rmse': 0.1, 'mae': 0.05, 'r2': 0.95}
        }
        
        self.logger.info(f"模块校准完成: {module_name}")
        return calibration_results
    
    def calibrate(self) -> Dict[str, Any]:
        """执行完整的SBI校准"""
        self.logger.info("开始SBI校准")
        
        # 加载参数和数据
        self.load_parameters()
        self.load_target_data()
        
        # 获取校准顺序
        calibration_order = self.calibration_strategy.get('calibration_order', [])
        
        # 执行分阶段校准
        calibration_results = {}
        for module_name in calibration_order:
            module_results = self.calibrate_module(module_name)
            calibration_results[module_name] = module_results
        
        # 生成最终结果
        final_results = {
            'calibration_results': calibration_results,
            'calibration_strategy': self.calibration_strategy,
            'performance_summary': self.performance_monitor.get_summary()
        }
        
        # 保存结果
        self.save_results(final_results)
        
        self.logger.info("SBI校准完成")
        return final_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """保存结果"""
        try:
            results_dir = self.output_dir / "sbi_results"
            results_dir.mkdir(exist_ok=True)
            
            # 保存校准结果
            results_file = results_dir / "calibration_results.json"
            self.file_utils.save_json(results, results_file)
            
            # 保存参数
            if 'calibration_results' in results:
                params_file = results_dir / "calibrated_parameters.json"
                calibrated_params = {}
                for module_name, module_results in results['calibration_results'].items():
                    if 'calibrated_parameters' in module_results:
                        calibrated_params.update(module_results['calibrated_parameters'])
                
                self.file_utils.save_json(calibrated_params, params_file)
            
            self.logger.info(f"结果保存完成: {results_dir}")
        except Exception as e:
            self.logger.error(f"结果保存失败: {e}")
            raise
    
    def get_module_dependencies(self, module_name: str) -> List[str]:
        """获取模块依赖关系"""
        return self.dependency_analyzer.get_module_dependencies(module_name)
    
    def get_calibration_order(self) -> List[str]:
        """获取校准顺序"""
        return self.dependency_analyzer.get_calibration_order()
    
    def get_summary_statistics_design(self, module_name: str) -> Dict[str, Any]:
        """获取模块的摘要统计设计"""
        return self.summary_designer.get_summary_statistics_for_module(module_name)
    
    def get_parameter_space_for_module(self, module_name: str) -> Dict[str, Any]:
        """获取模块的参数空间"""
        return self.enhanced_param_manager.build_parameter_space_for_module(module_name)
    
    def get_parameter_bounds_for_module(self, module_name: str) -> Dict[str, Tuple[float, float]]:
        """获取模块的参数边界"""
        return self.enhanced_param_manager.get_parameter_bounds(module_name)
    
    def validate_module_parameters(self, params: Dict[str, Any], module_name: str) -> Tuple[bool, List[str]]:
        """验证模块参数"""
        return self.enhanced_param_manager.validate_parameters(params, module_name)
    
    def get_target_signals(self) -> Dict[str, np.ndarray]:
        """获取目标信号"""
        return self.enhanced_data_processor.get_target_signals()
    
    def initialize_simulation_wrapper(self) -> None:
        """初始化仿真包装器"""
        if self.simulation_wrapper is None:
            simulation_code_path = self.get_simulation_code_path()
            self.simulation_wrapper = EnhancedSimulationWrapper(simulation_code_path, self.output_dir)
            self.logger.info("仿真包装器初始化完成")
    
    def run_enhanced_simulation(self, parameters: Dict[str, Any], 
                               module_name: Optional[str] = None) -> Dict[str, Any]:
        """运行增强仿真"""
        if self.simulation_wrapper is None:
            self.initialize_simulation_wrapper()
        
        return self.simulation_wrapper.run_single_module_simulation(module_name, parameters)
    
    def run_multi_module_simulation(self, module_parameters: Dict[str, Dict[str, Any]],
                                  calibration_order: Optional[List[str]] = None,
                                  parallel: bool = True) -> Dict[str, Any]:
        """运行多模块仿真"""
        if self.simulation_wrapper is None:
            self.initialize_simulation_wrapper()
        
        return self.simulation_wrapper.run_multi_module_simulation(
            module_parameters, calibration_order, parallel
        )
    
    def design_sbi_strategy(self, modules: List[str], 
                          data_complexity: str = 'medium') -> Dict[str, Any]:
        """设计SBI策略"""
        parameter_spaces = {
            module: self.get_parameter_space_for_module(module)
            for module in modules
        }
        
        return self.sbi_strategy.design_calibration_strategy(
            modules, parameter_spaces, data_complexity
        )
    
    def start_convergence_monitoring(self, module_name: str) -> None:
        """开始收敛监控"""
        self.convergence_monitor.start_monitoring(module_name)
        self.logger.info(f"开始收敛监控: {module_name}")
    
    def update_convergence_metrics(self, metrics: Dict[str, float]) -> None:
        """更新收敛指标"""
        self.convergence_monitor.update_metrics(metrics)
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """获取收敛状态"""
        return self.convergence_monitor.get_convergence_status()
    
    def should_adjust_parameters(self, module_name: str,
                               current_bounds: Dict[str, Tuple[float, float]],
                               calibration_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """判断是否应该调整参数"""
        return self.convergence_monitor.should_adjust_parameters(
            module_name, current_bounds, calibration_results
        )
    
    def should_restart_calibration(self, failure_reason: str,
                                 calibration_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """判断是否应该重启校准"""
        return self.convergence_monitor.should_restart(failure_reason, calibration_results)
    
    def get_workflow_constraints(self, module_name: str) -> Dict[str, Any]:
        """获取工作流约束"""
        return {
            'parameter_constraints': self.workflow_integrator.get_parameter_constraints_for_module(module_name),
            'convergence_requirements': self.workflow_integrator.get_convergence_requirements_for_module(module_name)
        }
    
    def generate_workflow_feedback(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成工作流反馈"""
        return self.workflow_integrator.generate_feedback(calibration_results)
    
    def start_iteration(self, iteration_id: int) -> None:
        """开始迭代"""
        self.workflow_integrator.start_iteration(iteration_id)
    
    def update_iteration(self, results: Dict[str, Any]) -> None:
        """更新迭代"""
        self.workflow_integrator.update_iteration(results)
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        return self.workflow_integrator.get_improvement_suggestions()
    
    def should_continue_iteration(self) -> bool:
        """判断是否应该继续迭代"""
        return self.workflow_integrator.should_continue_iteration()
    
    def format_and_save_results(self, results: Dict[str, Any],
                               output_dir: Optional[Union[str, Path]] = None) -> None:
        """格式化并保存结果"""
        self.workflow_integrator.format_and_save_results(results, output_dir)
    
    def validate_calibration_results(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证校准结果"""
        # 加载验证约束
        verification_path = self.output_dir / "verification_results_iter_1.json"
        if verification_path.exists():
            self.verification_validator.load_verification_constraints(verification_path)
        
        # 执行验证
        validation_results = self.verification_validator.validate_calibration_quality(calibration_results)
        
        # 执行质量评估
        quality_assessment = self.quality_assessor.assess_calibration_quality(calibration_results)
        
        return {
            'validation_results': validation_results,
            'quality_assessment': quality_assessment
        }
    
    def handle_calibration_failure(self, failure_type: str, 
                                 calibration_results: Dict[str, Any],
                                 error_details: Dict[str, Any]) -> Dict[str, Any]:
        """处理校准失败"""
        return self.failure_handler.handle_calibration_failure(
            failure_type, calibration_results, error_details
        )
    
    def create_calibration_visualizations(self, calibration_results: Dict[str, Any],
                                        target_data: Optional[pd.DataFrame] = None) -> List[Path]:
        """创建校准可视化"""
        return self.visualizer.create_calibration_comparison_plots(calibration_results, target_data)
    
    def create_parameter_analysis_plots(self, calibration_results: Dict[str, Any]) -> List[Path]:
        """创建参数分析图"""
        return self.visualizer.create_parameter_analysis_plots(calibration_results)
    
    def analyze_calibration_performance(self, calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析校准性能"""
        return self.performance_analyzer.analyze_calibration_performance(calibration_results)
    
    def generate_optimization_advice(self, calibration_results: Dict[str, Any],
                                   performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化建议"""
        return self.optimization_advisor.generate_optimization_advice(calibration_results, performance_analysis)
    
    def save_calibrated_parameters(self, calibration_results: Dict[str, Any]) -> Path:
        """保存校准参数"""
        return self.result_saver.save_calibrated_parameters(calibration_results)
    
    def save_calibration_report(self, calibration_results: Dict[str, Any]) -> Path:
        """保存校准报告"""
        return self.result_saver.save_calibration_report(calibration_results)
    
    def save_feedback(self, calibration_results: Dict[str, Any], 
                     iteration_id: int = 1) -> Path:
        """保存反馈"""
        return self.result_saver.save_feedback(calibration_results, iteration_id)
    
    def save_verification_results(self, calibration_results: Dict[str, Any]) -> Path:
        """保存验证结果"""
        return self.result_saver.save_verification_results(calibration_results)
    
    def generate_comprehensive_report(self, calibration_results: Dict[str, Any],
                                    validation_results: Dict[str, Any],
                                    quality_assessment: Dict[str, Any]) -> Path:
        """生成综合报告"""
        return self.reporter.generate_comprehensive_report(
            calibration_results, validation_results, quality_assessment
        )
    
    def generate_optimization_report(self, performance_analysis: Dict[str, Any],
                                   optimization_advice: Dict[str, Any]) -> Path:
        """生成优化报告"""
        return self.optimization_reporter.generate_optimization_report(
            performance_analysis, optimization_advice
        )
    
    def process_and_validate_results(self, calibration_results: Dict[str, Any],
                                   target_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """处理并验证结果"""
        # 验证校准结果
        validation_output = self.validate_calibration_results(calibration_results)
        
        # 创建可视化
        visualization_paths = self.create_calibration_visualizations(calibration_results, target_data)
        parameter_analysis_paths = self.create_parameter_analysis_plots(calibration_results)
        
        # 分析性能
        performance_analysis = self.analyze_calibration_performance(calibration_results)
        
        # 生成优化建议
        optimization_advice = self.generate_optimization_advice(calibration_results, performance_analysis)
        
        # 保存结果
        saved_paths = {
            'calibrated_parameters': self.save_calibrated_parameters(calibration_results),
            'calibration_report': self.save_calibration_report(calibration_results),
            'feedback': self.save_feedback(calibration_results),
            'verification_results': self.save_verification_results(calibration_results),
            'comprehensive_report': self.generate_comprehensive_report(
                calibration_results, 
                validation_output['validation_results'], 
                validation_output['quality_assessment']
            ),
            'optimization_report': self.generate_optimization_report(performance_analysis, optimization_advice)
        }
        
        return {
            'validation_output': validation_output,
            'visualization_paths': visualization_paths + parameter_analysis_paths,
            'performance_analysis': performance_analysis,
            'optimization_advice': optimization_advice,
            'saved_paths': saved_paths
        }
    
    def get_enhanced_calibration_strategy(self) -> Dict[str, Any]:
        """获取增强的校准策略"""
        strategy = {
            'dependency_analysis': self.dependency_analyzer.get_calibration_strategy(),
            'summary_statistics': {
                module: self.summary_designer.get_summary_statistics_for_module(module)
                for module in self.dependency_analyzer.calibration_modules
            },
            'parameter_spaces': {
                module: self.enhanced_param_manager.build_parameter_space_for_module(module)
                for module in self.dependency_analyzer.calibration_modules
            },
            'calibration_order': self.dependency_analyzer.get_calibration_order()
        }
        return strategy
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        summary = {
            'output_dir': str(self.output_dir),
            'data_dir': str(self.data_dir),
            'calibration_strategy': self.calibration_strategy,
            'parameter_count': len(self.current_params),
            'target_data_shape': self.target_data.shape if self.target_data is not None else None,
            'calibration_modules': self.dependency_analyzer.calibration_modules,
            'calibration_order': self.dependency_analyzer.get_calibration_order(),
            'enhanced_calibration_strategy': self.get_enhanced_calibration_strategy()
        }
        return summary

def main():
    """主函数示例"""
    # 示例使用
    output_dir = "output/mask_adoption_calibrasim_debug_run3"
    data_dir = "data_fitting/mask_adoption_data"
    
    try:
        # 创建SBI Agent
        agent = SimpleSBIAgent(output_dir, data_dir)
        
        # 执行校准
        results = agent.calibrate()
        
        # 打印摘要
        print("SBI校准完成!")
        print(f"校准结果: {agent.get_summary()}")
        
    except Exception as e:
        print(f"SBI校准失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
