"""
收敛监控和自动调整模块
实现多级收敛监控、自动参数空间调整、收敛失败处理、早停和重启机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime, timedelta
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)

class ConvergenceMonitor:
    """收敛监控器"""
    
    def __init__(self, window_size: int = 10, tolerance: float = 1e-4):
        self.window_size = window_size
        self.tolerance = tolerance
        self.metric_history = deque(maxlen=window_size)
        self.convergence_status = {}
        self.convergence_thresholds = {
            'rmse': 0.01,
            'mae': 0.005,
            'r2': 0.001
        }
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """更新指标"""
        self.metric_history.append(metrics.copy())
        
        # 检查收敛状态
        for metric_name, threshold in self.convergence_thresholds.items():
            if metric_name in metrics:
                self._check_convergence(metric_name, threshold)
    
    def _check_convergence(self, metric_name: str, threshold: float) -> None:
        """检查收敛状态"""
        if len(self.metric_history) < self.window_size:
            return
        
        # 计算指标变化
        recent_values = [m[metric_name] for m in list(self.metric_history)[-self.window_size:]]
        if len(recent_values) < 2:
            return
        
        # 计算变化率
        change_rate = abs(recent_values[-1] - recent_values[0]) / abs(recent_values[0]) if recent_values[0] != 0 else 0
        
        # 检查是否收敛
        is_converged = change_rate < threshold
        
        self.convergence_status[metric_name] = {
            'converged': is_converged,
            'change_rate': change_rate,
            'threshold': threshold,
            'recent_values': recent_values
        }
    
    def is_converged(self, metric_name: Optional[str] = None) -> bool:
        """检查是否收敛"""
        if metric_name:
            return self.convergence_status.get(metric_name, {}).get('converged', False)
        else:
            return all(
                status.get('converged', False) 
                for status in self.convergence_status.values()
            )
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """获取收敛摘要"""
        return {
            'convergence_status': self.convergence_status,
            'overall_converged': self.is_converged(),
            'metric_count': len(self.metric_history),
            'window_size': self.window_size
        }
    
    def reset(self) -> None:
        """重置监控器"""
        self.metric_history.clear()
        self.convergence_status.clear()

class AdaptiveParameterSpaceAdjuster:
    """自适应参数空间调整器"""
    
    def __init__(self):
        self.adjustment_history = []
        self.parameter_sensitivity = {}
        self.adjustment_strategies = {
            'expand': self._expand_parameter_space,
            'contract': self._contract_parameter_space,
            'shift': self._shift_parameter_space,
            'refine': self._refine_parameter_space
        }
    
    def adjust_parameter_space(self, module_name: str,
                             current_bounds: Dict[str, Tuple[float, float]],
                             calibration_results: Dict[str, Any],
                             convergence_status: Dict[str, Any]) -> Dict[str, Any]:
        """调整参数空间"""
        adjustment = {
            'module_name': module_name,
            'timestamp': time.time(),
            'adjustments': {},
            'strategy': 'none',
            'reason': ''
        }
        
        # 基于收敛状态选择调整策略
        if not convergence_status.get('overall_converged', False):
            if self._should_expand_space(calibration_results):
                adjustment = self._expand_parameter_space(module_name, current_bounds, calibration_results)
            elif self._should_contract_space(calibration_results):
                adjustment = self._contract_parameter_space(module_name, current_bounds, calibration_results)
        else:
            if self._should_refine_space(calibration_results):
                adjustment = self._refine_parameter_space(module_name, current_bounds, calibration_results)
        
        self.adjustment_history.append(adjustment)
        return adjustment
    
    def _should_expand_space(self, calibration_results: Dict[str, Any]) -> bool:
        """判断是否应该扩展参数空间"""
        # 检查参数是否集中在边界
        parameter_values = calibration_results.get('parameter_values', {})
        boundary_count = sum(
            1 for value in parameter_values.values()
            if isinstance(value, (int, float)) and (value <= 0.1 or value >= 0.9)
        )
        
        return boundary_count > len(parameter_values) * 0.5
    
    def _should_contract_space(self, calibration_results: Dict[str, Any]) -> bool:
        """判断是否应该收缩参数空间"""
        # 检查参数是否分散
        parameter_values = calibration_results.get('parameter_values', {})
        if not parameter_values:
            return False
        
        values = [v for v in parameter_values.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            return False
        
        variance = np.var(values)
        return variance > 0.1  # 高方差表示参数分散
    
    def _should_refine_space(self, calibration_results: Dict[str, Any]) -> bool:
        """判断是否应该细化参数空间"""
        final_metrics = calibration_results.get('final_metrics', {})
        rmse = final_metrics.get('rmse', 1.0)
        return rmse > 0.1  # RMSE仍然较高
    
    def _expand_parameter_space(self, module_name: str,
                               current_bounds: Dict[str, Tuple[float, float]],
                               calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """扩展参数空间"""
        adjustments = {}
        
        for param_name, bounds in current_bounds.items():
            # 扩展边界到原来的1.5倍
            center = (bounds[0] + bounds[1]) / 2
            width = bounds[1] - bounds[0]
            new_width = width * 1.5
            
            new_bounds = (
                max(0, center - new_width / 2),
                min(1, center + new_width / 2)
            )
            
            adjustments[param_name] = {
                'old_bounds': bounds,
                'new_bounds': new_bounds,
                'action': 'expand'
            }
        
        return {
            'module_name': module_name,
            'timestamp': time.time(),
            'adjustments': adjustments,
            'strategy': 'expand',
            'reason': 'parameter_boundary_concentration'
        }
    
    def _contract_parameter_space(self, module_name: str,
                                current_bounds: Dict[str, Tuple[float, float]],
                                calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """收缩参数空间"""
        adjustments = {}
        
        for param_name, bounds in current_bounds.items():
            # 收缩边界到原来的0.8倍
            center = (bounds[0] + bounds[1]) / 2
            width = bounds[1] - bounds[0]
            new_width = width * 0.8
            
            new_bounds = (
                max(0, center - new_width / 2),
                min(1, center + new_width / 2)
            )
            
            adjustments[param_name] = {
                'old_bounds': bounds,
                'new_bounds': new_bounds,
                'action': 'contract'
            }
        
        return {
            'module_name': module_name,
            'timestamp': time.time(),
            'adjustments': adjustments,
            'strategy': 'contract',
            'reason': 'parameter_dispersion'
        }
    
    def _shift_parameter_space(self, module_name: str,
                             current_bounds: Dict[str, Tuple[float, float]],
                             calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """平移参数空间"""
        adjustments = {}
        
        # 基于参数值分布平移空间
        parameter_values = calibration_results.get('parameter_values', {})
        
        for param_name, bounds in current_bounds.items():
            if param_name in parameter_values:
                value = parameter_values[param_name]
                if isinstance(value, (int, float)):
                    # 将参数空间中心移到当前值附近
                    width = bounds[1] - bounds[0]
                    new_center = value
                    
                    new_bounds = (
                        max(0, new_center - width / 2),
                        min(1, new_center + width / 2)
                    )
                    
                    adjustments[param_name] = {
                        'old_bounds': bounds,
                        'new_bounds': new_bounds,
                        'action': 'shift'
                    }
        
        return {
            'module_name': module_name,
            'timestamp': time.time(),
            'adjustments': adjustments,
            'strategy': 'shift',
            'reason': 'parameter_value_centering'
        }
    
    def _refine_parameter_space(self, module_name: str,
                              current_bounds: Dict[str, Tuple[float, float]],
                              calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """细化参数空间"""
        adjustments = {}
        
        for param_name, bounds in current_bounds.items():
            # 在最佳值附近细化空间
            parameter_values = calibration_results.get('parameter_values', {})
            if param_name in parameter_values:
                best_value = parameter_values[param_name]
                if isinstance(best_value, (int, float)):
                    # 在最佳值附近创建更小的搜索空间
                    refinement_factor = 0.5
                    width = bounds[1] - bounds[0]
                    new_width = width * refinement_factor
                    
                    new_bounds = (
                        max(0, best_value - new_width / 2),
                        min(1, best_value + new_width / 2)
                    )
                    
                    adjustments[param_name] = {
                        'old_bounds': bounds,
                        'new_bounds': new_bounds,
                        'action': 'refine'
                    }
        
        return {
            'module_name': module_name,
            'timestamp': time.time(),
            'adjustments': adjustments,
            'strategy': 'refine',
            'reason': 'convergence_refinement'
        }

class EarlyStoppingManager:
    """早停管理器"""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metrics = {}
        self.wait_count = 0
        self.early_stop_triggered = False
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """检查早停条件"""
        if self.early_stop_triggered:
            return True
        
        # 检查主要指标
        primary_metric = 'rmse'  # 主要指标
        if primary_metric not in metrics:
            return False
        
        current_value = metrics[primary_metric]
        
        if primary_metric not in self.best_metrics:
            self.best_metrics[primary_metric] = current_value
            self.wait_count = 0
            return False
        
        best_value = self.best_metrics[primary_metric]
        
        # 检查是否有改善
        if current_value < best_value - self.min_delta:
            self.best_metrics[primary_metric] = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # 检查是否达到耐心阈值
        if self.wait_count >= self.patience:
            self.early_stop_triggered = True
            logger.info(f"早停触发: 等待 {self.wait_count} 次迭代无改善")
            return True
        
        return False
    
    def reset(self) -> None:
        """重置早停管理器"""
        self.best_metrics.clear()
        self.wait_count = 0
        self.early_stop_triggered = False

class RestartManager:
    """重启管理器"""
    
    def __init__(self, max_restarts: int = 3, restart_delay: float = 1.0):
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.restart_history = []
    
    def should_restart(self, failure_reason: str, 
                      calibration_results: Dict[str, Any]) -> bool:
        """判断是否应该重启"""
        if self.restart_count >= self.max_restarts:
            return False
        
        # 记录重启原因
        restart_record = {
            'restart_count': self.restart_count,
            'failure_reason': failure_reason,
            'timestamp': time.time(),
            'calibration_results': calibration_results
        }
        self.restart_history.append(restart_record)
        
        # 判断重启条件
        restart_conditions = [
            'convergence_failure',
            'parameter_cancellation',
            'simulation_timeout',
            'memory_error'
        ]
        
        should_restart = any(condition in failure_reason.lower() for condition in restart_conditions)
        
        if should_restart:
            self.restart_count += 1
            logger.info(f"重启决策: {failure_reason} (第 {self.restart_count} 次重启)")
        
        return should_restart
    
    def get_restart_strategy(self, failure_reason: str) -> Dict[str, Any]:
        """获取重启策略"""
        strategy = {
            'restart_count': self.restart_count,
            'delay': self.restart_delay,
            'parameter_adjustments': {},
            'method_changes': {}
        }
        
        if 'convergence_failure' in failure_reason.lower():
            strategy['parameter_adjustments'] = {
                'expand_bounds': True,
                'increase_iterations': True
            }
        elif 'parameter_cancellation' in failure_reason.lower():
            strategy['parameter_adjustments'] = {
                'add_constraints': True,
                'reduce_correlations': True
            }
        elif 'simulation_timeout' in failure_reason.lower():
            strategy['method_changes'] = {
                'reduce_complexity': True,
                'use_simpler_sbi': True
            }
        
        return strategy
    
    def reset(self) -> None:
        """重置重启管理器"""
        self.restart_count = 0
        self.restart_history.clear()

class ConvergenceMonitorManager:
    """收敛监控管理器"""
    
    def __init__(self):
        self.convergence_monitor = ConvergenceMonitor()
        self.parameter_adjuster = AdaptiveParameterSpaceAdjuster()
        self.early_stopping = EarlyStoppingManager()
        self.restart_manager = RestartManager()
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self, module_name: str) -> None:
        """开始监控"""
        self.convergence_monitor.reset()
        self.early_stopping.reset()
        self.restart_manager.reset()
        self.monitoring_active = True
        
        logger.info(f"开始收敛监控: {module_name}")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring_active = False
        logger.info("停止收敛监控")
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """更新指标"""
        self.convergence_monitor.update_metrics(metrics)
        
        # 检查早停
        if self.early_stopping.check_early_stopping(metrics):
            self.stop_monitoring()
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """获取收敛状态"""
        return {
            'convergence_summary': self.convergence_monitor.get_convergence_summary(),
            'early_stopping': {
                'triggered': self.early_stopping.early_stop_triggered,
                'wait_count': self.early_stopping.wait_count,
                'best_metrics': self.early_stopping.best_metrics
            },
            'restart_info': {
                'restart_count': self.restart_manager.restart_count,
                'max_restarts': self.restart_manager.max_restarts
            }
        }
    
    def should_adjust_parameters(self, module_name: str,
                               current_bounds: Dict[str, Tuple[float, float]],
                               calibration_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """判断是否应该调整参数"""
        convergence_status = self.convergence_monitor.get_convergence_summary()
        
        if not convergence_status['overall_converged']:
            adjustment = self.parameter_adjuster.adjust_parameter_space(
                module_name, current_bounds, calibration_results, convergence_status
            )
            return True, adjustment
        
        return False, {}
    
    def should_restart(self, failure_reason: str, 
                      calibration_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """判断是否应该重启"""
        if self.restart_manager.should_restart(failure_reason, calibration_results):
            strategy = self.restart_manager.get_restart_strategy(failure_reason)
            return True, strategy
        
        return False, {}
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            'convergence_status': self.get_convergence_status(),
            'parameter_adjustments': len(self.parameter_adjuster.adjustment_history),
            'restart_history': len(self.restart_manager.restart_history),
            'monitoring_active': self.monitoring_active
        }





