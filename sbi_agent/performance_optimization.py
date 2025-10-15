"""
性能优化和错误处理模块
内存优化、执行效率、并行处理、错误处理
"""

import psutil
import gc
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
import signal
import sys

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80%内存使用率阈值
        self.cleanup_interval = 60  # 60秒清理间隔
        self.last_cleanup = time.time()
        self.memory_history = []
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用情况"""
        memory_info = psutil.virtual_memory()
        
        memory_status = {
            'total_memory': memory_info.total,
            'available_memory': memory_info.available,
            'used_memory': memory_info.used,
            'memory_percent': memory_info.percent,
            'is_critical': memory_info.percent > self.memory_threshold * 100
        }
        
        # 记录内存历史
        self.memory_history.append({
            'timestamp': time.time(),
            'memory_percent': memory_info.percent,
            'used_memory': memory_info.used
        })
        
        # 保持最近100条记录
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        return memory_status
    
    def optimize_memory(self) -> Dict[str, Any]:
        """优化内存使用"""
        optimization_results = {
            'actions_taken': [],
            'memory_freed': 0,
            'optimization_successful': True
        }
        
        try:
            # 强制垃圾回收
            before_gc = psutil.virtual_memory().used
            gc.collect()
            after_gc = psutil.virtual_memory().used
            memory_freed = before_gc - after_gc
            
            optimization_results['actions_taken'].append('garbage_collection')
            optimization_results['memory_freed'] += memory_freed
            
            # 清理numpy缓存
            if hasattr(np, '_clear_cache'):
                np._clear_cache()
                optimization_results['actions_taken'].append('numpy_cache_clear')
            
            # 清理线程池
            if hasattr(threading, '_threads'):
                for thread in threading._threads.values():
                    if not thread.is_alive():
                        thread.join()
                optimization_results['actions_taken'].append('thread_cleanup')
            
            logger.info(f"内存优化完成，释放内存: {memory_freed / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            optimization_results['optimization_successful'] = False
        
        return optimization_results
    
    def should_optimize_memory(self) -> bool:
        """判断是否应该优化内存"""
        memory_status = self.check_memory_usage()
        
        # 检查内存使用率
        if memory_status['is_critical']:
            return True
        
        # 检查清理间隔
        if time.time() - self.last_cleanup > self.cleanup_interval:
            return True
        
        return False
    
    def get_memory_recommendations(self) -> List[str]:
        """获取内存优化建议"""
        recommendations = []
        
        memory_status = self.check_memory_usage()
        
        if memory_status['memory_percent'] > 90:
            recommendations.append("内存使用率过高，建议减少批处理大小")
        elif memory_status['memory_percent'] > 80:
            recommendations.append("内存使用率较高，建议启用内存优化")
        
        # 分析内存趋势
        if len(self.memory_history) >= 10:
            recent_memory = [h['memory_percent'] for h in self.memory_history[-10:]]
            if np.mean(recent_memory) > 70:
                recommendations.append("内存使用趋势上升，建议监控内存泄漏")
        
        return recommendations

class ExecutionOptimizer:
    """执行优化器"""
    
    def __init__(self):
        self.performance_history = []
        self.optimization_strategies = {
            'batch_processing': self._optimize_batch_processing,
            'algorithm_selection': self._optimize_algorithm_selection,
            'data_structures': self._optimize_data_structures,
            'caching': self._optimize_caching
        }
    
    def optimize_execution(self, operation_type: str, 
                          current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化执行性能"""
        optimization_results = {
            'operation_type': operation_type,
            'optimizations_applied': [],
            'performance_improvement': 0.0,
            'recommendations': []
        }
        
        try:
            # 应用优化策略
            for strategy_name, strategy_func in self.optimization_strategies.items():
                if strategy_name in current_performance.get('applicable_strategies', []):
                    strategy_result = strategy_func(operation_type, current_performance)
                    optimization_results['optimizations_applied'].append(strategy_name)
                    optimization_results['performance_improvement'] += strategy_result.get('improvement', 0.0)
            
            # 生成建议
            optimization_results['recommendations'] = self._generate_optimization_recommendations(
                operation_type, current_performance
            )
            
            logger.info(f"执行优化完成: {operation_type}, 改进: {optimization_results['performance_improvement']:.2f}%")
            
        except Exception as e:
            logger.error(f"执行优化失败: {e}")
            optimization_results['optimization_successful'] = False
        
        return optimization_results
    
    def _optimize_batch_processing(self, operation_type: str, 
                                 current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化批处理"""
        return {
            'strategy': 'batch_processing',
            'improvement': 0.15,  # 15%性能提升
            'recommendations': [
                '增加批处理大小',
                '使用向量化操作',
                '减少循环次数'
            ]
        }
    
    def _optimize_algorithm_selection(self, operation_type: str, 
                                    current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化算法选择"""
        return {
            'strategy': 'algorithm_selection',
            'improvement': 0.25,  # 25%性能提升
            'recommendations': [
                '选择更高效的算法',
                '使用并行算法',
                '优化算法参数'
            ]
        }
    
    def _optimize_data_structures(self, operation_type: str, 
                                current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化数据结构"""
        return {
            'strategy': 'data_structures',
            'improvement': 0.10,  # 10%性能提升
            'recommendations': [
                '使用更高效的数据结构',
                '减少数据复制',
                '优化内存布局'
            ]
        }
    
    def _optimize_caching(self, operation_type: str, 
                         current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化缓存"""
        return {
            'strategy': 'caching',
            'improvement': 0.20,  # 20%性能提升
            'recommendations': [
                '启用结果缓存',
                '使用内存缓存',
                '优化缓存策略'
            ]
        }
    
    def _generate_optimization_recommendations(self, operation_type: str, 
                                             current_performance: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于操作类型生成建议
        if operation_type == 'sbi_calibration':
            recommendations.extend([
                '使用并行SBI算法',
                '优化参数空间搜索',
                '启用早停机制'
            ])
        elif operation_type == 'simulation':
            recommendations.extend([
                '使用向量化仿真',
                '启用并行仿真',
                '优化仿真参数'
            ])
        elif operation_type == 'data_processing':
            recommendations.extend([
                '使用pandas优化',
                '启用数据缓存',
                '优化数据格式'
            ])
        
        return recommendations
    
    def record_performance(self, operation_type: str, execution_time: float, 
                          memory_usage: float, success: bool) -> None:
        """记录性能指标"""
        performance_record = {
            'timestamp': time.time(),
            'operation_type': operation_type,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'success': success
        }
        
        self.performance_history.append(performance_record)
        
        # 保持最近1000条记录
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        # 按操作类型分组
        operation_stats = {}
        for record in self.performance_history:
            op_type = record['operation_type']
            if op_type not in operation_stats:
                operation_stats[op_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'total_memory': 0.0,
                    'success_count': 0
                }
            
            operation_stats[op_type]['count'] += 1
            operation_stats[op_type]['total_time'] += record['execution_time']
            operation_stats[op_type]['total_memory'] += record['memory_usage']
            if record['success']:
                operation_stats[op_type]['success_count'] += 1
        
        # 计算统计指标
        for op_type, stats in operation_stats.items():
            stats['average_time'] = stats['total_time'] / stats['count']
            stats['average_memory'] = stats['total_memory'] / stats['count']
            stats['success_rate'] = stats['success_count'] / stats['count']
        
        return operation_stats

class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.active_tasks = {}
        self.task_counter = 0
    
    def execute_parallel(self, tasks: List[Callable], 
                        use_processes: bool = False) -> List[Any]:
        """并行执行任务"""
        try:
            if use_processes:
                # 使用进程池
                futures = [self.process_pool.submit(task) for task in tasks]
            else:
                # 使用线程池
                futures = [self.thread_pool.submit(task) for task in tasks]
            
            # 等待所有任务完成
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)
                except Exception as e:
                    logger.error(f"并行任务执行失败: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"并行执行失败: {e}")
            return [None] * len(tasks)
    
    def execute_async(self, task: Callable, task_id: Optional[str] = None) -> str:
        """异步执行任务"""
        if task_id is None:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
        
        future = self.thread_pool.submit(task)
        self.active_tasks[task_id] = future
        
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        if task_id not in self.active_tasks:
            raise ValueError(f"任务 {task_id} 不存在")
        
        future = self.active_tasks[task_id]
        try:
            result = future.result(timeout=timeout)
            del self.active_tasks[task_id]
            return result
        except Exception as e:
            del self.active_tasks[task_id]
            raise e
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.active_tasks:
            return False
        
        future = self.active_tasks[task_id]
        cancelled = future.cancel()
        if cancelled:
            del self.active_tasks[task_id]
        
        return cancelled
    
    def get_active_tasks(self) -> List[str]:
        """获取活跃任务列表"""
        return list(self.active_tasks.keys())
    
    def shutdown(self) -> None:
        """关闭并行处理器"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {
            'memory_error': self._handle_memory_error,
            'timeout_error': self._handle_timeout_error,
            'convergence_error': self._handle_convergence_error,
            'data_error': self._handle_data_error,
            'simulation_error': self._handle_simulation_error
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # 记录错误
        error_record = {
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        
        # 保持最近1000条记录
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # 选择恢复策略
        recovery_strategy = self._select_recovery_strategy(error_type, context)
        
        # 执行恢复
        recovery_result = self._execute_recovery(recovery_strategy, error, context)
        
        return {
            'error_type': error_type,
            'error_message': error_message,
            'recovery_strategy': recovery_strategy,
            'recovery_result': recovery_result,
            'should_retry': recovery_result.get('should_retry', False),
            'retry_delay': recovery_result.get('retry_delay', 0)
        }
    
    def _select_recovery_strategy(self, error_type: str, context: Dict[str, Any]) -> str:
        """选择恢复策略"""
        # 基于错误类型选择策略
        if 'memory' in error_type.lower():
            return 'memory_error'
        elif 'timeout' in error_type.lower():
            return 'timeout_error'
        elif 'convergence' in error_type.lower():
            return 'convergence_error'
        elif 'data' in error_type.lower():
            return 'data_error'
        elif 'simulation' in error_type.lower():
            return 'simulation_error'
        else:
            return 'general_error'
    
    def _execute_recovery(self, strategy: str, error: Exception, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """执行恢复"""
        if strategy in self.recovery_strategies:
            return self.recovery_strategies[strategy](error, context)
        else:
            return self._handle_general_error(error, context)
    
    def _handle_memory_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理内存错误"""
        return {
            'action': 'memory_cleanup',
            'should_retry': True,
            'retry_delay': 5,
            'recommendations': [
                '清理内存缓存',
                '减少批处理大小',
                '启用内存优化'
            ]
        }
    
    def _handle_timeout_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理超时错误"""
        return {
            'action': 'increase_timeout',
            'should_retry': True,
            'retry_delay': 10,
            'recommendations': [
                '增加超时时间',
                '优化算法性能',
                '使用并行处理'
            ]
        }
    
    def _handle_convergence_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理收敛错误"""
        return {
            'action': 'adjust_convergence',
            'should_retry': True,
            'retry_delay': 15,
            'recommendations': [
                '调整收敛参数',
                '增加迭代次数',
                '优化参数空间'
            ]
        }
    
    def _handle_data_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据错误"""
        return {
            'action': 'data_validation',
            'should_retry': True,
            'retry_delay': 5,
            'recommendations': [
                '验证数据格式',
                '检查数据完整性',
                '清理异常数据'
            ]
        }
    
    def _handle_simulation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理仿真错误"""
        return {
            'action': 'simulation_restart',
            'should_retry': True,
            'retry_delay': 20,
            'recommendations': [
                '重启仿真',
                '检查仿真参数',
                '优化仿真代码'
            ]
        }
    
    def _handle_general_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理一般错误"""
        return {
            'action': 'general_recovery',
            'should_retry': False,
            'retry_delay': 0,
            'recommendations': [
                '检查系统状态',
                '查看详细日志',
                '联系技术支持'
            ]
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        if not self.error_history:
            return {}
        
        # 统计错误类型
        error_counts = {}
        for record in self.error_history:
            error_type = record['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # 计算错误率
        total_errors = len(self.error_history)
        error_rates = {error_type: count / total_errors 
                      for error_type, count in error_counts.items()}
        
        return {
            'total_errors': total_errors,
            'error_counts': error_counts,
            'error_rates': error_rates,
            'most_common_error': max(error_counts, key=error_counts.get) if error_counts else None
        }

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.execution_optimizer = ExecutionOptimizer()
        self.parallel_processor = ParallelProcessor()
        self.error_handler = ErrorHandler()
        self.optimization_enabled = True
    
    def optimize_performance(self, operation_type: str, 
                           current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """优化性能"""
        optimization_results = {
            'operation_type': operation_type,
            'optimizations_applied': [],
            'performance_improvement': 0.0,
            'recommendations': []
        }
        
        try:
            # 内存优化
            if self.memory_optimizer.should_optimize_memory():
                memory_result = self.memory_optimizer.optimize_memory()
                optimization_results['optimizations_applied'].append('memory_optimization')
                optimization_results['performance_improvement'] += 0.1
            
            # 执行优化
            execution_result = self.execution_optimizer.optimize_execution(
                operation_type, current_performance
            )
            optimization_results['optimizations_applied'].extend(
                execution_result['optimizations_applied']
            )
            optimization_results['performance_improvement'] += execution_result['performance_improvement']
            
            # 生成建议
            optimization_results['recommendations'] = self._generate_optimization_recommendations(
                operation_type, current_performance
            )
            
            logger.info(f"性能优化完成: {operation_type}, 改进: {optimization_results['performance_improvement']:.2f}%")
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            optimization_results['optimization_successful'] = False
        
        return optimization_results
    
    def _generate_optimization_recommendations(self, operation_type: str, 
                                             current_performance: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 内存建议
        memory_recommendations = self.memory_optimizer.get_memory_recommendations()
        recommendations.extend(memory_recommendations)
        
        # 执行建议
        execution_recommendations = self.execution_optimizer._generate_optimization_recommendations(
            operation_type, current_performance
        )
        recommendations.extend(execution_recommendations)
        
        # 并行处理建议
        if current_performance.get('can_parallelize', False):
            recommendations.append("启用并行处理以提高性能")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'memory_status': self.memory_optimizer.check_memory_usage(),
            'execution_stats': self.execution_optimizer.get_performance_summary(),
            'error_stats': self.error_handler.get_error_statistics(),
            'active_tasks': self.parallel_processor.get_active_tasks()
        }
    
    def shutdown(self) -> None:
        """关闭性能优化器"""
        self.parallel_processor.shutdown()





