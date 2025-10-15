"""
日志工具模块
提供日志配置、错误记录、性能监控等功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import time
import functools
import traceback

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """开始计时"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回持续时间"""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        self.durations[name] = duration
        del self.start_times[name]
        return duration
    
    def get_duration(self, name: str) -> float:
        """获取持续时间"""
        return self.durations.get(name, 0.0)
    
    def get_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        return self.durations.copy()

def performance_monitor(name: str):
    """性能监控装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = getattr(args[0], 'performance_monitor', None)
            if monitor is None:
                monitor = PerformanceMonitor()
                args[0].performance_monitor = monitor
            
            monitor.start_timer(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(name)
                logger = logging.getLogger(func.__module__)
                logger.info(f"性能监控 [{name}]: {duration:.4f}秒")
        return wrapper
    return decorator

class LoggerConfig:
    """日志配置类"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file
        self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.max_file_size = max_file_size
        self.backup_count = backup_count
    
    def setup_logger(self, name: str = "sbi_agent") -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(self.log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if self.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_file, 
                maxBytes=self.max_file_size, 
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(self.log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """处理异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        self.logger.error(
            "未捕获的异常",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """记录错误"""
        error_msg = f"错误 {context}: {str(error)}"
        self.logger.error(error_msg)
        self.logger.debug(f"错误详情: {traceback.format_exc()}")
    
    def log_warning(self, message: str, context: str = "") -> None:
        """记录警告"""
        warning_msg = f"警告 {context}: {message}"
        self.logger.warning(warning_msg)
    
    def log_info(self, message: str, context: str = "") -> None:
        """记录信息"""
        info_msg = f"信息 {context}: {message}"
        self.logger.info(info_msg)

class SBIProgressLogger:
    """SBI进度日志器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.iteration = 0
        self.start_time = None
    
    def start_calibration(self, total_iterations: int) -> None:
        """开始校准"""
        self.iteration = 0
        self.start_time = time.time()
        self.logger.info(f"SBI校准开始，总迭代次数: {total_iterations}")
    
    def log_iteration(self, iteration: int, metrics: Dict[str, float]) -> None:
        """记录迭代"""
        self.iteration = iteration
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"迭代 {iteration}, 耗时: {elapsed_time:.2f}s, 指标: {metrics_str}")
    
    def log_convergence(self, converged: bool, reason: str = "") -> None:
        """记录收敛状态"""
        status = "收敛" if converged else "未收敛"
        self.logger.info(f"SBI校准{status}: {reason}")
    
    def end_calibration(self, final_metrics: Dict[str, float]) -> None:
        """结束校准"""
        total_time = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"SBI校准完成，总耗时: {total_time:.2f}s")
        
        for metric, value in final_metrics.items():
            self.logger.info(f"最终指标 {metric}: {value:.4f}")

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> logging.Logger:
    """设置日志系统"""
    
    # 创建日志目录
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir_path / f"sbi_agent_{timestamp}.log")
    
    # 配置日志
    config = LoggerConfig(
        log_level=log_level,
        log_file=log_file
    )
    
    logger = config.setup_logger("sbi_agent")
    
    # 设置异常处理
    error_handler = ErrorHandler(logger)
    sys.excepthook = error_handler.handle_exception
    
    logger.info("日志系统初始化完成")
    return logger

def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)





