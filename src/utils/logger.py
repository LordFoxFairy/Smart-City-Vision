import logging
from logging.handlers import RotatingFileHandler
import os
from configs.config import config

def setup_logger(name: str = "city_event_system") -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL.upper())
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 格式化日志
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    log_file = os.path.join(config.LOG_DIR, "system.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_ROTATION,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 创建全局日志实例
logger = setup_logger()
