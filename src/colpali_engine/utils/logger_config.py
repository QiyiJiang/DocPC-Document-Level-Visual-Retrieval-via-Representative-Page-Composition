import os
import logging
import sys
from datetime import datetime

# 日志文件名格式
DATETIME = datetime.now().strftime('%Y-%m-%d-%H%M%S')
# DATETIME = "debug" # 为了方便，调试的时候输出到 debug.log 文件

def setup_logger(name, log_file=None):
    """Function to setup logger with the given name and log file."""
    level = logging.INFO
    loggers = logging.getLogger(name)
    loggers.setLevel(level)

    # 检查 file handler 是否已存在
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in loggers.handlers)
    if not has_file_handler:
        if log_file is None:
            os.makedirs("logs", exist_ok=True)
            LOG_FILE = f'logs/project-{DATETIME}.log'
        else:
            os.makedirs(log_file, exist_ok=True)
            LOG_FILE = f'{log_file}/project-{DATETIME}.log'
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        loggers.addHandler(file_handler)

    # 检查 console handler 是否已存在
    has_console_handler = any(isinstance(h, logging.StreamHandler) for h in loggers.handlers)
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        loggers.addHandler(console_handler)

    # 禁用传播到根logger，防止日志重复
    loggers.propagate = False

    return loggers

logger = setup_logger('ColPali')