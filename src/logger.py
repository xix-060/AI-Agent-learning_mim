"""日志工具 - 统一封装标准库 logging 模块。

用法:
    from src.logger import get_logger
    logger = get_logger("train")
    logger.info("开始训练")
    logger.warning("数据量偏少")
    logger.error("加载模型失败", exc_info=True)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 统一格式：时间 | 级别 | 模块名 | 消息
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 记录已配置过的 logger，避免重复添加 handler（logging 最常见坑）
_configured_loggers: set[str] = set()


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    max_bytes: int = 2 * 1024 * 1024,  # 单文件上限 2MB
    backup_count: int = 3,
) -> logging.Logger:
    """获取一个配置好的 logger。

    控制台输出始终启用；指定 log_file 时追加滚动文件输出。
    重复用同名 name 调用不会重复添加 handler。

    Args:
        name: logger 名称，通常用模块名或场景名（如 "train"、"agent"）。
        level: 日志级别，默认 INFO。
        log_file: 日志文件路径，None 表示只输出到控制台。
        max_bytes: 单个日志文件最大字节数，超出后滚动分片。
        backup_count: 保留的历史分片数。

    Returns:
        配置好的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)

    # 已配置过直接返回，避免重复 handler 导致日志重复输出
    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # 不往 root logger 传，防止重复打印

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    # 1. 控制台 handler（始终启用）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 handler（可选，带滚动分片）
    if log_file is not None:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # 文件 handler 失败不阻塞整体，控制台仍可用
            logger.warning("无法创建日志文件 %s：%s，仅使用控制台输出", log_path, e)

    _configured_loggers.add(name)
    return logger


def get_logger(name: str = "ai-agent") -> logging.Logger:
    """快捷获取默认配置的 logger（INFO 级别，仅控制台）。

    Args:
        name: logger 名称，默认 "ai-agent"。

    Returns:
        配置好的 logging.Logger 实例。
    """
    return setup_logger(name)


if __name__ == "__main__":
    # 自测：演示各级别输出
    log = get_logger("demo")
    log.debug("debug 不会显示（默认 INFO）")
    log.info("info 正常信息")
    log.warning("warning 警告")
    log.error("error 错误")
    log.info("带文件输出的演示见 setup_logger(log_file=...)")
