from contextlib import contextmanager


@contextmanager
def adjust_log_level(logger, level='ERROR'):
    """Context manager to change log level"""
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)
