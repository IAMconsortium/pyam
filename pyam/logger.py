from contextlib import contextmanager
import logging

# globally accessible logger
_LOGGER = None


def logger():
    """Access global logger"""
    global _LOGGER
    if _LOGGER is None:
        logging.basicConfig()
        _LOGGER = logging.getLogger()
        _LOGGER.setLevel('INFO')
    return _LOGGER


@contextmanager
def adjust_log_level(level='ERROR'):
    """Context manager to change log level"""
    old_level = _LOGGER.getEffectiveLevel()
    _LOGGER.setLevel(level)
    yield
    _LOGGER.setLevel(old_level)
