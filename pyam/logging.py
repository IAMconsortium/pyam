from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def adjust_log_level(logger, level='ERROR'):
    """Context manager to change log level"""
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


def deprecation_warning(msg, type='This method'):
    """Write deprecation warning to log"""
    warn = 'is deprecated and will be removed in future versions.'
    logger.warning('{} {} {}'.format(type, warn, msg))
