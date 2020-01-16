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


def deprecation_warning(msg):
    """Write deprecation warning to log"""
    warn = 'This method is deprecated and will be removed in future versions.'
    logger.warning('{} {}'.format(warn, msg))
