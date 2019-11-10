# import logging
from contextlib import contextmanager

# # globally accessible logger
# _LOGGER = None


# def logger():
#     """Access global logger"""
#     global _LOGGER
#     if _LOGGER is None:
#         logging.basicConfig()
#         _LOGGER = logging.getLogger()
#         _LOGGER.setLevel('INFO')
#     return _LOGGER


@contextmanager
def adjust_log_level(logger, level='ERROR'):
    """Context manager to change log level"""
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)
