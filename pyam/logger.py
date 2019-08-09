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


class ignoreWarnings():
    """Fixture to ignore logging messages below `level`"""
    def __init__(self, level='ERROR'):
        self.level = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(level)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        _LOGGER.setLevel(self.level)
