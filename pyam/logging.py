from contextlib import contextmanager
import logging
import warnings

logger = logging.getLogger(__name__)


@contextmanager
def adjust_log_level(logger="pyam", level="ERROR"):
    """Context manager to change log level"""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


def deprecation_warning(msg, type="This method", stacklevel=3):
    """Write deprecation warning to log"""
    warn = "is deprecated and will be removed in future versions."
    warnings.warn(
        "{} {} {}".format(type, warn, msg), DeprecationWarning, stacklevel=stacklevel
    )


class ConfigPseudoHandler(logging.Handler):
    """Pseudo logging handler to defer configuring logging until the first message

    Registers itself as a handler for the provided logger and temporarily
    sets the logger as sensitive to INFO messages. Upon receival of the first
    message (of at least INFO level), it configures logging with the provided
    `config_kwargs` and prints `log_msg`

    Parameters
    ----------
    logger : logging.Logger
        Logger to listen for the first message
    log_msg : str, optional
        Message to print once logging is configured, by default None
    **config_kwargs
        Arguments to pass on to logging.basicConfig
    """

    def __init__(self, logger, log_msg=None, **config_kwargs):
        super().__init__()

        self.logger = logger
        self.log_msg = log_msg
        self.config_kwargs = config_kwargs

        self.logger.addHandler(self)

        # temporarily set the logging level to a non-standard value,
        # slightly below logging.INFO == 20 and use that as a sentinel
        # to switch back to logging.NOTSET later
        self.logger.setLevel(19)

    def emit(self, record):
        self.logger.removeHandler(self)

        if self.logger.level == 19:
            self.logger.setLevel(logging.NOTSET)

        if not self.logger.root.hasHandlers():
            logging.basicConfig(**self.config_kwargs)

            if self.log_msg is not None:
                self.logger.info(self.log_msg)


# Give the Handler a function like alias
defer_logging_config = ConfigPseudoHandler
