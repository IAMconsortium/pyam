import logging
import sys

from pyam.logger import logger, adjust_log_level


def test_context_adjust_log_level():
    assert logger().getEffectiveLevel() == 20
    with adjust_log_level():
        assert logger().getEffectiveLevel() == 40
    assert logger().getEffectiveLevel() == 20


def test_adjusting_level_for_not_initialized_logger():
    # de-initialize logger to simulate adjust_log_level called before logger
    pyam_logger = sys.modules['pyam.logger']
    pyam_logger._LOGGER = None
    with adjust_log_level():
        pass


def test_logger_namespacing(test_df, caplog):
    with caplog.at_level(logging.INFO, logger="pyam"):
        test_df.filter(model="junk")

    assert caplog.record_tuples == [(
        "pyam.core",  # namespacing
        logging.WARNING,  # level
        "Filtered IamDataFrame is empty!",  # message
    )]


def test_adjusting_logger_level(test_df, caplog):
    def throw_warning():
        logging.warning("This is a root warning")

    with caplog.at_level(logging.INFO, logger="pyam"):
        test_df.filter(model="junk")
        throw_warning()

    assert caplog.record_tuples == [
        ("pyam.core", logging.WARNING, "Filtered IamDataFrame is empty!"),
        ("root", logging.WARNING, "This is a root warning"),
    ]

    with caplog.at_level(logging.ERROR, logger="pyam"):
        test_df.filter(model="junk")
        throw_warning()

    # only the root warning should come through now i.e. we can silence pyam
    # without silencing everything
    assert caplog.record_tuples == [
        ("root", logging.WARNING, "This is a root warning"),
    ]
