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
