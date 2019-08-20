from pyam import logger
from pyam.logger import adjust_log_level


def test_context_adjust_log_level():
    assert logger().getEffectiveLevel() == 20
    with adjust_log_level():
        assert logger().getEffectiveLevel() == 40
    assert logger().getEffectiveLevel() == 20
