import pytest
from pyam.string_utils import format_superscript, format_subscript,\
    unformat_superscript, unformat_subscript


NORMALSCRIPT = '0123456789x'
SUBSCRIPT = '₀₁₂₃₄₅₆₇₈₉ₓ'
SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹ˣ'


@pytest.mark.parametrize("s, sup", [
    ['foo1bar2', 'foo¹bar²'],
    [['foo1bar2', 'foo1bar3'], ['foo¹bar²', 'foo¹bar³']]
])
def test_superscript_formatting(s, sup):
    """Test that formatting from and to superscript works"""
    assert format_superscript(s) == sup
    assert unformat_superscript(sup) == s


@pytest.mark.parametrize("s, sub", [
    ['foo1bar2', 'foo₁bar₂'],
    [['foo1bar2', 'foo1bar3'], ['foo₁bar₂', 'foo₁bar₃']]
])
def test_subscript_formatting(s, sub):
    """Test that formatting from and to subscript works"""
    assert format_subscript(s) == sub
    assert unformat_subscript(sub) == s
