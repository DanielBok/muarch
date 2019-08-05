from muarch.funcs.time_unit import get_integer_time_unit

import pytest


@pytest.mark.parametrize("time_unit, expected", [
    (4, 4),
    ("week", 52),
    ("weekly", 52),
    ("month", 12),
    ("monthly", 12),
    ("quarter", 4),
    ("quarterly", 4),
    ("semi-annual", 2),
    ("semi-annually", 2),
    ("annual", 1),
    ("annually", 1),
    ("year", 1),
    ("yearly", 1),
])
def test_get_integer_time_unit(time_unit, expected):
    assert get_integer_time_unit(time_unit) == expected


@pytest.mark.parametrize("time_unit", [
    4.0,
    "wrong-value"
])
def test_get_integer_time_unit_errors(time_unit):
    with pytest.raises(ValueError):
        get_integer_time_unit(time_unit)
