from typing import Union

__all__ = ['get_integer_time_unit']


def get_integer_time_unit(time_unit: Union[int, str]):
    if isinstance(time_unit, int):
        return time_unit
    if isinstance(time_unit, str):
        if time_unit.lower() in ('week', 'weekly'):
            return 52
        if time_unit.lower() in ('month', 'monthly'):
            return 12
        if time_unit.lower() in ('quarter', 'quarterly'):
            return 4
        if time_unit.lower() in ('semi-annual', 'semi-annually'):
            return 2
        if time_unit.lower() in ('annual', 'annually', 'year', 'yearly'):
            return 1

    raise ValueError(f"Unacceptable time_unit argument, {time_unit}")
