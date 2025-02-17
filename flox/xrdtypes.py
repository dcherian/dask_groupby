import functools

import numpy as np
from numpy.typing import DTypeLike

from . import xrutils as utils

# Use as a sentinel value to indicate a dtype appropriate NA value.
NA = utils.ReprObject("<NA>")


@functools.total_ordering
class AlwaysGreaterThan:
    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


@functools.total_ordering
class AlwaysLessThan:
    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


# Equivalence to np.inf (-np.inf) for object-type
INF = AlwaysGreaterThan()
NINF = AlwaysLessThan()


def maybe_promote(dtype):
    """Simpler equivalent of pandas.core.common._maybe_promote

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    dtype : Promoted dtype that can hold missing values.
    fill_value : Valid missing value for the promoted dtype.
    """
    # N.B. these casting rules should match pandas
    if np.issubdtype(dtype, np.floating):
        fill_value = np.nan
    elif np.issubdtype(dtype, np.timedelta64):
        # See https://github.com/numpy/numpy/issues/10685
        # np.timedelta64 is a subclass of np.integer
        # Check np.timedelta64 before np.integer
        fill_value = np.timedelta64("NaT")
    elif np.issubdtype(dtype, np.integer):
        dtype = np.float32 if dtype.itemsize <= 2 else np.float64
        fill_value = np.nan
    elif np.issubdtype(dtype, np.complexfloating):
        fill_value = np.nan + np.nan * 1j
    elif np.issubdtype(dtype, np.datetime64):
        fill_value = np.datetime64("NaT")
    else:
        dtype = object
        fill_value = np.nan
    return np.dtype(dtype), fill_value


NAT_TYPES = {np.datetime64("NaT").dtype, np.timedelta64("NaT").dtype}


def get_fill_value(dtype):
    """Return an appropriate fill value for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : Missing value corresponding to this dtype.
    """
    _, fill_value = maybe_promote(dtype)
    return fill_value


def get_pos_infinity(dtype, max_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    max_for_int : bool
        Return np.iinfo(dtype).max instead of np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, np.floating):
        return np.inf

    if issubclass(dtype.type, np.integer):
        if max_for_int:
            dtype = np.int64 if dtype.kind in "Mm" else dtype
            return np.iinfo(dtype).max
        else:
            return np.inf

    if issubclass(dtype.type, np.complexfloating):
        return np.inf + 1j * np.inf

    if issubclass(dtype.type, np.bool_):
        return True

    return INF


def get_neg_infinity(dtype, min_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    min_for_int : bool
        Return np.iinfo(dtype).min instead of -np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """

    if is_datetime_like(dtype):
        unit, _ = np.datetime_data(dtype)
        return dtype.type(np.iinfo(np.int64).min + 1, unit)

    if issubclass(dtype.type, np.floating):
        return -np.inf

    if issubclass(dtype.type, np.integer):
        if min_for_int:
            return np.iinfo(dtype).min
        else:
            return -np.inf

    if issubclass(dtype.type, np.complexfloating):
        return -np.inf - 1j * np.inf

    if issubclass(dtype.type, np.bool_):
        return False

    return NINF


def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types"""
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


def _normalize_dtype(
    dtype: DTypeLike, array_dtype: np.dtype, preserves_dtype: bool, fill_value=None
) -> np.dtype:
    if dtype is None:
        if not preserves_dtype:
            dtype = _maybe_promote_int(array_dtype)
        else:
            dtype = array_dtype
    if dtype is np.floating:
        # mean, std, var always result in floating
        # but we preserve the array's dtype if it is floating
        if array_dtype.kind in "fcmM":
            dtype = array_dtype
        else:
            dtype = np.dtype("float64")
    elif not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if fill_value not in [None, INF, NINF, NA]:
        dtype = np.result_type(dtype, fill_value)
    return dtype


def _maybe_promote_int(dtype) -> np.dtype:
    # https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    # The dtype of a is used by default unless a has an integer dtype of less precision
    # than the default platform integer.
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if dtype.kind == "i":
        dtype = np.result_type(dtype, np.int_)
    elif dtype.kind == "u":
        dtype = np.result_type(dtype, np.uint)
    return dtype


def _get_fill_value(dtype, fill_value):
    """Returns dtype appropriate infinity. Returns +Inf equivalent for None."""
    if fill_value in [None, NA] and dtype.kind in "US":
        return ""
    if fill_value == INF or fill_value is None:
        return get_pos_infinity(dtype, max_for_int=True)
    if fill_value == NINF:
        return get_neg_infinity(dtype, min_for_int=True)
    if fill_value == NA:
        if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
            return np.nan
        # This is madness, but npg checks that fill_value is compatible
        # with array dtype even if the fill_value is never used.
        elif np.issubdtype(dtype, np.integer):
            return get_neg_infinity(dtype, min_for_int=True)
        elif np.issubdtype(dtype, np.timedelta64):
            return np.timedelta64("NaT")
        elif np.issubdtype(dtype, np.datetime64):
            return np.datetime64("NaT")
        else:
            return None
    return fill_value
