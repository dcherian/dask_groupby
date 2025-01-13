# The functions defined here were copied based on the source code
# defined in xarray

import datetime
import importlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from packaging.version import Version


def module_available(module: str, minversion: str | None = None) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    has = importlib.util.find_spec(module) is not None
    if has:
        mod = importlib.import_module(module)
        return Version(mod.__version__) >= Version(minversion) if minversion is not None else True
    else:
        return False


if module_available("numpy", minversion="2.0.0"):
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.numeric import normalize_axis_index  # type: ignore[no-redef]


try:
    import cftime
except ImportError:
    cftime = None


try:
    import dask.array

    dask_array_type = dask.array.Array
except ImportError:
    dask_array_type = ()  # type: ignore[assignment, misc]


def asarray(data, xp=np):
    return data if is_duck_array(data) else xp.asarray(data)


def is_duck_array(value: Any) -> bool:
    """Checks if value is a duck array."""
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (
            (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
            or hasattr(value, "__array_namespace__")
        )
    )


def is_chunked_array(x) -> bool:
    """True if dask or cubed"""
    return is_duck_dask_array(x) or (is_duck_array(x) and hasattr(x, "chunks"))


def is_dask_collection(x):
    try:
        import dask

        return dask.is_dask_collection(x)

    except ImportError:
        return False


def is_duck_dask_array(x):
    return is_duck_array(x) and is_dask_collection(x)


def is_duck_cubed_array(x):
    try:
        import cubed

        return is_duck_array(x) and isinstance(x, cubed.Array)
    except ImportError:
        return False


class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other) -> bool:
        if isinstance(other, ReprObject):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self._value))


def is_scalar(value: Any, include_0d: bool = True) -> bool:
    """Whether to treat a value as a scalar.

    Any non-iterable, string, dict, or 0-D array
    """
    NON_NUMPY_SUPPORTED_ARRAY_TYPES = (dask_array_type, pd.Index)

    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, str | bytes | dict)
        or not (
            isinstance(value, (Iterable,) + NON_NUMPY_SUPPORTED_ARRAY_TYPES)
            or hasattr(value, "__array_function__")
        )
    )


def notnull(data):
    if not is_duck_array(data):
        data = np.asarray(data)

    scalar_type = data.dtype.type
    if issubclass(scalar_type, np.bool_ | np.integer | np.character | np.void):
        # these types cannot represent missing values
        return np.ones_like(data, dtype=bool)
    else:
        out = isnull(data)
        np.logical_not(out, out=out)
        return out


def isnull(data):
    if not is_duck_array(data):
        data = np.asarray(data)
    scalar_type = data.dtype.type
    if issubclass(scalar_type, np.datetime64 | np.timedelta64):
        # datetime types use NaT for null
        # note: must check timedelta64 before integers, because currently
        # timedelta64 inherits from np.integer
        return np.isnat(data)
    elif issubclass(scalar_type, np.inexact):
        # float types use NaN for null
        return np.isnan(data)
    elif issubclass(scalar_type, np.bool_ | np.integer | np.character | np.void):
        # these types cannot represent missing values
        return np.zeros_like(data, dtype=bool)
    else:
        # at this point, array should have dtype=object
        if isinstance(data, (np.ndarray, dask_array_type)):  # noqa
            return pd.isnull(data)
        else:
            # Not reachable yet, but intended for use with other duck array
            # types. For full consistency with pandas, we should accept None as
            # a null value as well as NaN, but it isn't clear how to do this
            # with duck typing.
            return data != data


def datetime_to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    """Convert an array containing datetime-like data to numerical values.
    Convert the datetime array to a timedelta relative to an offset.
    Parameters
    ----------
    array : array-like
        Input data
    offset : None, datetime or cftime.datetime
        Datetime offset. If None, this is set by default to the array's minimum
        value to reduce round off errors.
    datetime_unit : {None, Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        If not None, convert output to a given datetime unit. Note that some
        conversions are not allowed due to non-linear relationships between units.
    dtype : dtype
        Output dtype.
    Returns
    -------
    array
        Numerical representation of datetime object relative to an offset.
    Notes
    -----
    Some datetime unit conversions won't work, for example from days to years, even
    though some calendars would allow for them (e.g. no_leap). This is because there
    is no `cftime.timedelta` object.
    """
    # TODO: make this function dask-compatible?
    # Set offset to minimum if not given
    if offset is None:
        if array.dtype.kind in "Mm":
            offset = _datetime_nanmin(array)
        else:
            offset = array.min()

    # Compute timedelta object.
    # For np.datetime64, this can silently yield garbage due to overflow.
    # One option is to enforce 1970-01-01 as the universal offset.

    # This map_blocks call is for backwards compatibility.
    # dask == 2021.04.1 does not support subtracting object arrays
    # which is required for cftime
    if is_duck_dask_array(array) and np.issubdtype(array.dtype, object):
        array = array.map_blocks(lambda a, b: a - b, offset, meta=array._meta)
    else:
        array = array - offset

    # Scalar is converted to 0d-array
    if not hasattr(array, "dtype"):
        array = np.array(array)

    # Convert timedelta objects to float by first converting to microseconds.
    if array.dtype.kind in "O":
        return py_timedelta_to_float(array, datetime_unit or "ns").astype(dtype)

    # Convert np.NaT to np.nan
    elif array.dtype.kind in "mM":
        # Convert to specified timedelta units.
        if datetime_unit:
            array = array / np.timedelta64(1, datetime_unit)
        return np.where(isnull(array), np.nan, array.astype(dtype))


def timedelta_to_numeric(value, datetime_unit="ns", dtype=float):
    """Convert a timedelta-like object to numerical values.

    Parameters
    ----------
    value : datetime.timedelta, numpy.timedelta64, pandas.Timedelta, str
        Time delta representation.
    datetime_unit : {Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        The time units of the output values. Note that some conversions are not allowed due to
        non-linear relationships between units.
    dtype : type
        The output data type.

    """
    import datetime as dt

    if isinstance(value, dt.timedelta):
        out = py_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, np.timedelta64):
        out = np_timedelta64_to_float(value, datetime_unit)
    elif isinstance(value, pd.Timedelta):
        out = pd_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, str):
        try:
            a = pd.to_timedelta(value)
        except ValueError:
            raise ValueError(f"Could not convert {value!r} to timedelta64 using pandas.to_timedelta")
        return py_timedelta_to_float(a, datetime_unit)
    else:
        raise TypeError(
            f"Expected value of type str, pandas.Timedelta, datetime.timedelta "
            f"or numpy.timedelta64, but received {type(value).__name__}"
        )
    return out.astype(dtype)


def _to_pytimedelta(array, unit="us"):
    return array.astype(f"timedelta64[{unit}]").astype(datetime.timedelta)


def np_timedelta64_to_float(array, datetime_unit):
    """Convert numpy.timedelta64 to float.

    Notes
    -----
    The array is first converted to microseconds, which is less likely to
    cause overflow errors.
    """
    array = array.astype("timedelta64[ns]").astype(np.float64)
    conversion_factor = np.timedelta64(1, "ns") / np.timedelta64(1, datetime_unit)
    return conversion_factor * array


def pd_timedelta_to_float(value, datetime_unit):
    """Convert pandas.Timedelta to float.

    Notes
    -----
    Built on the assumption that pandas timedelta values are in nanoseconds,
    which is also the numpy default resolution.
    """
    value = value.to_timedelta64()
    return np_timedelta64_to_float(value, datetime_unit)


def _timedelta_to_seconds(array):
    return np.reshape([a.total_seconds() for a in array.ravel()], array.shape) * 1e6


def py_timedelta_to_float(array, datetime_unit):
    """Convert a timedelta object to a float, possibly at a loss of resolution."""
    array = asarray(array)
    if is_duck_dask_array(array):
        array = array.map_blocks(_timedelta_to_seconds, meta=np.array([], dtype=np.float64))
    else:
        array = _timedelta_to_seconds(array)
    conversion_factor = np.timedelta64(1, "us") / np.timedelta64(1, datetime_unit)
    return conversion_factor * array


def _contains_cftime_datetimes(array) -> bool:
    """Check if an array contains cftime.datetime objects"""
    if cftime is None:
        return False
    else:
        if array.dtype == np.dtype("O") and array.size > 0:
            sample = array.ravel()[0]
            if is_duck_dask_array(sample):
                sample = sample.compute()
                if isinstance(sample, np.ndarray):
                    sample = sample.item()
            return isinstance(sample, cftime.datetime)
        else:
            return False


def _datetime_nanmin(array):
    """nanmin() function for datetime64.

    Caveats that this function deals with:

    - In numpy < 1.18, min() on datetime64 incorrectly ignores NaT
    - numpy nanmin() don't work on datetime64 (all versions at the moment of writing)
    - dask min() does not work on datetime64 (all versions at the moment of writing)
    """
    from .xrdtypes import is_datetime_like

    dtype = array.dtype
    assert is_datetime_like(dtype)
    # (NaT).astype(float) does not produce NaN...
    array = np.where(pd.isnull(array), np.nan, array.astype(float))
    array = np.nanmin(array)
    if isinstance(array, float):
        array = np.array(array)
    # ...but (NaN).astype("M8") does produce NaT
    return array.astype(dtype)


def _select_along_axis(values, idx, axis):
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def nanfirst(values, axis, keepdims=False):
    if isinstance(axis, tuple):
        (axis,) = axis
    values = np.asarray(values)
    axis = normalize_axis_index(axis, values.ndim)
    idx_first = np.argmax(~pd.isnull(values), axis=axis)
    result = _select_along_axis(values, idx_first, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result


def nanlast(values, axis, keepdims=False):
    if isinstance(axis, tuple):
        (axis,) = axis
    values = np.asarray(values)
    axis = normalize_axis_index(axis, values.ndim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(values)[rev], axis=axis)
    result = _select_along_axis(values, idx_last, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result
