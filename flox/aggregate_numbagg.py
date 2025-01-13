from functools import partial

import numbagg
import numbagg.grouped
import numpy as np
from packaging.version import Version

NUMBAGG_SUPPORTS_DDOF = Version(numbagg.__version__) >= Version("0.7.0")

DEFAULT_FILL_VALUE = {
    "nansum": 0,
    "nanmean": np.nan,
    "nanvar": np.nan,
    "nanstd": np.nan,
    "nanmin": np.nan,
    "nanmax": np.nan,
    "nanany": False,
    "nanall": False,
    "nansum_of_squares": 0,
    "nanprod": 1,
    "nancount": 0,
    "nanargmax": np.nan,
    "nanargmin": np.nan,
    "nanfirst": np.nan,
    "nanlast": np.nan,
}

CAST_TO = {
    # "nansum": {np.bool_: np.int64},
    "nanmean": {np.int_: np.float64},
    "nanvar": {np.int_: np.float64},
    "nanstd": {np.int_: np.float64},
    "nanfirst": {np.datetime64: np.int64, np.timedelta64: np.int64},
    "nanlast": {np.datetime64: np.int64, np.timedelta64: np.int64},
}


FILLNA = {"nansum": 0, "nanprod": 1}


def _numbagg_wrapper(
    group_idx,
    array,
    *,
    func,
    axis=-1,
    size=None,
    fill_value=None,
    dtype=None,
    **kwargs,
):
    cast_to = CAST_TO.get(func, None)
    if cast_to:
        for from_, to_ in cast_to.items():
            if np.issubdtype(array.dtype, from_):
                array = array.astype(to_, copy=False)

    func_ = getattr(numbagg.grouped, f"group_{func}")

    result = func_(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        **kwargs,
        # The following are unsupported
        # fill_value=fill_value,
        # dtype=dtype,
    ).astype(dtype, copy=False)

    return result


def nanvar(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, ddof=0):
    kwargs = {}
    if NUMBAGG_SUPPORTS_DDOF:
        kwargs["ddof"] = ddof
    elif ddof != 1:
        raise ValueError("Need numbagg >= v0.7.0 to support ddof != 1")
    return _numbagg_wrapper(
        group_idx,
        array,
        axis=axis,
        size=size,
        func="nanvar",
        **kwargs,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanstd(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, ddof=0):
    kwargs = {}
    if NUMBAGG_SUPPORTS_DDOF:
        kwargs["ddof"] = ddof
    elif ddof != 1:
        raise ValueError("Need numbagg >= v0.7.0 to support ddof != 1")
    return _numbagg_wrapper(
        group_idx,
        array,
        axis=axis,
        size=size,
        func="nanstd",
        **kwargs,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanlen(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if array.dtype.kind in "US":
        array = np.broadcast_to(np.array([1]), array.shape)
    return _numbagg_wrapper(
        group_idx,
        array,
        axis=axis,
        size=size,
        func="nancount",
        # fill_value=fill_value,
        # dtype=dtype,
    )


nansum = partial(_numbagg_wrapper, func="nansum")
nanmean = partial(_numbagg_wrapper, func="nanmean")
nanprod = partial(_numbagg_wrapper, func="nanprod")
nansum_of_squares = partial(_numbagg_wrapper, func="nansum_of_squares")
nanprod = partial(_numbagg_wrapper, func="nanprod")
nanfirst = partial(_numbagg_wrapper, func="nanfirst")
nanlast = partial(_numbagg_wrapper, func="nanlast")
# nanargmax = partial(_numbagg_wrapper, func="nanargmax)
# nanargmin = partial(_numbagg_wrapper, func="nanargmin)
nanmax = partial(_numbagg_wrapper, func="nanmax")
nanmin = partial(_numbagg_wrapper, func="nanmin")
any = partial(_numbagg_wrapper, func="nanany")
all = partial(_numbagg_wrapper, func="nanall")

# sum = nansum
# mean = nanmean
# sum_of_squares = nansum_of_squares
