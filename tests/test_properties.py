import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("dask")
pytest.importorskip("cftime")

import dask
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given, note, settings

import flox
from flox.core import groupby_reduce, groupby_scan
from flox.xrutils import _contains_cftime_datetimes, _to_pytimedelta, datetime_to_numeric, isnull, notnull

from . import BLOCKWISE_FUNCS, assert_equal
from .strategies import all_arrays, by_arrays, chunked_arrays, func_st, numeric_dtypes, numeric_like_arrays
from .strategies import chunks as chunks_strategy

dask.config.set(scheduler="sync")


def ffill(array, axis, dtype=None):
    return flox.aggregate_flox.ffill(np.zeros(array.shape[-1], dtype=int), array, axis=axis)


def bfill(array, axis, dtype=None):
    return flox.aggregate_flox.ffill(
        np.zeros(array.shape[-1], dtype=int),
        array[::-1],
        axis=axis,
    )[::-1]


NUMPY_SCAN_FUNCS: dict[str, Callable] = {
    "nancumsum": np.nancumsum,
    "ffill": ffill,
    "bfill": bfill,
}  # "cumsum": np.cumsum,


def not_overflowing_array(array: np.ndarray[Any, Any]) -> bool:
    if array.dtype.kind in "Mm":
        array = array.view(np.int64)
    if array.dtype.kind == "f":
        info = np.finfo(array.dtype)
    elif array.dtype.kind in ["i", "u"]:
        info = np.iinfo(array.dtype)  # type: ignore[assignment]
    else:
        return True

    array = array.ravel()
    array = array[notnull(array)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = bool(np.all((array < info.max / array.size) & (array > info.min / array.size)))
    # note(f"returning {result}, {array.min()} vs {info.min}, {array.max()} vs {info.max}")
    return result


@given(
    data=st.data(),
    array=st.one_of(all_arrays, chunked_arrays()),
    func=func_st,
)
@settings(deadline=None)
def test_groupby_reduce(data, array, func: str) -> None:
    # overflow behaviour differs between bincount and sum (for example)
    assume(not_overflowing_array(array))
    # TODO: fix var for complex numbers upstream
    assume(not (("quantile" in func or "var" in func or "std" in func) and array.dtype.kind == "c"))
    assume(not ("quantile" in func and array.dtype.kind == "b"))
    # arg* with nans in array are weird
    assume("arg" not in func and not np.any(isnull(array).ravel()))

    # TODO: funny bugs with overflows here
    is_cftime = _contains_cftime_datetimes(array)
    assume(not (is_cftime and func in ["prod", "nanprod"]))

    axis = -1
    by = data.draw(
        by_arrays(
            elements={
                "alphabet": st.just("a"),
                "min_value": 1,
                "max_value": 1,
                "min_size": 1,
                "max_size": 1,
            },
            shape=st.just((array.shape[-1],)),
        )
    )
    if func in BLOCKWISE_FUNCS and isinstance(array, dask.array.Array):
        array = array.rechunk({axis: -1})
    assert len(np.unique(by)) == 1
    kwargs = {"q": 0.8} if "quantile" in func else {}
    flox_kwargs: dict[str, Any] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        actual, *_ = groupby_reduce(
            array,
            by,
            func=func,
            axis=axis,
            engine="numpy",
            **flox_kwargs,
            finalize_kwargs=kwargs,
        )

        # numpy-groupies always does the calculation in float64
        if (
            ("var" in func or "std" in func or "sum" in func or "mean" in func or "quantile" in func)
            and array.dtype.kind == "f"
            and array.dtype.itemsize != 8
        ):
            # bincount always accumulates in float64,
            # casting to float64 handles std more like npg does.
            # Setting dtype=float64 works fine for sum, mean.
            cast_to = array.dtype
            array = array.astype(np.float64)
            note(f"casting array to float64, cast_to={cast_to!r}")
        else:
            cast_to = None

        if array.dtype.kind in "Mm":
            array = array.view(np.int64)
            cast_to = array.dtype
        elif is_cftime:
            offset = array.min()
            array = datetime_to_numeric(array, offset, datetime_unit="us")
        note(("kwargs:", kwargs, "cast_to:", cast_to))
        expected = getattr(np, func)(array, axis=axis, keepdims=True, **kwargs)
        if cast_to is not None:
            note(("casting to:", cast_to))
            expected = expected.astype(cast_to)
            actual = actual.astype(cast_to)
        if is_cftime:
            expected = _to_pytimedelta(expected, unit="us") + offset

    note(("expected: ", expected, "actual: ", actual))
    tolerance = {"atol": 1e-15}
    assert_equal(expected, actual, tolerance)


@given(
    data=st.data(),
    array=chunked_arrays(arrays=numeric_like_arrays),
    func=func_st,
)
def test_groupby_reduce_numpy_vs_dask(data, array, func: str) -> None:
    numpy_array = array.compute()
    # overflow behaviour differs between bincount and sum (for example)
    assume(not_overflowing_array(numpy_array))
    # TODO: fix var for complex numbers upstream
    assume(not (("quantile" in func or "var" in func or "std" in func) and array.dtype.kind == "c"))
    # # arg* with nans in array are weird
    assume("arg" not in func and not np.any(isnull(numpy_array.ravel())))
    if func in ["nanmedian", "nanquantile", "median", "quantile"]:
        array = array.rechunk({-1: -1})

    axis = -1
    by = data.draw(by_arrays(shape=st.just((array.shape[-1],))))
    kwargs = {"q": 0.8} if "quantile" in func else {}
    flox_kwargs: dict[str, Any] = {}

    kwargs = dict(
        func=func,
        axis=axis,
        engine="numpy",
        **flox_kwargs,
        finalize_kwargs=kwargs,
    )
    result_dask, *_ = groupby_reduce(array, by, **kwargs)
    result_numpy, *_ = groupby_reduce(numpy_array, by, **kwargs)
    assert_equal(result_numpy, result_dask)


@settings(report_multiple_bugs=False)
@given(
    data=st.data(),
    array=chunked_arrays(arrays=numeric_like_arrays),
    func=st.sampled_from(tuple(NUMPY_SCAN_FUNCS)),
)
def test_scans(data, array: dask.array.Array, func: str) -> None:
    if "cum" in func:
        assume(not_overflowing_array(np.asarray(array)))

    by = data.draw(by_arrays(shape=st.just((array.shape[-1],))))
    axis = array.ndim - 1

    # Too many float32 edge-cases!
    if "cum" in func and array.dtype.kind == "f" and array.dtype.itemsize == 4:
        assume(False)
    numpy_array = array.compute()
    if numpy_array.dtype.kind not in "Mm":
        assume((np.abs(numpy_array) < 2**53).all())

    if numpy_array.dtype.kind in "Mm":
        dtype = numpy_array.dtype
        asnumeric = numpy_array.view(np.int64)
    else:
        asnumeric = numpy_array
        dtype = NUMPY_SCAN_FUNCS[func](asnumeric[..., [0]], axis=axis).dtype
    expected = np.empty_like(numpy_array, dtype=dtype)
    group_idx, uniques = pd.factorize(by)
    for i in range(len(uniques)):
        mask = group_idx == i
        if not mask.any():
            note((by, group_idx, uniques))
            raise ValueError
        expected[..., mask] = NUMPY_SCAN_FUNCS[func](asnumeric[..., mask], axis=axis)

    if dtype:
        expected = expected.astype(dtype)
    note((numpy_array, group_idx, array.chunks))

    tolerance = {"rtol": 1e-13, "atol": 1e-15}
    actual = groupby_scan(numpy_array, by, func=func, axis=-1, dtype=dtype)
    assert_equal(actual, expected, tolerance)

    actual = groupby_scan(array, by, func=func, axis=-1, dtype=dtype)
    assert_equal(actual, expected, tolerance)


@given(data=st.data(), array=chunked_arrays())
def test_ffill_bfill_reverse(data, array: dask.array.Array) -> None:
    by = data.draw(by_arrays(shape=st.just((array.shape[-1],))))

    def reverse(arr):
        return arr[..., ::-1]

    forward = groupby_scan(array, by, func="ffill")
    as_numpy = groupby_scan(array.compute(), by, func="ffill")
    assert_equal(forward, as_numpy)

    backward = groupby_scan(array, by, func="bfill")
    as_numpy = groupby_scan(array.compute(), by, func="bfill")
    assert_equal(backward, as_numpy)

    backward_reversed = reverse(groupby_scan(reverse(array), reverse(by), func="bfill"))
    assert_equal(forward, backward_reversed)

    forward_reversed = reverse(groupby_scan(reverse(array), reverse(by), func="ffill"))
    assert_equal(forward_reversed, backward)


@given(
    data=st.data(),
    array=chunked_arrays(),
    func=st.sampled_from(["first", "last", "nanfirst", "nanlast"]),
)
def test_first_last(data, array: dask.array.Array, func: str) -> None:
    by = data.draw(by_arrays(shape=st.just((array.shape[-1],))))

    INVERSES = {
        "first": "last",
        "last": "first",
        "nanfirst": "nanlast",
        "nanlast": "nanfirst",
    }
    MATES = {
        "first": "nanfirst",
        "last": "nanlast",
        "nanfirst": "first",
        "nanlast": "last",
    }
    inverse = INVERSES[func]
    mate = MATES[func]

    if func in ["first", "last"]:
        array = array.rechunk((*array.chunks[:-1], -1))

    for arr in [array, array.compute()]:
        forward, *fg = groupby_reduce(arr, by, func=func, engine="flox")
        reverse, *rg = groupby_reduce(arr[..., ::-1], by[..., ::-1], func=inverse, engine="flox")

        assert forward.dtype == reverse.dtype
        assert forward.dtype == arr.dtype

        assert_equal(fg, rg)
        assert_equal(forward, reverse)

    if arr.dtype.kind == "f" and not isnull(array.compute()).any():
        if mate in ["first", "last"]:
            array = array.rechunk((*array.chunks[:-1], -1))

        first, *_ = groupby_reduce(array, by, func=func, engine="flox")
        second, *_ = groupby_reduce(array, by, func=mate, engine="flox")
        assert_equal(first, second)


@given(data=st.data(), func=st.sampled_from(["nanfirst", "nanlast"]))
def test_first_last_useless(data, func):
    shape = data.draw(npst.array_shapes())
    by = data.draw(by_arrays(shape=st.just(shape[slice(-1, None)])))
    chunks = data.draw(chunks_strategy(shape=shape))
    array = np.zeros(shape, dtype=np.int8)
    if chunks is not None:
        array = dask.array.from_array(array, chunks=chunks)
    actual, groups = groupby_reduce(array, by, axis=-1, func=func, engine="numpy")
    expected = np.zeros(shape[:-1] + (len(groups),), dtype=array.dtype)
    assert_equal(actual, expected)


@given(
    func=st.sampled_from(["sum", "prod", "nansum", "nanprod"]),
    engine=st.sampled_from(["numpy", "flox"]),
    array_dtype=st.none() | numeric_dtypes,
    dtype=st.none() | numeric_dtypes,
)
def test_agg_dtype_specified(func, array_dtype, dtype, engine):
    # regression test for GH388
    counts = np.array([0, 2, 1, 0, 1], dtype=array_dtype)
    group = np.array([1, 1, 1, 2, 2])
    actual, _ = groupby_reduce(
        counts,
        group,
        expected_groups=(np.array([1, 2]),),
        func=func,
        dtype=dtype,
        engine=engine,
    )
    expected = getattr(np, func)(counts, keepdims=True, dtype=dtype)
    assert actual.dtype == expected.dtype
