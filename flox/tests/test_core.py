from __future__ import annotations

import itertools
from functools import partial, reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from numpy_groupies.aggregate_numpy import aggregate

from flox.aggregations import Aggregation
from flox.core import (
    _convert_expected_groups_to_index,
    _get_optimal_chunks_for_groups,
    _normalize_indexes,
    _validate_reindex,
    factorize_,
    find_group_cohorts,
    groupby_reduce,
    rechunk_for_cohorts,
    reindex_,
    subset_to_blocks,
)

from . import (
    assert_equal,
    assert_equal_tuple,
    has_dask,
    raise_if_dask_computes,
    requires_dask,
)

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
nan_labels = labels.astype(float)  # copy
nan_labels[:5] = np.nan
labels2d = np.array([labels[:5], np.flip(labels[:5])])

# isort:off
if has_dask:
    import dask
    import dask.array as da
    from dask.array import from_array

    dask.config.set(scheduler="sync")
else:

    def dask_array_ones(*args):
        return None


# isort:on

ALL_FUNCS = (
    "sum",
    "nansum",
    "argmax",
    pytest.param("nanargmax", marks=(pytest.mark.skip,)),
    "prod",
    "nanprod",
    "mean",
    "nanmean",
    "var",
    "nanvar",
    "std",
    "nanstd",
    "max",
    "nanmax",
    "min",
    "nanmin",
    "argmin",
    pytest.param("nanargmin", marks=(pytest.mark.skip,)),
    "any",
    "all",
    pytest.param("median", marks=(pytest.mark.skip,)),
    pytest.param("nanmedian", marks=(pytest.mark.skip,)),
)

if TYPE_CHECKING:
    from flox.core import T_Agg, T_Engine, T_ExpectedGroupsOpt, T_Method

    # Let anything through in kwargs for code readability, will likely miss a lot of
    # type errors within these dicts though:
    T_Kwargs = dict[str, Any]


def test_alignment_error():
    da = np.ones((12,))
    labels = np.ones((5,))

    with pytest.raises(ValueError):
        groupby_reduce(da, labels, func="mean")


@pytest.mark.parametrize("dtype", (float, int))
@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("expected_groups", [None, [0, 1, 2], np.array([0, 1, 2])])
@pytest.mark.parametrize(
    "func, array, by, expected",
    [
        ("sum", np.ones((12,)), labels, [3, 4, 5]),  # form 1
        ("sum", np.ones((12,)), nan_labels, [1, 4, 2]),  # form 1
        ("sum", np.ones((2, 12)), labels, [[3, 4, 5], [3, 4, 5]]),  # form 3
        ("sum", np.ones((2, 12)), nan_labels, [[1, 4, 2], [1, 4, 2]]),  # form 3
        ("sum", np.ones((2, 12)), np.array([labels, labels]), [6, 8, 10]),  # form 1 after reshape
        ("sum", np.ones((2, 12)), np.array([nan_labels, nan_labels]), [2, 8, 4]),
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
        ("count", np.ones((12,)), labels, [3, 4, 5]),  # form 1
        ("count", np.ones((12,)), nan_labels, [1, 4, 2]),  # form 1
        ("count", np.ones((2, 12)), labels, [[3, 4, 5], [3, 4, 5]]),  # form 3
        ("count", np.ones((2, 12)), nan_labels, [[1, 4, 2], [1, 4, 2]]),  # form 3
        ("count", np.ones((2, 12)), np.array([labels, labels]), [6, 8, 10]),  # form 1 after reshape
        ("count", np.ones((2, 12)), np.array([nan_labels, nan_labels]), [2, 8, 4]),
        ("nanmean", np.ones((12,)), labels, [1, 1, 1]),  # form 1
        ("nanmean", np.ones((12,)), nan_labels, [1, 1, 1]),  # form 1
        ("nanmean", np.ones((2, 12)), labels, [[1, 1, 1], [1, 1, 1]]),  # form 3
        ("nanmean", np.ones((2, 12)), nan_labels, [[1, 1, 1], [1, 1, 1]]),  # form 3
        ("nanmean", np.ones((2, 12)), np.array([labels, labels]), [1, 1, 1]),
        ("nanmean", np.ones((2, 12)), np.array([nan_labels, nan_labels]), [1, 1, 1]),
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
    ],
)
def test_groupby_reduce(
    engine: T_Engine,
    func: T_Agg,
    array: np.ndarray,
    by: np.ndarray,
    expected: list[float],
    expected_groups: T_ExpectedGroupsOpt,
    chunk: bool,
    dtype: np.typing.DTypeLike,
) -> None:
    array = array.astype(dtype)
    if chunk:
        if not has_dask or expected_groups is None:
            pytest.skip()
        array = da.from_array(array, chunks=(3,) if array.ndim == 1 else (1, 3))
        by = da.from_array(by, chunks=(3,) if by.ndim == 1 else (1, 3))

    if func == "mean" or func == "nanmean":
        expected_result = np.array(expected, dtype=np.float64)
    elif func == "sum":
        expected_result = np.array(expected, dtype=dtype)
    elif func == "count":
        expected_result = np.array(expected, dtype=np.int64)

    result, groups, = groupby_reduce(
        array,
        by,
        func=func,
        expected_groups=expected_groups,
        fill_value=123,
        engine=engine,
    )
    # we use pd.Index(expected_groups).to_numpy() which is always int64
    # for the values in this tests
    g_dtype = by.dtype if expected_groups is None else np.int64

    assert_equal(groups, np.array([0, 1, 2], g_dtype))
    assert_equal(expected_result, result)


def gen_array_by(
    size: tuple[int, ...], func: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    by = np.ones(size[-1])
    rng = np.random.default_rng(12345)
    array: np.ndarray[Any, Any] = rng.random(size)
    if "nan" in func and "nanarg" not in func:
        array[[1, 4, 5], ...] = np.nan
    elif "nanarg" in func and len(size) > 1:
        array[[1, 4, 5], 1] = np.nan
    if func in ["any", "all"]:
        array = array > 0.5
    return array, by


@pytest.mark.parametrize("chunks", [None, -1, 3, 4])
@pytest.mark.parametrize("nby", [1, 2, 3])
@pytest.mark.parametrize("size", ((12,), (12, 9)))
@pytest.mark.parametrize("add_nan_by", [True, False])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_groupby_reduce_all(
    nby, size: tuple[int, ...], chunks, func: str, add_nan_by: bool, engine: T_Engine
) -> None:
    if chunks is not None and not has_dask:
        pytest.skip()
    if "arg" in func and engine == "flox":
        pytest.skip()

    array, by = gen_array_by(size, func)
    if chunks:
        array = dask.array.from_array(array, chunks=chunks)
    bys = [by] * nby
    bys = [b + idx for idx, b in enumerate(bys)]
    if add_nan_by:
        for idx in range(nby):
            bys[idx][2 * idx : 2 * idx + 3] = np.nan
    nanmask = reduce(np.logical_or, (np.isnan(b) for b in bys))

    finalize_kwargs: list[T_Kwargs] = [{}]
    if "var" in func or "std" in func:
        finalize_kwargs = finalize_kwargs + [{"ddof": 1}, {"ddof": 0}]
        fill_value = np.nan
        tolerance = {"rtol": 1e-14, "atol": 1e-16}
    else:
        fill_value = None
        tolerance = None

    for kwargs in finalize_kwargs:
        flox_kwargs: T_Kwargs = dict(
            func=func, engine=engine, finalize_kwargs=kwargs, fill_value=fill_value
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            if "arg" in func and add_nan_by:
                array[..., nanmask] = np.nan
                expected = getattr(np, "nan" + func)(array, axis=-1, **kwargs)
            else:
                expected = getattr(np, func)(array[..., ~nanmask], axis=-1, **kwargs)
        for _ in range(nby):
            expected = np.expand_dims(expected, -1)

        actual, *groups = groupby_reduce(array, *bys, **flox_kwargs)
        assert actual.ndim == (array.ndim + nby - 1)
        assert expected.ndim == (array.ndim + nby - 1)
        expected_groups = tuple(np.array([idx + 1.0]) for idx in range(nby))
        for actual_group, expect in zip(groups, expected_groups):
            assert_equal(actual_group, expect)
        if "arg" in func:
            assert actual.dtype.kind == "i"
        assert_equal(actual, expected, tolerance)

        if not has_dask or chunks is None:
            continue

        params = list(itertools.product(["map-reduce"], [True, False, None]))
        params.extend(itertools.product(["cohorts"], [False, None]))
        for method, reindex in params:
            call = partial(
                groupby_reduce, array, *bys, method=method, reindex=reindex, **flox_kwargs
            )
            if "arg" in func and reindex is True:
                # simple_combine with argreductions not supported right now
                with pytest.raises(NotImplementedError):
                    call()
                continue
            actual, *groups = call()
            if "arg" not in func:
                # make sure we use simple combine
                assert any("simple-combine" in key for key in actual.dask.layers.keys())
            else:
                assert any("grouped-combine" in key for key in actual.dask.layers.keys())
            for actual_group, expect in zip(groups, expected_groups):
                assert_equal(actual_group, expect, tolerance)
            if "arg" in func:
                assert actual.dtype.kind == "i"
            assert_equal(actual, expected, tolerance)


@requires_dask
@pytest.mark.parametrize("size", ((12,), (12, 5)))
@pytest.mark.parametrize("func", ("argmax", "nanargmax", "argmin", "nanargmin"))
def test_arg_reduction_dtype_is_int(size: tuple[int, ...], func: str) -> None:
    """avoid bugs being hidden by the xfail in the above test."""

    rng = np.random.default_rng(12345)
    array = rng.random(size)
    by = np.ones(size[-1])

    if "nanarg" in func and len(size) > 1:
        array[[1, 4, 5], 1] = np.nan

    expected = getattr(np, func)(array, axis=-1)
    expected = np.expand_dims(expected, -1)

    actual, _ = groupby_reduce(array, by, func=func, engine="numpy")
    assert actual.dtype.kind == "i"

    actual, _ = groupby_reduce(da.from_array(array, chunks=3), by, func=func, engine="numpy")
    assert actual.dtype.kind == "i"


def test_groupby_reduce_count() -> None:
    array = np.array([0, 0, np.nan, np.nan, np.nan, 1, 1])
    labels = np.array(["a", "b", "b", "b", "c", "c", "c"])
    result, _ = groupby_reduce(array, labels, func="count")
    assert_equal(result, np.array([1, 1, 2], dtype=np.int64))


def test_func_is_aggregation() -> None:
    from flox.aggregations import mean

    array = np.array([0, 0, np.nan, np.nan, np.nan, 1, 1])
    labels = np.array(["a", "b", "b", "b", "c", "c", "c"])
    expected, _ = groupby_reduce(array, labels, func="mean")
    actual, _ = groupby_reduce(array, labels, func=mean)
    assert_equal(actual, expected)


@requires_dask
@pytest.mark.parametrize("func", ("sum", "prod"))
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_groupby_reduce_preserves_dtype(dtype, func: str) -> None:
    array = np.ones((2, 12), dtype=dtype)
    by = np.array([labels] * 2)
    result, _ = groupby_reduce(from_array(array, chunks=(-1, 4)), by, func=func)
    assert result.dtype == array.dtype


def test_numpy_reduce_nd_md() -> None:
    array = np.ones((2, 12))
    by = np.array([labels] * 2)

    expected = aggregate(by.ravel(), array.ravel(), func="sum")
    result, groups = groupby_reduce(array, by, func="sum", fill_value=123)
    actual = reindex_(result, groups, pd.Index(np.unique(by)), axis=0, fill_value=0)
    np.testing.assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    by = np.array([labels] * 2)

    expected = aggregate(by.ravel(), array.reshape(4, 24), func="sum", axis=-1, fill_value=0)
    result, groups = groupby_reduce(array, by, func="sum")
    actual = reindex_(result, groups, pd.Index(np.unique(by)), axis=-1, fill_value=0)
    assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    by = np.broadcast_to(np.array([labels] * 2), array.shape)
    expected = aggregate(by.ravel(), array.ravel(), func="sum", axis=-1)
    result, groups = groupby_reduce(array, by, func="sum")
    actual = reindex_(result, groups, pd.Index(np.unique(by)), axis=-1, fill_value=0)
    assert_equal(expected, actual)

    array = np.ones((2, 3, 4))
    by = np.ones((2, 3, 4))

    actual, _ = groupby_reduce(array, by, axis=(1, 2), func="sum")
    expected = np.sum(array, axis=(1, 2), keepdims=True).squeeze(2)
    assert_equal(actual, expected)


@requires_dask
@pytest.mark.parametrize("reindex", [None, False, True])
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("add_nan", [False, True])
@pytest.mark.parametrize("dtype", (float,))
@pytest.mark.parametrize(
    "shape, array_chunks, group_chunks",
    [
        ((12,), (3,), 3),  # form 1
        ((12,), (3,), (4,)),  # form 1, chunks not aligned
        ((12,), ((3, 5, 4),), (2,)),  # form 1
        ((10, 12), (3, 3), -1),  # form 3
        ((10, 12), (3, 3), 3),  # form 3
    ],
)
def test_groupby_agg_dask(
    func: str,
    shape: tuple[int, ...],
    array_chunks: tuple[int, ...],
    group_chunks,
    add_nan: bool,
    dtype,
    engine: T_Engine,
    reindex: bool | None,
) -> None:
    """Tests groupby_reduce with dask arrays against groupby_reduce with numpy arrays"""

    rng = np.random.default_rng(12345)
    array = dask.array.from_array(rng.random(shape), chunks=array_chunks).astype(dtype)
    array = dask.array.ones(shape, chunks=array_chunks)

    if func in ["first", "last"]:
        pytest.skip()

    if "arg" in func and (engine == "flox" or reindex):
        pytest.skip()

    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
    if add_nan:
        labels = labels.astype(float)
        labels[:3] = np.nan  # entire block is NaN when group_chunks=3
        labels[-2:] = np.nan

    kwargs: T_Kwargs = dict(
        func=func, expected_groups=[0, 1, 2], fill_value=False if func in ["all", "any"] else 123
    )

    expected, _ = groupby_reduce(array.compute(), labels, engine="numpy", **kwargs)
    actual, _ = groupby_reduce(array.compute(), labels, engine=engine, **kwargs)
    assert_equal(actual, expected)

    with raise_if_dask_computes():
        actual, _ = groupby_reduce(array, labels, engine=engine, **kwargs)
    assert_equal(actual, expected)

    by = from_array(labels, group_chunks)
    with raise_if_dask_computes():
        actual, _ = groupby_reduce(array, by, engine=engine, **kwargs)
    assert_equal(expected, actual)

    kwargs["expected_groups"] = [0, 2, 1]
    with raise_if_dask_computes():
        actual, groups = groupby_reduce(array, by, engine=engine, **kwargs, sort=False)
    assert_equal(groups, np.array([0, 2, 1], dtype=np.intp))
    assert_equal(expected, actual[..., [0, 2, 1]])

    with raise_if_dask_computes():
        actual, groups = groupby_reduce(array, by, engine=engine, **kwargs, sort=True)
    assert_equal(groups, np.array([0, 1, 2], np.intp))
    assert_equal(expected, actual)


def test_numpy_reduce_axis_subset(engine: T_Engine) -> None:
    # TODO: add NaNs
    by = labels2d
    array = np.ones_like(by, dtype=np.int64)
    kwargs: T_Kwargs = dict(func="count", engine=engine, fill_value=0)
    result, _ = groupby_reduce(array, by, **kwargs, axis=1)
    assert_equal(result, np.array([[2, 3], [2, 3]], dtype=np.int64))

    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(by)
    result, _ = groupby_reduce(array, by, **kwargs, axis=1)
    subarr = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]], dtype=np.int64)
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, **kwargs, axis=2)
    subarr = np.array([[2, 3], [2, 3]], dtype=np.int64)
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, **kwargs, axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]], dtype=np.int64)
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, **kwargs, axis=(2, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by[0, ...], **kwargs, axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]], dtype=np.int64)
    assert_equal(result, expected)


@requires_dask
def test_dask_reduce_axis_subset() -> None:

    by = labels2d
    array = np.ones_like(by, dtype=np.int64)
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(2, 3)),
            da.from_array(by, chunks=(2, 2)),
            func="count",
            axis=1,
            expected_groups=[0, 2],
        )
    assert_equal(result, np.array([[2, 3], [2, 3]], dtype=np.int64))

    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(by)
    subarr = np.array([[1, 1], [1, 1], [123, 2], [1, 1], [1, 1]], dtype=np.int64)
    expected = np.tile(subarr, (3, 1, 1))
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(by, chunks=(2, 2, 2)),
            func="count",
            axis=1,
            expected_groups=[0, 2],
            fill_value=123,
        )
    assert_equal(result, expected)

    subarr = np.array([[2, 3], [2, 3]], dtype=np.int64)
    expected = np.tile(subarr, (3, 1, 1))
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(by, chunks=(2, 2, 2)),
            func="count",
            axis=2,
            expected_groups=[0, 2],
        )
    assert_equal(result, expected)

    with pytest.raises(NotImplementedError):
        groupby_reduce(
            da.from_array(array, chunks=(1, 3, 2)),
            da.from_array(by, chunks=(2, 2, 2)),
            func="count",
            axis=2,
        )


@requires_dask
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize(
    "axis", [None, (0, 1, 2), (0, 1), (0, 2), (1, 2), 0, 1, 2, (0,), (1,), (2,)]
)
def test_groupby_reduce_axis_subset_against_numpy(func: str, axis, engine: T_Engine) -> None:
    if "arg" in func and engine == "flox":
        pytest.skip()

    if not isinstance(axis, int) and "arg" in func and (axis is None or len(axis) > 1):
        pytest.skip()
    fill_value: bool | float
    if func in ["all", "any"]:
        fill_value = False
    else:
        fill_value = 123

    if "var" in func or "std" in func:
        tolerance = {"rtol": 1e-14, "atol": 1e-16}
    else:
        tolerance = None
    # tests against the numpy output to make sure dask compute matches
    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    rng = np.random.default_rng(12345)
    array = rng.random(by.shape)
    kwargs = dict(
        func=func, axis=axis, expected_groups=[0, 2], fill_value=fill_value, engine=engine
    )
    with raise_if_dask_computes():
        actual, _ = groupby_reduce(
            da.from_array(array, chunks=(-1, 2, 3)),
            da.from_array(by, chunks=(-1, 2, 2)),
            **kwargs,
        )
    expected, _ = groupby_reduce(array, by, **kwargs)
    if engine == "flox":
        kwargs.pop("engine")
        expected_npg, _ = groupby_reduce(array, by, **kwargs, engine="numpy")
        assert_equal(expected_npg, expected)
    assert_equal(actual, expected, tolerance)


@pytest.mark.parametrize("reindex,chunks", [(None, None), (False, (2, 2, 3)), (True, (2, 2, 3))])
@pytest.mark.parametrize(
    "axis, groups, expected_shape",
    [
        (2, [0, 1, 2], (3, 5, 3)),
        (None, [0, 1, 2], (3,)),  # global reduction; 0 shaped group axis
        (None, [0], (1,)),  # global reduction; 0 shaped group axis; 1 group
    ],
)
def test_groupby_reduce_nans(
    reindex: bool | None, chunks, axis, groups, expected_shape: tuple[int, ...], engine: T_Engine
) -> None:
    def _maybe_chunk(arr):
        if chunks:
            if not has_dask:
                pytest.skip()
            return da.from_array(arr, chunks=chunks)
        else:
            return arr

    # test when entire by  are NaNs
    by = np.full((3, 5, 2), fill_value=np.nan)
    array = np.ones_like(by)

    # along an axis; requires expected_group
    # TODO: this should check for fill_value
    result, _ = groupby_reduce(
        _maybe_chunk(array),
        _maybe_chunk(by),
        func="count",
        expected_groups=groups,
        axis=axis,
        fill_value=0,
        engine=engine,
        reindex=reindex,
    )
    assert_equal(result, np.zeros(expected_shape, dtype=np.intp))

    # now when subsets are NaN
    # labels = np.array([0, 0, 1, 1, 1], dtype=float)
    # labels2d = np.array([labels[:5], np.flip(labels[:5])])
    # labels2d[0, :5] = np.nan
    # labels2d[1, 5:] = np.nan
    # by = np.broadcast_to(labels2d, (3, *labels2d.shape))


@requires_dask
@pytest.mark.parametrize(
    "expected_groups, reindex", [(None, None), ([0, 1, 2], True), ([0, 1, 2], False)]
)
def test_groupby_all_nan_blocks(expected_groups, reindex: bool | None, engine: T_Engine) -> None:
    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
    nan_labels = labels.astype(float)  # copy
    nan_labels[:5] = np.nan

    array, by, expected = (
        np.ones((2, 12), dtype=np.int64),
        np.array([nan_labels, nan_labels[::-1]]),
        np.array([2, 8, 4], dtype=np.int64),
    )

    actual, _ = groupby_reduce(
        da.from_array(array, chunks=(1, 3)),
        da.from_array(by, chunks=(1, 3)),
        func="sum",
        expected_groups=expected_groups,
        engine=engine,
        reindex=reindex,
        method="map-reduce",
    )
    assert_equal(actual, expected)


@pytest.mark.parametrize("axis", (0, 1, 2, -1))
def test_reindex(axis: int) -> None:
    shape = [2, 2, 2]
    fill_value = 0

    array = np.broadcast_to(np.array([1, 2]), shape)
    groups = np.array(["a", "b"])
    expected_groups = pd.Index(["a", "b", "c"])
    actual = reindex_(array, groups, expected_groups, fill_value=fill_value, axis=axis)

    if axis < 0:
        axis = array.ndim + axis
    result_shape = tuple(len(expected_groups) if ax == axis else s for ax, s in enumerate(shape))
    slicer = tuple(slice(None, s) for s in shape)
    expected = np.full(result_shape, fill_value)
    expected[slicer] = array

    assert_equal(actual, expected)


@pytest.mark.xfail
def test_bad_npg_behaviour() -> None:
    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=int)
    # fmt: off
    array = np.array([[1] * 12, [1] * 12])
    # fmt: on
    assert_equal(aggregate(labels, array, axis=-1, func="argmax"), np.array([[0, 5, 2], [0, 5, 2]]))

    assert (
        aggregate(
            np.array([0, 1, 2, 0, 1, 2]), np.array([-np.inf, 0, 0, -np.inf, 0, 0]), func="max"
        )[0]
        == -np.inf
    )


@pytest.mark.xfail
@pytest.mark.parametrize("func", ("nanargmax", "nanargmin"))
def test_npg_nanarg_bug(func: str) -> None:
    array = np.array([1, 1, 2, 1, 1, np.nan, 6, 1])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1]) - 1

    actual = aggregate(labels, array, func=func).astype(int)
    expected = getattr(np, func)(array)
    assert_equal(actual, expected)


@pytest.mark.parametrize("method", ["cohorts", "map-reduce"])
@pytest.mark.parametrize("chunk_labels", [False, True])
@pytest.mark.parametrize("chunks", ((), (1,), (2,)))
def test_groupby_bins(
    chunk_labels: bool, chunks: tuple[int, ...], engine: T_Engine, method: T_Method
) -> None:
    array = [1, 1, 1, 1, 1, 1]
    labels = [0.2, 1.5, 1.9, 2, 3, 20]

    if method in ["split-reduce", "cohorts"] and chunk_labels:
        pytest.xfail()

    if chunks:
        if not has_dask:
            pytest.skip()
        array = dask.array.from_array(array, chunks=chunks)
        if chunk_labels:
            labels = dask.array.from_array(labels, chunks=chunks)

    with raise_if_dask_computes():
        actual, groups = groupby_reduce(
            array,
            labels,
            func="count",
            expected_groups=np.array([1, 2, 4, 5]),
            isbin=True,
            fill_value=0,
            engine=engine,
            method=method,
        )
    expected = np.array([3, 1, 0], dtype=np.int64)
    for left, right in zip(groups, pd.IntervalIndex.from_arrays([1, 2, 4], [2, 4, 5]).to_numpy()):
        assert left == right
    assert_equal(actual, expected)


@pytest.mark.parametrize(
    "inchunks, expected",
    [
        [(1,) * 10, (3, 2, 2, 3)],
        [(2,) * 5, (3, 2, 2, 3)],
        [(3, 3, 3, 1), (3, 2, 5)],
        [(3, 1, 1, 2, 1, 1, 1), (3, 2, 2, 3)],
        [(3, 2, 2, 3), (3, 2, 2, 3)],
        [(4, 4, 2), (3, 4, 3)],
        [(5, 5), (5, 5)],
        [(6, 4), (5, 5)],
        [(7, 3), (7, 3)],
        [(8, 2), (7, 3)],
        [(9, 1), (10,)],
        [(10,), (10,)],
    ],
)
def test_rechunk_for_blockwise(inchunks: tuple[int, ...], expected: tuple[int, ...]) -> None:
    labels = np.array([1, 1, 1, 2, 2, 3, 3, 5, 5, 5])
    assert _get_optimal_chunks_for_groups(inchunks, labels) == expected


@requires_dask
@pytest.mark.parametrize(
    "expected, labels, chunks, merge",
    [
        [[[1, 2, 3, 4]], [1, 2, 3, 1, 2, 3, 4], (3, 4), True],
        [[[1, 2, 3], [4]], [1, 2, 3, 1, 2, 3, 4], (3, 4), False],
        [[[1], [2], [3], [4]], [1, 2, 3, 1, 2, 3, 4], (2, 2, 2, 1), False],
        [[[1], [2], [3], [4]], [1, 2, 3, 1, 2, 3, 4], (2, 2, 2, 1), True],
        [[[1, 2, 3], [4]], [1, 2, 3, 1, 2, 3, 4], (3, 3, 1), True],
        [[[1, 2, 3], [4]], [1, 2, 3, 1, 2, 3, 4], (3, 3, 1), False],
        [
            [[0], [1, 2, 3, 4], [5]],
            np.repeat(np.arange(6), [4, 4, 12, 2, 3, 4]),
            (4, 8, 4, 9, 4),
            True,
        ],
    ],
)
def test_find_group_cohorts(expected, labels, chunks: tuple[int, ...], merge: bool) -> None:
    actual = list(find_group_cohorts(labels, (chunks,), merge).values())
    assert actual == expected, (actual, expected)


@requires_dask
@pytest.mark.parametrize(
    "chunk_at,expected",
    [
        [1, ((1, 6, 1, 6, 1, 6, 1, 6, 1, 1),)],
        [0, ((7, 7, 7, 7, 2),)],
        [3, ((3, 4, 3, 4, 3, 4, 3, 4, 2),)],
    ],
)
def test_rechunk_for_cohorts(chunk_at: int, expected) -> None:
    array = dask.array.ones((30,), chunks=7)
    labels = np.arange(0, 30) % 7
    rechunked = rechunk_for_cohorts(array, axis=-1, force_new_chunk_at=chunk_at, labels=labels)
    assert rechunked.chunks == expected


@pytest.mark.parametrize("chunks", [None, 3])
@pytest.mark.parametrize("fill_value", [123, np.nan])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_fill_value_behaviour(func: str, chunks, fill_value: float, engine: T_Engine) -> None:
    # fill_value = np.nan tests promotion of int counts to float
    # This is used by xarray
    if func in ["all", "any"] or "arg" in func:
        pytest.skip()
    if chunks is not None and not has_dask:
        pytest.skip()

    if func == "count":

        def npfunc(x):
            x = np.asarray(x)
            return (~np.isnan(x)).sum()

    else:
        npfunc = getattr(np, func)

    by = np.array([1, 2, 3, 1, 2, 3])
    array = np.array([np.nan, 1, 1, np.nan, 1, 1])
    if chunks:
        array = dask.array.from_array(array, chunks)
    actual, _ = groupby_reduce(
        array, by, func=func, engine=engine, fill_value=fill_value, expected_groups=[0, 1, 2, 3]
    )
    expected = np.array([fill_value, fill_value, npfunc([1.0, 1.0]), npfunc([1.0, 1.0])])
    assert_equal(actual, expected)


@requires_dask
@pytest.mark.parametrize("func", ["mean", "sum"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_dtype_preservation(dtype: str, func: str, engine: T_Engine) -> None:
    expected: np.typing.DTypeLike
    if func == "sum" or (func == "mean" and "float" in dtype):
        expected = np.dtype(dtype)
    elif func == "mean" and "int" in dtype:
        expected = np.float64
    array = np.ones((20,), dtype=dtype)
    by = np.ones(array.shape, dtype=int)
    actual, _ = groupby_reduce(array, by, func=func, engine=engine)
    assert actual.dtype == expected

    array = dask.array.from_array(array, chunks=(4,))
    actual, _ = groupby_reduce(array, by, func=func, engine=engine)
    assert actual.dtype == expected


@requires_dask
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "labels_dtype", [pytest.param(np.int32, marks=pytest.mark.xfail), np.int64]
)
@pytest.mark.parametrize("method", ["map-reduce", "cohorts"])
def test_cohorts_map_reduce_consistent_dtypes(method: T_Method, dtype, labels_dtype) -> None:
    repeats = np.array([4, 4, 12, 2, 3, 4], dtype=np.int32)
    labels = np.repeat(np.arange(6, dtype=labels_dtype), repeats)
    array = dask.array.from_array(labels.astype(dtype), chunks=(4, 8, 4, 9, 4))

    actual, actual_groups = groupby_reduce(array, labels, func="count", method=method)
    assert_equal(actual_groups, np.arange(6, dtype=labels.dtype))
    assert_equal(actual, repeats.astype(np.int64))

    actual, actual_groups = groupby_reduce(array, labels, func="sum", method=method)
    assert_equal(actual_groups, np.arange(6, dtype=labels.dtype))
    assert_equal(actual, np.array([0, 4, 24, 6, 12, 20], dtype))


@requires_dask
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("axis", (-1, None))
@pytest.mark.parametrize("method", ["blockwise", "cohorts", "map-reduce", "split-reduce"])
def test_cohorts_nd_by(func: str, method: T_Method, axis: int | None, engine: T_Engine) -> None:
    o = dask.array.ones((3,), chunks=-1)
    o2 = dask.array.ones((2, 3), chunks=-1)

    array = dask.array.block([[o, 2 * o], [3 * o2, 4 * o2]])
    by = array.compute().astype(np.int64)
    by[0, 1] = 30
    by[2, 1] = 40
    by[0, 4] = 31
    array = dask.array.broadcast_to(array, (2, 3) + array.shape)

    if "arg" in func and (axis is None or engine == "flox"):
        pytest.skip()

    fill_value: bool | int
    if func in ["any", "all"]:
        fill_value = False
    else:
        fill_value = -123

    if axis is not None and method != "map-reduce":
        pytest.xfail()

    kwargs: T_Kwargs = dict(
        func=func, engine=engine, method=method, axis=axis, fill_value=fill_value
    )
    actual, groups = groupby_reduce(array, by, **kwargs)
    expected, sorted_groups = groupby_reduce(array.compute(), by, **kwargs)
    assert_equal(groups, sorted_groups)
    assert_equal(actual, expected)

    actual, groups = groupby_reduce(array, by, sort=False, **kwargs)
    if method == "map-reduce":
        assert_equal(groups, np.array([1, 30, 2, 31, 3, 4, 40], dtype=np.int64))
    else:
        assert_equal(groups, np.array([1, 30, 2, 31, 3, 40, 4], dtype=np.int64))
    reindexed = reindex_(actual, groups, pd.Index(sorted_groups))
    assert_equal(reindexed, expected)


@pytest.mark.parametrize("func", ["sum", "count"])
@pytest.mark.parametrize("fill_value, expected", ((0, np.integer), (np.nan, np.floating)))
def test_dtype_promotion(
    func: str, fill_value: int, expected: np.typing.DTypeLike, engine: T_Engine
) -> None:
    array = np.array([1, 1])
    by = [0, 1]

    actual, _ = groupby_reduce(
        array, by, func=func, expected_groups=[1, 2], fill_value=fill_value, engine=engine
    )
    assert np.issubdtype(actual.dtype, expected)


@pytest.mark.parametrize("func", ["mean", "nanmean"])
def test_empty_bins(func: str, engine: T_Engine) -> None:
    array = np.ones((2, 3, 2))
    by = np.broadcast_to([0, 1], array.shape)

    actual, _ = groupby_reduce(
        array,
        by,
        func=func,
        expected_groups=[-1, 0, 1, 2],
        isbin=True,
        engine=engine,
        axis=(0, 1, 2),
    )
    expected = np.array([1.0, 1.0, np.nan])
    assert_equal(actual, expected)


def test_datetime_binning() -> None:
    time_bins = pd.date_range(start="2010-08-01", end="2010-08-15", freq="24H")
    by = pd.date_range("2010-08-01", "2010-08-15", freq="15min")

    (actual,) = _convert_expected_groups_to_index((time_bins,), isbin=(True,), sort=False)
    expected = pd.IntervalIndex.from_arrays(time_bins[:-1], time_bins[1:])
    assert_equal(actual, expected)

    ret = factorize_((by.to_numpy(),), axis=0, expected_groups=(actual,))
    group_idx = ret[0]
    expected = pd.cut(by, time_bins).codes.copy()
    expected[0] = 14  # factorize doesn't return -1 for nans
    assert_equal(group_idx, expected)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_bool_reductions(func: str, engine: T_Engine) -> None:
    if "arg" in func and engine == "flox":
        pytest.skip()
    groups = np.array([1, 1, 1])
    data = np.array([True, True, False])
    expected = np.expand_dims(getattr(np, func)(data), -1)
    actual, _ = groupby_reduce(data, groups, func=func, engine=engine)
    assert_equal(expected, actual)


@requires_dask
def test_map_reduce_blockwise_mixed() -> None:
    t = pd.date_range("2000-01-01", "2000-12-31", freq="D").to_series()
    data = t.dt.dayofyear
    actual, _ = groupby_reduce(
        dask.array.from_array(data.values, chunks=365),
        t.dt.month,
        func="mean",
        method="cohorts",
    )
    expected, _ = groupby_reduce(data, t.dt.month, func="mean")
    assert_equal(expected, actual)


@requires_dask
@pytest.mark.parametrize("method", ["split-reduce", "blockwise", "map-reduce", "cohorts"])
def test_group_by_datetime(engine: T_Engine, method: T_Method) -> None:
    kwargs: T_Kwargs = dict(
        func="mean",
        method=method,
        engine=engine,
    )
    t = pd.date_range("2000-01-01", "2000-12-31", freq="D").to_series()
    data = t.dt.dayofyear
    daskarray = dask.array.from_array(data.values, chunks=30)

    actual, _ = groupby_reduce(daskarray, t, **kwargs)
    expected = data.to_numpy().astype(float)
    assert_equal(expected, actual)

    if method == "blockwise":
        return None

    edges = pd.date_range("1999-12-31", "2000-12-31", freq="M").to_series().to_numpy()
    actual, _ = groupby_reduce(daskarray, t.to_numpy(), isbin=True, expected_groups=edges, **kwargs)
    expected = data.resample("M").mean().to_numpy()
    assert_equal(expected, actual)

    actual, _ = groupby_reduce(
        np.broadcast_to(daskarray, (2, 3, daskarray.shape[-1])),
        t.to_numpy(),
        isbin=True,
        expected_groups=edges,
        **kwargs,
    )
    expected = np.broadcast_to(expected, (2, 3, expected.shape[-1]))
    assert_equal(expected, actual)


def test_factorize_values_outside_bins() -> None:

    vals = factorize_(
        (np.arange(10).reshape(5, 2), np.arange(10).reshape(5, 2)),
        axis=(0, 1),
        expected_groups=(
            pd.IntervalIndex.from_breaks(np.arange(2, 8, 1)),
            pd.IntervalIndex.from_breaks(np.arange(2, 8, 1)),
        ),
        reindex=True,
        fastpath=True,
    )
    actual = vals[0]
    expected = np.array([[-1, -1], [-1, 0], [6, 12], [18, 24], [-1, -1]], np.int64)
    assert_equal(expected, actual)


def test_multiple_groupers() -> None:
    actual, *_ = groupby_reduce(
        np.ones((5, 2)),
        np.arange(10).reshape(5, 2),
        np.arange(10).reshape(5, 2),
        axis=(0, 1),
        expected_groups=(
            pd.IntervalIndex.from_breaks(np.arange(2, 8, 1)),
            pd.IntervalIndex.from_breaks(np.arange(2, 8, 1)),
        ),
        reindex=True,
        func="count",
    )
    expected = np.eye(5, 5, dtype=np.int64)
    assert_equal(expected, actual)


def test_factorize_reindex_sorting_strings() -> None:
    kwargs: T_Kwargs = dict(
        by=(np.array(["El-Nino", "La-Nina", "boo", "Neutral"]),),
        axis=-1,
        expected_groups=(np.array(["El-Nino", "Neutral", "foo", "La-Nina"]),),
    )

    expected = factorize_(**kwargs, reindex=True, sort=True)[0]
    assert_equal(expected, np.array([0, 1, 4, 2], dtype=np.int64))

    expected = factorize_(**kwargs, reindex=True, sort=False)[0]
    assert_equal(expected, np.array([0, 3, 4, 1], dtype=np.int64))

    expected = factorize_(**kwargs, reindex=False, sort=False)[0]
    assert_equal(expected, np.array([0, 1, 2, 3], dtype=np.int64))

    expected = factorize_(**kwargs, reindex=False, sort=True)[0]
    assert_equal(expected, np.array([0, 1, 3, 2], dtype=np.int64))


def test_factorize_reindex_sorting_ints() -> None:
    kwargs: T_Kwargs = dict(
        by=(np.array([-10, 1, 10, 2, 3, 5]),),
        axis=-1,
        expected_groups=(np.array([0, 1, 2, 3, 4, 5], np.int64),),
    )

    expected = factorize_(**kwargs, reindex=True, sort=True)[0]
    assert_equal(expected, np.array([6, 1, 6, 2, 3, 5], dtype=np.int64))

    expected = factorize_(**kwargs, reindex=True, sort=False)[0]
    assert_equal(expected, np.array([6, 1, 6, 2, 3, 5], dtype=np.int64))

    kwargs["expected_groups"] = (np.arange(5, -1, -1),)

    expected = factorize_(**kwargs, reindex=True, sort=True)[0]
    assert_equal(expected, np.array([6, 1, 6, 2, 3, 5], dtype=np.int64))

    expected = factorize_(**kwargs, reindex=True, sort=False)[0]
    assert_equal(expected, np.array([6, 4, 6, 3, 2, 0], dtype=np.int64))


@requires_dask
def test_custom_aggregation_blockwise() -> None:
    def grouped_median(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
        return aggregate(
            group_idx,
            array,
            func=np.median,
            axis=axis,
            size=size,
            fill_value=fill_value,
            dtype=dtype,
        )

    agg_median = Aggregation(
        name="median", numpy=grouped_median, fill_value=-1, chunk=None, combine=None
    )

    array = np.arange(100, dtype=np.float32).reshape(5, 20)
    by = np.ones((20,))

    actual, _ = groupby_reduce(array, by, func=agg_median, axis=-1)
    expected = np.median(array, axis=-1, keepdims=True)
    assert_equal(expected, actual)

    methods: list[T_Method] = ["map-reduce", "cohorts"]
    for method in methods:
        with pytest.raises(NotImplementedError):
            groupby_reduce(
                dask.array.from_array(array, chunks=(1, -1)),
                by,
                func=agg_median,
                axis=-1,
                method=method,
            )

    actual, _ = groupby_reduce(
        dask.array.from_array(array, chunks=(1, -1)),
        by,
        func=agg_median,
        axis=-1,
        method="blockwise",
    )
    assert_equal(expected, actual)


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype(func, dtype, engine):
    if "arg" in func or func in ["any", "all"]:
        pytest.skip()
    arr = np.ones((4, 12), dtype=dtype)
    labels = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])
    actual, _ = groupby_reduce(arr, labels, func=func, dtype=np.float64)
    assert actual.dtype == np.dtype("float64")


@requires_dask
def test_subset_blocks():
    array = dask.array.random.random((120,), chunks=(4,))

    blockid = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    subset = subset_to_blocks(array, blockid)
    assert subset.blocks.shape == (len(blockid),)


@requires_dask
@pytest.mark.parametrize(
    "flatblocks, expected",
    (
        ((0, 1, 2, 3, 4), (slice(None),)),
        ((1, 2, 3), (slice(1, 4),)),
        ((1, 3), ([1, 3],)),
        ((0, 1, 3), ([0, 1, 3],)),
    ),
)
def test_normalize_block_indexing_1d(flatblocks, expected):
    nblocks = 5
    array = dask.array.ones((nblocks,), chunks=(1,))
    expected = tuple(np.array(i) if isinstance(i, list) else i for i in expected)
    actual = _normalize_indexes(array, flatblocks, array.blocks.shape)
    assert_equal_tuple(expected, actual)


@requires_dask
@pytest.mark.parametrize(
    "flatblocks, expected",
    (
        ((0, 1, 2, 3, 4), (0, slice(None))),
        ((1, 2, 3), (0, slice(1, 4))),
        ((1, 3), (0, [1, 3])),
        ((0, 1, 3), (0, [0, 1, 3])),
        (tuple(range(10)), (slice(0, 2), slice(None))),
        ((0, 1, 3, 5, 6, 8), (slice(0, 2), [0, 1, 3])),
        ((0, 3, 4, 5, 6, 8, 24), np.ix_([0, 1, 4], [0, 1, 3, 4])),
    ),
)
def test_normalize_block_indexing_2d(flatblocks, expected):
    nblocks = 5
    ndim = 2
    array = dask.array.ones((nblocks,) * ndim, chunks=(1,) * ndim)
    expected = tuple(np.array(i) if isinstance(i, list) else i for i in expected)
    actual = _normalize_indexes(array, flatblocks, array.blocks.shape)
    assert_equal_tuple(expected, actual)


@requires_dask
def test_subset_block_passthrough():
    # full slice pass through
    array = dask.array.ones((5,), chunks=(1,))
    subset = subset_to_blocks(array, np.arange(5))
    assert subset.name == array.name

    array = dask.array.ones((5, 5), chunks=1)
    subset = subset_to_blocks(array, np.arange(25))
    assert subset.name == array.name


@requires_dask
@pytest.mark.parametrize(
    "flatblocks, expectidx",
    [
        (np.arange(10), (slice(2), slice(None))),
        (np.arange(8), (slice(2), slice(None))),
        ([0, 10], ([0, 2], slice(1))),
        ([0, 7], (slice(2), [0, 2])),
        ([0, 7, 9], (slice(2), [0, 2, 4])),
        ([0, 6, 12, 14], (slice(3), [0, 1, 2, 4])),
        ([0, 12, 14, 19], np.ix_([0, 2, 3], [0, 2, 4])),
    ],
)
def test_subset_block_2d(flatblocks, expectidx):
    array = dask.array.from_array(np.arange(25).reshape((5, 5)), chunks=1)
    subset = subset_to_blocks(array, flatblocks)
    assert len(subset.dask.layers) == 2
    assert_equal(subset, array.compute()[expectidx])


@pytest.mark.parametrize(
    "expected, reindex, func, expected_groups, by_is_dask",
    [
        # argmax only False
        [False, None, "argmax", None, False],
        # True when by is numpy but expected is None
        [True, None, "sum", None, False],
        # False when by is dask but expected is None
        [False, None, "sum", None, True],
        # if expected_groups then always True
        [True, None, "sum", [1, 2, 3], False],
        [True, None, "sum", ([1], [2]), False],
        [True, None, "sum", ([1], [2]), True],
        [True, None, "sum", ([1], None), False],
        [True, None, "sum", ([1], None), True],
    ],
)
def test_validate_reindex_map_reduce(expected, reindex, func, expected_groups, by_is_dask):
    actual = _validate_reindex(reindex, func, "map-reduce", expected_groups, by_is_dask)
    assert actual == expected


def test_validate_reindex():
    for method in ["map-reduce", "cohorts"]:
        with pytest.raises(NotImplementedError):
            _validate_reindex(True, "argmax", method, expected_groups=None, by_is_dask=False)

    for method in ["blockwise", "cohorts"]:
        with pytest.raises(ValueError):
            _validate_reindex(True, "sum", method, expected_groups=None, by_is_dask=False)

        for func in ["sum", "argmax"]:
            actual = _validate_reindex(None, func, method, expected_groups=None, by_is_dask=False)
            assert actual is False


@requires_dask
def test_1d_blockwise_sort_optimization():
    # Make sure for resampling problems sorting isn't done.
    time = pd.Series(pd.date_range("2020-09-01", "2020-12-31 23:59", freq="3H"))
    array = dask.array.ones((len(time),), chunks=(224,))

    actual, _ = groupby_reduce(array, time.dt.dayofyear.values, method="blockwise", func="count")
    assert all("getitem" not in k for k in actual.dask)

    actual, _ = groupby_reduce(
        array, time.dt.dayofyear.values[::-1], sort=True, method="blockwise", func="count"
    )
    assert any("getitem" in k for k in actual.dask.layers)

    actual, _ = groupby_reduce(
        array, time.dt.dayofyear.values[::-1], sort=False, method="blockwise", func="count"
    )
    assert all("getitem" not in k for k in actual.dask.layers)