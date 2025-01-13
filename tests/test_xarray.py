import numpy as np
import pandas as pd
import pytest

# isort: off
xr = pytest.importorskip("xarray")
# isort: on

from flox.xarray import rechunk_for_blockwise, xarray_reduce

from . import (
    ALL_FUNCS,
    assert_equal,
    has_dask,
    raise_if_dask_computes,
    requires_cftime,
    requires_dask,
)

if has_dask:
    import dask

    dask.config.set(scheduler="sync")

# test against legacy xarray implementation
# avoid some compilation overhead
xr.set_options(use_flox=False, use_numbagg=False, use_bottleneck=False)
tolerance64 = {"rtol": 1e-15, "atol": 1e-18}
np.random.seed(123)


@pytest.mark.parametrize("reindex", [None, False, True])
@pytest.mark.parametrize("min_count", [None, 1, 3])
@pytest.mark.parametrize("add_nan", [True, False])
@pytest.mark.parametrize("skipna", [True, False])
def test_xarray_reduce(skipna, add_nan, min_count, engine_no_numba, reindex):
    engine = engine_no_numba
    if skipna is False and min_count is not None:
        pytest.skip()

    arr = np.ones((4, 12))

    if add_nan:
        arr[1, ...] = np.nan
        arr[[0, 2], [3, 4]] = np.nan

    labels = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])
    labels = np.array(labels)
    labels2 = np.array([1, 2, 2, 1])

    da = xr.DataArray(
        arr,
        dims=("x", "y"),
        coords={"labels2": ("x", labels2), "labels": ("y", labels)},
    ).expand_dims(z=4)

    expected = da.groupby("labels").sum(skipna=skipna, min_count=min_count)
    actual = xarray_reduce(
        da,
        "labels",
        func="sum",
        skipna=skipna,
        min_count=min_count,
        engine=engine,
        reindex=reindex,
    )
    assert_equal(expected, actual)

    da["labels2"] = da.labels2.astype(float)
    da["labels2"][0] = np.nan
    expected = da.groupby("labels2").sum(skipna=skipna, min_count=min_count)
    actual = xarray_reduce(
        da,
        "labels2",
        func="sum",
        skipna=skipna,
        min_count=min_count,
        engine=engine,
        reindex=reindex,
    )
    assert_equal(expected, actual)

    # test dimension ordering
    # actual = xarray_reduce(
    #    da.transpose("y", ...), "labels", func="sum", skipna=skipna, min_count=min_count
    # )
    # assert_equal(expected, actual)


# TODO: sort
@pytest.mark.parametrize("pass_expected_groups", [True, False])
@pytest.mark.parametrize("chunk", (pytest.param(True, marks=requires_dask), False))
def test_xarray_reduce_multiple_groupers(pass_expected_groups, chunk, engine_no_numba):
    if chunk and pass_expected_groups is False:
        pytest.skip()
    engine = engine_no_numba

    arr = np.ones((4, 12))
    labels = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])
    labels2 = np.array([1, 2, 2, 1])

    da = xr.DataArray(
        arr,
        dims=("x", "y"),
        coords={"labels2": ("x", labels2), "labels": ("y", labels)},
    ).expand_dims(z=4)

    if chunk:
        da = da.chunk({"x": 2, "z": 1})

    expected = xr.DataArray(
        [[4, 4], [8, 8], [10, 10], [2, 2]],
        dims=("labels", "labels2"),
        coords={"labels": ["a", "b", "c", "f"], "labels2": [1, 2]},
    ).expand_dims(z=4)

    kwargs = dict(func="count", engine=engine)
    if pass_expected_groups:
        kwargs["expected_groups"] = (expected.labels.data, expected.labels2.data)

    with raise_if_dask_computes():
        actual = xarray_reduce(da, da.labels, da.labels2, **kwargs)
    xr.testing.assert_identical(expected, actual)

    with raise_if_dask_computes():
        actual = xarray_reduce(da, "labels", da.labels2, **kwargs)
    xr.testing.assert_identical(expected, actual)

    with raise_if_dask_computes():
        actual = xarray_reduce(da, "labels", "labels2", **kwargs)
    xr.testing.assert_identical(expected, actual)

    if pass_expected_groups:
        kwargs["expected_groups"] = (expected.labels2.data, expected.labels.data)
    with raise_if_dask_computes():
        actual = xarray_reduce(da, "labels2", "labels", **kwargs)
    xr.testing.assert_identical(expected.transpose("z", "labels2", "labels"), actual)


@pytest.mark.parametrize("pass_expected_groups", [True, False])
@pytest.mark.parametrize("chunk", (pytest.param(True, marks=requires_dask), False))
def test_xarray_reduce_multiple_groupers_2(pass_expected_groups, chunk, engine_no_numba):
    if chunk and pass_expected_groups is False:
        pytest.skip()
    engine = engine_no_numba

    arr = np.ones((2, 12))
    labels = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])

    da = xr.DataArray(
        arr, dims=("x", "y"), coords={"labels2": ("y", labels), "labels": ("y", labels)}
    ).expand_dims(z=4)

    if chunk:
        da = da.chunk({"x": 2, "z": 1})

    expected = xr.DataArray(
        [[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]],
        dims=("labels", "labels2"),
        coords={
            "labels": ["a", "b", "c", "f"],
            "labels2": ["a", "b", "c", "f"],
        },
    ).expand_dims(z=4, x=2)

    kwargs = dict(func="count", engine=engine)
    if pass_expected_groups:
        kwargs["expected_groups"] = (expected.labels.data, expected.labels.data)

    with raise_if_dask_computes():
        actual = xarray_reduce(da, "labels", "labels2", **kwargs)
    xr.testing.assert_identical(expected, actual)

    with pytest.raises(NotImplementedError):
        xarray_reduce(da, "labels", "labels2", dim=..., **kwargs)


@requires_dask
@pytest.mark.parametrize(
    "expected_groups",
    (None, (None, None), [[1, 2], [1, 2]]),
)
def test_validate_expected_groups(expected_groups):
    da = xr.DataArray([1.0, 2.0], dims="x", coords={"labels": ("x", [1, 2]), "labels2": ("x", [1, 2])})
    with pytest.raises(ValueError):
        xarray_reduce(
            da.chunk({"x": 1}),
            "labels",
            "labels2",
            func="count",
            expected_groups=expected_groups,
        )


@requires_cftime
@pytest.mark.parametrize("func", ["first", "last", "min", "max", "count"])
def test_xarray_reduce_cftime_var(engine, func):
    times = xr.date_range("1980-09-01 00:00", "1982-09-18 00:00", freq="ME", calendar="noleap")
    ds = xr.Dataset({"var": ("time", times)}, coords={"time": np.repeat(np.arange(4), 6)})

    actual = xarray_reduce(ds, ds.time, func=func)
    expected = getattr(ds.groupby("time"), func)()
    xr.testing.assert_identical(actual, expected)


@requires_cftime
@requires_dask
def test_xarray_reduce_single_grouper(engine_no_numba):
    engine = engine_no_numba
    # DataArray
    ds = xr.Dataset(
        {
            "Tair": (
                ("time", "x", "y"),
                dask.array.ones((36, 205, 275), chunks=(9, -1, -1)),
            )
        },
        coords={"time": xr.date_range("1980-09-01 00:00", "1983-09-18 00:00", freq="ME", calendar="noleap")},
    )
    actual = xarray_reduce(ds.Tair, ds.time.dt.month, func="mean", engine=engine)
    expected = ds.Tair.groupby("time.month").mean()
    xr.testing.assert_allclose(actual, expected)

    # Ellipsis reduction
    actual = xarray_reduce(ds.Tair, ds.time.dt.month, func="mean", dim=..., engine=engine)
    expected = ds.Tair.groupby("time.month").mean(...)
    xr.testing.assert_allclose(actual, expected)

    # Dataset
    expected = ds.groupby("time.month").mean()
    actual = xarray_reduce(ds, ds.time.dt.month, func="mean", engine=engine)
    xr.testing.assert_allclose(actual, expected)

    # reduce along other dimensions
    expected = ds.groupby("time.month").mean(("x", "y"))
    actual = xarray_reduce(ds, ds.time.dt.month, dim=["x", "y"], func="mean", engine=engine)
    xr.testing.assert_allclose(actual, expected)

    # add data var with missing grouper dim
    ds["foo"] = ("bar", [1, 2, 3])
    expected = ds.groupby("time.month").mean()
    actual = xarray_reduce(ds, ds.time.dt.month, func="mean", engine=engine)
    xr.testing.assert_allclose(actual, expected)
    del ds["foo"]

    # non-dim coord with missing grouper dim
    ds.coords["foo"] = ("bar", [1, 2, 3])
    expected = ds.groupby("time.month").mean()
    actual = xarray_reduce(ds, ds.time.dt.month, func="mean", engine=engine)
    xr.testing.assert_allclose(actual, expected)
    del ds["foo"]

    # unindexed dim
    by = ds.time.dt.month.drop_vars("time")
    ds = ds.drop_vars("time")
    expected = ds.groupby(by).mean()
    actual = xarray_reduce(ds, by, func="mean")
    xr.testing.assert_allclose(actual, expected)


def test_xarray_reduce_errors():
    da = xr.DataArray(np.ones((12,)), dims="x")
    by = xr.DataArray(np.ones((12,)), dims="x")

    with pytest.raises(ValueError, match="group by unnamed"):
        xarray_reduce(da, by, func="mean")

    by.name = "by"
    with pytest.raises(ValueError, match="Cannot reduce over"):
        xarray_reduce(da, by, func="mean", dim="foo")

    if has_dask:
        with pytest.raises(ValueError, match="provide expected_groups"):
            xarray_reduce(da, by.chunk(), func="mean")


@requires_dask
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
def test_rechunk_for_blockwise(inchunks, expected):
    labels = np.array([1, 1, 1, 2, 2, 3, 3, 5, 5, 5])

    da = xr.DataArray(dask.array.ones((10,), chunks=inchunks), dims="x", name="foo")
    rechunked = rechunk_for_blockwise(da, "x", xr.DataArray(labels, dims="x"))
    assert rechunked.chunks == (expected,)

    da = xr.DataArray(dask.array.ones((5, 10), chunks=(-1, inchunks)), dims=("y", "x"), name="foo")
    rechunked = rechunk_for_blockwise(da, "x", xr.DataArray(labels, dims="x"))
    assert rechunked.chunks == ((5,), expected)
    ds = da.to_dataset()

    rechunked = rechunk_for_blockwise(ds, "x", xr.DataArray(labels, dims="x"))
    assert rechunked.foo.chunks == ((5,), expected)


# everything below this is copied from xarray's test_groupby.py
# TODO: chunk these
# TODO: dim=None, dim=Ellipsis, groupby unindexed dim


def test_groupby_duplicate_coordinate_labels(engine_no_numba):
    engine = engine_no_numba
    # fix for http://stackoverflow.com/questions/38065129
    array = xr.DataArray([1, 2, 3], [("x", [1, 1, 2])])
    expected = xr.DataArray([3, 3], [("x", [1, 2])])
    actual = xarray_reduce(array, array.x, func="sum", engine=engine)
    assert_equal(expected, actual)


def test_multi_index_groupby_sum(engine_no_numba):
    engine = engine_no_numba
    # regression test for xarray GH873
    ds = xr.Dataset(
        {"foo": (("x", "y", "z"), np.ones((3, 4, 2)))},
        {"x": ["a", "b", "c"], "y": [1, 2, 3, 4]},
    )
    expected = ds.sum("z")
    stacked = ds.stack(space=["x", "y"])
    actual = xarray_reduce(stacked, "space", dim="z", func="sum", engine=engine)
    expected_xarray = stacked.groupby("space").sum("z")
    assert_equal(expected_xarray, actual)
    assert_equal(expected, actual.unstack("space"))

    actual = xarray_reduce(stacked.foo, "space", dim="z", func="sum", engine=engine)
    assert_equal(expected.foo, actual.unstack("space"))

    ds = xr.Dataset(
        dict(a=(("z",), np.ones(10))),
        coords=dict(b=(("z"), np.arange(2).repeat(5)), c=(("z"), np.arange(5).repeat(2))),
    ).set_index(bc=["b", "c"])
    expected = ds.groupby("bc").sum()
    actual = xarray_reduce(ds, "bc", func="sum")
    assert_equal(expected, actual)


@pytest.mark.parametrize("chunks", (None, pytest.param(2, marks=requires_dask)))
def test_xarray_groupby_bins(chunks, engine_no_numba):
    engine = engine_no_numba
    array = xr.DataArray([1, 1, 1, 1, 1], dims="x")
    labels = xr.DataArray([1, 1.5, 1.9, 2, 3], dims="x", name="labels")

    if chunks:
        array = array.chunk({"x": chunks})
        labels = labels.chunk({"x": chunks})

    kwargs = dict(
        dim="x",
        func="count",
        engine=engine,
        expected_groups=np.array([1, 2, 4, 5]),
        isbin=True,
        fill_value=0,
    )
    with raise_if_dask_computes():
        actual = xarray_reduce(array, labels, **kwargs)
    expected = xr.DataArray(
        np.array([3, 1, 0]),
        dims="labels_bins",
        coords={"labels_bins": [pd.Interval(1, 2), pd.Interval(2, 4), pd.Interval(4, 5)]},
    )
    xr.testing.assert_equal(actual, expected)

    # 3D array, 2D by, single dim, with NaNs in by
    array = array.expand_dims(y=2, z=3)
    labels = labels.expand_dims(y=2).copy()
    labels.data[-1, -1] = np.nan
    with raise_if_dask_computes():
        actual = xarray_reduce(array, labels, **kwargs)
    expected = xr.DataArray(
        np.array([[[3, 1, 0]] * 3, [[3, 0, 0]] * 3]),
        dims=("y", "z", "labels_bins"),
        coords={"labels_bins": [pd.Interval(1, 2), pd.Interval(2, 4), pd.Interval(4, 5)]},
    )
    xr.testing.assert_equal(actual, expected)


@requires_dask
def test_func_is_aggregation():
    from flox.aggregations import mean

    ds = xr.Dataset(
        {
            "Tair": (
                ("time", "x", "y"),
                dask.array.ones((36, 205, 275), chunks=(9, -1, -1)),
            )
        },
        coords={"time": xr.date_range("1980-09-01 00:00", "1983-09-18 00:00", freq="ME", calendar="noleap")},
    )
    expected = xarray_reduce(ds.Tair, ds.time.dt.month, func="mean")
    actual = xarray_reduce(ds.Tair, ds.time.dt.month, func=mean)
    xr.testing.assert_allclose(actual, expected)

    with pytest.raises(ValueError):
        xarray_reduce(ds.Tair, ds.time.dt.month, func=mean, skipna=True)

    with pytest.raises(ValueError):
        xarray_reduce(ds.Tair, ds.time.dt.month, func=mean, skipna=False)


# @requires_dask
# def test_cache():
#     pytest.importorskip("cachey")

#     from flox.cache import cache

#     ds = xr.Dataset(
#         {
#             "foo": (("x", "y"), dask.array.ones((10, 20), chunks=2)),
#             "bar": (("x", "y"), dask.array.ones((10, 20), chunks=2)),
#         },
#         coords={"labels": ("y", np.repeat([1, 2], 10))},
#     )

#     cache.clear()
#     xarray_reduce(ds, "labels", func="mean", method="cohorts")
#     assert len(cache.data) == 1

#     xarray_reduce(ds, "labels", func="mean", method="blockwise")
#     assert len(cache.data) == 2


@requires_dask
@pytest.mark.parametrize("method", ["cohorts", "map-reduce"])
def test_groupby_bins_indexed_coordinate(method):
    ds = xr.Dataset(
        {
            "air": (
                ("time", "lat", "lon"),
                dask.array.random.random((125, 25, 53), chunks=(20, 5, -1)),
            )
        },
        coords={
            "time": pd.date_range("2013-01-01", "2013-02-01", freq="6h"),
            "lat": np.arange(75.0, 14.9, -2.5),
            "lon": np.arange(200.0, 331.0, 2.5),
        },
    )
    bins = [40, 50, 60, 70]
    expected = ds.groupby_bins("lat", bins=bins).mean(keep_attrs=True, dim=...)
    actual = xarray_reduce(
        ds,
        ds.lat,
        dim=ds.air.dims,
        expected_groups=([40, 50, 60, 70],),
        isbin=(True,),
        func="mean",
        method=method,
    )
    xr.testing.assert_allclose(expected, actual)

    actual = xarray_reduce(
        ds,
        ds.lat,
        dim=ds.air.dims,
        expected_groups=pd.IntervalIndex.from_breaks([40, 50, 60, 70]),
        func="mean",
        method=method,
    )
    xr.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("chunk", (True, False))
def test_mixed_grouping(chunk):
    if not has_dask and chunk:
        pytest.skip()
    # regression test for https://github.com/xarray-contrib/flox/pull/111
    sa = 10
    sb = 13
    sc = 3

    x = xr.Dataset(
        {
            "v0": xr.DataArray(
                ((np.arange(sa * sb * sc) / sa) % 1).reshape((sa, sb, sc)),
                dims=("a", "b", "c"),
            ),
            "v1": xr.DataArray((np.arange(sa * sb) % 3).reshape(sa, sb), dims=("a", "b")),
        }
    )
    if chunk:
        x["v0"] = x["v0"].chunk({"a": 5})

    r = xarray_reduce(
        x["v0"],
        x["v1"],
        x["v0"],
        expected_groups=(np.arange(6), np.linspace(0, 1, num=5)),
        isbin=[False, True],
        func="count",
        dim="b",
        fill_value=0,
    )
    assert (r.sel(v1=[3, 4, 5]) == 0).all().data


def test_alignment_error():
    da = xr.DataArray(np.arange(10), dims="x", coords={"x": np.arange(10)})
    with pytest.raises(ValueError):
        xarray_reduce(da, da.x.sel(x=slice(5)), func="count")


@pytest.mark.parametrize("add_nan", [True, False])
@pytest.mark.parametrize("dtype_out", [np.float64, "float64", np.dtype("float64")])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("chunk", (pytest.param(True, marks=requires_dask), False))
def test_dtype(add_nan, chunk, dtype, dtype_out, engine_no_numba):
    if engine_no_numba == "numbagg":
        # https://github.com/numbagg/numbagg/issues/121
        pytest.skip()
    engine = engine_no_numba
    xp = dask.array if chunk else np
    data = xp.linspace(0, 1, 48, dtype=dtype).reshape((4, 12))

    if add_nan:
        data[1, ...] = np.nan
        data[0, [0, 2]] = np.nan

    arr = xr.DataArray(
        data,
        dims=("x", "t"),
        coords={
            "labels": (
                "t",
                np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"]),
            )
        },
        name="arr",
    )
    kwargs = dict(func="mean", dtype=dtype_out, engine=engine)
    actual = xarray_reduce(arr, "labels", **kwargs)
    expected = arr.groupby("labels").mean(dtype="float64")

    assert actual.dtype == np.dtype("float64")
    assert actual.compute().dtype == np.dtype("float64")
    xr.testing.assert_allclose(expected, actual, **tolerance64)

    actual = xarray_reduce(arr.to_dataset(), "labels", **kwargs)
    expected = arr.to_dataset().groupby("labels").mean(dtype="float64")

    assert actual.arr.dtype == np.dtype("float64")
    assert actual.compute().arr.dtype == np.dtype("float64")
    xr.testing.assert_allclose(expected, actual.transpose("labels", ...), **tolerance64)


@pytest.mark.parametrize("chunk", [pytest.param(True, marks=requires_dask), False])
@pytest.mark.parametrize("use_flox", [True, False])
def test_dtype_accumulation(use_flox, chunk):
    datetimes = pd.date_range("2010-01", "2015-01", freq="6h", inclusive="left")
    samples = 10 + np.cos(2 * np.pi * 0.001 * np.arange(len(datetimes))) * 1
    samples += np.random.randn(len(datetimes))
    samples = samples.astype("float32")

    nan_indices = np.random.default_rng().integers(0, len(samples), size=5_000)
    samples[nan_indices] = np.nan

    da = xr.DataArray(samples, dims=("time",), coords=[datetimes])
    if chunk:
        da = da.chunk(time=1024)

    gb = da.groupby("time.month")

    with xr.set_options(use_flox=use_flox):
        expected = gb.reduce(np.nanmean)
        actual = gb.mean()
        xr.testing.assert_allclose(expected, actual)
        assert np.issubdtype(actual.dtype, np.float32)
        assert np.issubdtype(actual.compute().dtype, np.float32)

        expected = gb.reduce(np.nanmean, dtype="float64")
        actual = gb.mean(dtype="float64")
        assert np.issubdtype(actual.dtype, np.float64)
        assert np.issubdtype(actual.compute().dtype, np.float64)
        xr.testing.assert_allclose(expected, actual, **tolerance64)


def test_preserve_multiindex():
    """Regression test for GH issue #215"""

    vort = xr.DataArray(
        name="vort",
        data=np.random.uniform(size=(4, 2)),
        dims=["i", "face"],
        coords={"i": ("i", np.arange(4)), "face": ("face", np.arange(2))},
    )

    vort = (
        vort.coarsen(i=2)
        .construct(i=("i_region_coarse", "i_region"))
        .stack(region=["face", "i_region_coarse"])
    )

    bins = [np.linspace(0, 1, 10)]
    bin_intervals = tuple(pd.IntervalIndex.from_breaks(b) for b in bins)

    hist = xarray_reduce(
        xr.DataArray(1),  # weights
        vort,  # variables we want to bin
        func="count",  # count occurrences falling in bins
        expected_groups=bin_intervals,  # bins for each variable
        dim=["i_region"],  # broadcast dimensions
        fill_value=0,  # fill empty bins with 0 counts
    )

    assert "region" in hist.coords


def test_fill_value_xarray_behaviour():
    times = pd.date_range("2000-01-01", freq="6h", periods=10)
    ds = xr.Dataset(
        {
            "bar": (
                "time",
                [1, 2, 3, np.nan, np.nan, np.nan, 4, 5, np.nan, np.nan],
                {"meta": "data"},
            ),
            "time": times,
        }
    )

    pd.date_range("2000-01-01", freq="3h", periods=19)
    with xr.set_options(use_flox=False):
        expected = ds.resample(time="3h").sum()
    with xr.set_options(use_flox=True):
        actual = ds.resample(time="3h").sum()
    xr.testing.assert_identical(expected, actual)


def test_fill_value_xarray_binning():
    array = np.linspace(0, 10, 5 * 10, dtype=int).reshape(5, 10)

    x = np.array([0, 0, 1, 2, 2])
    y = np.arange(array.shape[1]) * 3
    u = np.linspace(0, 1, 5)

    data_array = xr.DataArray(data=array, coords={"x": x, "y": y, "u": ("x", u)}, dims=("x", "y"))
    with xr.set_options(use_flox=False):
        expected = data_array.groupby_bins("y", bins=4).mean()
    with xr.set_options(use_flox=True):
        actual = data_array.groupby_bins("y", bins=4).mean()

    xr.testing.assert_identical(expected, actual)


def test_groupby_2d_dataset():
    d = {
        "coords": {
            "bit_index": {
                "dims": ("bit_index",),
                "attrs": {"name": "bit_index"},
                "data": [0, 1],
            },
            "index": {"dims": ("index",), "data": [0, 6, 8, 10, 14]},
            "clifford": {"dims": ("index",), "attrs": {}, "data": [1, 1, 4, 10, 4]},
        },
        "dims": {"bit_index": 2, "index": 5},
        "data_vars": {
            "counts": {
                "dims": ("bit_index", "index"),
                "attrs": {
                    "name": "counts",
                },
                "data": [[18, 30, 45, 70, 38], [382, 370, 355, 330, 362]],
            }
        },
    }

    ds = xr.Dataset.from_dict(d)

    with xr.set_options(use_flox=False):
        expected = ds.groupby("clifford").mean()
    with xr.set_options(use_flox=True):
        actual = ds.groupby("clifford").mean()
    assert expected.counts.dims == actual.counts.dims  # https://github.com/pydata/xarray/issues/8292
    xr.testing.assert_identical(expected, actual)


@pytest.mark.parametrize("chunk", (pytest.param(True, marks=requires_dask), False))
def test_resampling_missing_groups(chunk):
    # Regression test for https://github.com/pydata/xarray/issues/8592
    time_coords = pd.to_datetime(["2018-06-13T03:40:36", "2018-06-13T05:50:37", "2018-06-15T03:02:34"])

    latitude_coords = [0.0]
    longitude_coords = [0.0]

    data = [[[1.0]], [[2.0]], [[3.0]]]

    da = xr.DataArray(
        data,
        coords={
            "time": time_coords,
            "latitude": latitude_coords,
            "longitude": longitude_coords,
        },
        dims=["time", "latitude", "longitude"],
    )
    if chunk:
        da = da.chunk(time=1)
    # Without chunking the dataarray, it works:
    with xr.set_options(use_flox=False):
        expected = da.resample(time="1D").mean()
    with xr.set_options(use_flox=True):
        actual = da.resample(time="1D").mean()
    xr.testing.assert_identical(expected, actual)


@pytest.mark.parametrize("q", (0.5, (0.5,), (0.5, 0.67, 0.85)))
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("chunk", [pytest.param(True, marks=requires_dask), False])
@pytest.mark.parametrize("by_ndim", [1, 2])
def test_multiple_quantiles(q, chunk, by_ndim, skipna):
    array = np.array([[1, -1, np.nan, 3, 4, 10, 5], [1, np.nan, np.nan, 3, 4, np.nan, np.nan]])
    labels = np.array([0, 0, 0, 1, 0, 1, 1])
    dims = ("y",)
    if by_ndim == 2:
        labels = np.broadcast_to(labels, (5, *labels.shape))
        array = np.broadcast_to(np.expand_dims(array, -2), (2, 5, array.shape[-1]))
        dims += ("y0",)

    if chunk:
        array = dask.array.from_array(array, chunks=(1,) + (-1,) * by_ndim)

    da = xr.DataArray(array, dims=("x", *dims))
    by = xr.DataArray(labels, dims=dims, name="by")

    actual = xarray_reduce(da, by, func="quantile", skipna=skipna, q=q, engine="flox")
    with xr.set_options(use_flox=False):
        expected = da.groupby(by).quantile(q, skipna=skipna)
    xr.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_direct_reduction(func):
    if "arg" in func or "mode" in func:
        pytest.skip()
    # regression test for https://github.com/pydata/xarray/issues/8819
    rand = np.random.choice([True, False], size=(2, 3))
    if func not in ["any", "all"]:
        rand = rand.astype(float)

    if "nan" in func:
        func = func[3:]
        kwargs = {"skipna": True}
    else:
        kwargs = {}

    if "first" not in func and "last" not in func:
        kwargs["dim"] = "y"

    if "quantile" in func:
        kwargs["q"] = 0.9

    data = xr.DataArray(rand, dims=("x", "y"), coords={"x": [10, 20], "y": [0, 1, 2]})
    with xr.set_options(use_flox=True):
        actual = xarray_reduce(data, "x", func=func, **kwargs)
    with xr.set_options(use_flox=False):
        expected = getattr(data.groupby("x", squeeze=False), func)(**kwargs)
    xr.testing.assert_identical(expected, actual)


@pytest.mark.parametrize("reduction", ["max", "min", "nanmax", "nanmin", "sum", "nansum", "prod", "nanprod"])
def test_groupby_preserve_dtype(reduction):
    # all groups are present, we should follow numpy exactly
    ds = xr.Dataset(
        {
            "test": (
                ["x", "y"],
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int16"),
            )
        },
        coords={"idx": ("x", [1, 2, 1])},
    )

    kwargs = {"engine": "numpy"}
    if "nan" in reduction:
        kwargs["skipna"] = True
    with xr.set_options(use_flox=True):
        actual = getattr(ds.groupby("idx"), reduction.removeprefix("nan"))(**kwargs).test.dtype
    expected = getattr(np, reduction)(ds.test.data, axis=0).dtype

    assert actual == expected
