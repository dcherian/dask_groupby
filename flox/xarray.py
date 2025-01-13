from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version

from .aggregations import Aggregation, Dim, _atleast_1d, quantile_new_dims_func
from .core import (
    _convert_expected_groups_to_index,
    _get_expected_groups,
    _validate_expected_groups,
    groupby_reduce,
)
from .core import rechunk_for_blockwise as rechunk_array_for_blockwise
from .core import rechunk_for_cohorts as rechunk_array_for_cohorts

if TYPE_CHECKING:
    from xarray.core.types import T_DataArray, T_Dataset

    from .core import T_ExpectedGroupsOpt, T_ExpectIndex, T_ExpectOpt

    Dims = str | Iterable[Hashable] | None


def _restore_dim_order(result, obj, by, no_groupby_reorder=False):
    def lookup_order(dimension):
        if dimension == by.name and by.ndim == 1:
            (dimension,) = by.dims
            if no_groupby_reorder:
                return -1e6  # some arbitrarily low value
        if dimension in obj.dims:
            axis = obj.get_axis_num(dimension)
        else:
            axis = 1e6  # some arbitrarily high value
        return axis

    new_order = sorted(result.dims, key=lookup_order)
    return result.transpose(*new_order)


def _broadcast_size_one_dims(*arrays, core_dims):
    """Broadcast by adding size-1 dimensions in the right place.

    Workaround because apply_ufunc doesn't support this yet.
    https://github.com/pydata/xarray/issues/3032#issuecomment-503337637

    Specialized to the groupby problem.
    """
    array_dims = set(core_dims[0])
    broadcasted = [arrays[0]]
    for dims, array in zip(core_dims[1:], arrays[1:]):
        assert set(dims).issubset(array_dims)
        order = [dims.index(d) for d in core_dims[0] if d in dims]
        array = array.transpose(*order)
        axis = [core_dims[0].index(d) for d in core_dims[0] if d not in dims]
        broadcasted.append(np.expand_dims(array, axis))

    return broadcasted


def xarray_reduce(
    obj: T_Dataset | T_DataArray,
    *by: T_DataArray | Hashable,
    func: str | Aggregation,
    expected_groups: T_ExpectedGroupsOpt = None,
    isbin: bool | Sequence[bool] = False,
    sort: bool = True,
    dim: Dims | ellipsis = None,
    fill_value=None,
    dtype: np.typing.DTypeLike = None,
    method: str | None = None,
    engine: str | None = None,
    keep_attrs: bool | None = True,
    skipna: bool | None = None,
    min_count: int | None = None,
    reindex: bool | None = None,
    **finalize_kwargs,
):
    """GroupBy reduce operations on xarray objects using numpy-groupies

    Parameters
    ----------
    obj : DataArray or Dataset
        Xarray object to reduce
    *by : DataArray or iterable of str or iterable of DataArray
        Variables with which to group by ``obj``
    func : {"all", "any", "count", "sum", "nansum", "mean", "nanmean", \
            "max", "nanmax", "min", "nanmin", "argmax", "nanargmax", "argmin", "nanargmin", \
            "quantile", "nanquantile", "median", "nanmedian", "mode", "nanmode", \
            "first", "nanfirst", "last", "nanlast"} or Aggregation
        Single function name or an Aggregation instance
    expected_groups : str or sequence
        expected group labels corresponding to each `by` variable
    isbin : iterable of bool
        If True, corresponding entry in ``expected_groups`` are bin edges.
        If False, the entry in ``expected_groups`` is treated as a simple label.
    sort : (optional), bool
        Whether groups should be returned in sorted order. Only applies for dask
        reductions when ``method`` is not ``"map-reduce"``. For ``"map-reduce"``, the groups
        are always sorted.
    dim : hashable
        dimension name along which to reduce. If None, reduces across all
        dimensions of `by`
    fill_value
        Value used for missing groups in the output i.e. when one of the labels
        in ``expected_groups`` is not actually present in ``by``.
    dtype : data-type, optional
        DType for the output. Can be anything accepted by ``np.dtype``.
    method : {"map-reduce", "blockwise", "cohorts"}, optional
        Strategy for reduction of dask arrays only:
          * ``"map-reduce"``:
            First apply the reduction blockwise on ``array``, then
            combine a few newighbouring blocks, apply the reduction.
            Continue until finalizing. Usually, ``func`` will need
            to be an Aggregation instance for this method to work.
            Common aggregations are implemented.
          * ``"blockwise"``:
            Only reduce using blockwise and avoid aggregating blocks
            together. Useful for resampling-style reductions where group
            members are always together. If  `by` is 1D,  `array` is automatically
            rechunked so that chunk boundaries line up with group boundaries
            i.e. each block contains all members of any group present
            in that block. For nD `by`, you must make sure that all members of a group
            are present in a single block.
          * ``"cohorts"``:
            Finds group labels that tend to occur together ("cohorts"),
            indexes out cohorts and reduces that subset using "map-reduce",
            repeat for all cohorts. This works well for many time groupings
            where the group labels repeat at regular intervals like 'hour',
            'month', dayofyear' etc. Optimize chunking ``array`` for this
            method by first rechunking using ``rechunk_for_cohorts``
            (for 1D ``by`` only).
    engine : {"flox", "numpy", "numba"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
            This is the default choice because it works for other array types.
          * ``"flox"``:
            Use an internal implementation where the data is sorted so that
            all members of a group occur sequentially, and then numpy.ufunc.reduceat
            is to used for the reduction. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
          * ``"numba"``:
            Use the implementations in ``numpy_groupies.aggregate_numba``.
          * ``"numbagg"``:
            Use the reductions supported by ``numbagg.grouped``. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
    keep_attrs : bool, optional
        Preserve attrs?
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes; other dtypes either do not
        have a sentinel missing value (int) or ``skipna=True`` has not been
        implemented (object, datetime64 or timedelta64).
    min_count : int, default: None
        The required number of valid values to perform the operation. If
        fewer than min_count non-NA values are present the result will be
        NA. Only used if skipna is set to True or defaults to True for the
        array's dtype.
    reindex : bool, optional
        Whether to "reindex" the blockwise results to `expected_groups` (possibly automatically detected).
        If True, the intermediate result of the blockwise groupby-reduction has a value for all expected groups,
        and the final result is a simple reduction of those intermediates. In nearly all cases, this is a significant
        boost in computation speed. For cases like time grouping, this may result in large intermediates relative to the
        original block size. Avoid that by using method="cohorts". By default, it is turned off for arg reductions.
    **finalize_kwargs
        kwargs passed to the finalize function, like ``ddof`` for var, std or ``q`` for quantile.

    Returns
    -------
    DataArray or Dataset
        Reduced object

    See Also
    --------
    flox.core.groupby_reduce

    Raises
    ------
    NotImplementedError
    ValueError

    Examples
    --------
    >>> import xarray as xr
    >>> from flox.xarray import xarray_reduce

    >>> # Create a group index:
    >>> labels = xr.DataArray(
    ...     [1, 2, 3, 1, 2, 3, 0, 0, 0],
    ...     dims="x",
    ...     name="label",
    ... )
    >>> # Create a DataArray to apply the group index on:
    >>> da = da = xr.ones_like(labels)
    >>> # Sum all values in da that matches the elements in the group index:
    >>> xarray_reduce(da, labels, func="sum")
    <xarray.DataArray 'label' (label: 4)> Size: 32B
    array([3, 2, 2, 2])
    Coordinates:
      * label    (label) int64 32B 0 1 2 3
    """

    if skipna is not None and isinstance(func, Aggregation):
        raise ValueError("skipna must be None when func is an Aggregation.")

    nby = len(by)
    for b in by:
        if isinstance(b, xr.DataArray) and b.name is None:
            raise ValueError("Cannot group by unnamed DataArrays.")

    # TODO: move to GroupBy._flox_reduce
    if keep_attrs is None:
        keep_attrs = True

    if isinstance(isbin, Sequence):
        isbins = isbin
    else:
        isbins = (isbin,) * nby

    expected_groups_valid = _validate_expected_groups(nby, expected_groups)

    if not sort:
        raise NotImplementedError("sort must be True for xarray_reduce")

    # eventually drop the variables we are grouping by
    maybe_drop = {b for b in by if isinstance(b, Hashable)}
    unindexed_dims = tuple(
        b
        for b, isbin_ in zip(by, isbins)
        if isinstance(b, Hashable) and not isbin_ and b in obj.dims and b not in obj.indexes
    )

    by_da = tuple(obj[g] if isinstance(g, Hashable) else g for g in by)

    grouper_dims = []
    for g in by_da:
        for d in g.dims:
            if d not in grouper_dims:
                grouper_dims.append(d)

    if isinstance(obj, xr.Dataset):
        ds = obj
    else:
        ds = obj._to_temp_dataset()

    try:
        from xarray.indexes import PandasMultiIndex
    except ImportError:
        PandasMultiIndex = tuple()  # type: ignore[assignment, misc]

    more_drop = set()
    for var in maybe_drop:
        maybe_midx = ds._indexes.get(var, None)
        if isinstance(maybe_midx, PandasMultiIndex):
            idx_coord_names = set(tuple(maybe_midx.index.names) + (maybe_midx.dim,))
            idx_other_names = idx_coord_names - set(maybe_drop)
            more_drop.update(idx_other_names)
    maybe_drop.update(more_drop)

    if dim is Ellipsis:
        if nby > 1:
            raise NotImplementedError("Multiple by are not allowed when dim is Ellipsis.")
        name_ = by_da[0].name
        if name_ in ds.dims and not isbins[0]:
            dim_tuple = tuple(d for d in obj.dims if d != name_)
        else:
            dim_tuple = tuple(obj.dims)
    elif dim is not None:
        dim_tuple = _atleast_1d(dim)
    else:
        dim_tuple = tuple(grouper_dims)

    # broadcast to make sure grouper dimensions are present in the array.
    exclude_dims = tuple(d for d in ds.dims if d not in grouper_dims and d not in dim_tuple)

    if any(d not in grouper_dims and d not in obj.dims for d in dim_tuple):
        raise ValueError(f"Cannot reduce over absent dimensions {dim}.")

    try:
        xr.align(ds, *by_da, join="exact", copy=False)
    except ValueError as e:
        raise ValueError("Object being grouped must be exactly aligned with every array in `by`.") from e

    needs_broadcast = any(
        not set(grouper_dims).issubset(set(variable.dims)) for variable in ds.data_vars.values()
    )
    if needs_broadcast:
        ds_broad = xr.broadcast(ds, *by_da, exclude=exclude_dims)[0]
    else:
        ds_broad = ds

    dims_not_in_groupers = tuple(d for d in dim_tuple if d not in grouper_dims)
    if dims_not_in_groupers == tuple(dim_tuple) and not any(isbins):
        # reducing along a dimension along which groups do not vary
        # This is really just a normal reduction.
        # This is not right when binning so we exclude.
        if isinstance(func, str) and func.startswith("nan"):
            raise ValueError(f"Specify func={func[3:]}, skipna=True instead of func={func}")
        elif isinstance(func, Aggregation):
            raise NotImplementedError(
                "func must be a string when reducing along a dimension not present in `by`"
            )
        # skipna is not supported for all reductions
        # https://github.com/pydata/xarray/issues/8819
        kwargs = {"skipna": skipna} if skipna is not None else {}
        kwargs.update(finalize_kwargs)
        result = getattr(ds_broad, func)(dim=dim_tuple, **kwargs)
        if isinstance(obj, xr.DataArray):
            return obj._from_temp_dataset(result)
        else:
            return result

    ds = ds.drop_vars([var for var in maybe_drop if var in ds.variables])

    axis = tuple(range(-len(dim_tuple), 0))

    # Set expected_groups and convert to index since we need coords, sizes
    # for output xarray objects
    expected_groups_valid_list: list[T_ExpectIndex] = []
    group_names: tuple[Any, ...] = ()
    group_sizes: dict[Any, int] = {}
    for idx, (b_, expect, isbin_) in enumerate(zip(by_da, expected_groups_valid, isbins)):
        group_name = f"{b_.name}_bins" if isbin_ or isinstance(expect, pd.IntervalIndex) else b_.name
        group_names += (group_name,)

        if isbin_ and isinstance(expect, int):
            raise NotImplementedError("flox does not support binning into an integer number of bins yet.")

        expect1: T_ExpectOpt
        if expect is None:
            if isbin_:
                raise ValueError(
                    f"Please provided bin edges for group variable {idx} "
                    f"named {group_name} in expected_groups."
                )
            expect1 = _get_expected_groups(b_.data, sort=sort)
        else:
            expect1 = expect
        expect_index = _convert_expected_groups_to_index((expect1,), (isbin_,), sort=sort)[0]

        # The if-check is for type hinting mainly, it narrows down the return
        # type of _convert_expected_groups_to_index to pure pd.Index:
        if expect_index is not None:
            expected_groups_valid_list.append(expect_index)
            group_sizes[group_name] = len(expect_index)
        else:
            # This will never be reached
            raise ValueError("expect_index cannot be None")

    def wrapper(array, *by, func, skipna, core_dims, **kwargs):
        array, *by = _broadcast_size_one_dims(array, *by, core_dims=core_dims)

        # Handle skipna here because I need to know dtype to make a good default choice.
        # We cannot handle this easily for xarray Datasets in xarray_reduce
        if skipna and func in ["all", "any", "count"]:
            raise ValueError(f"skipna cannot be truthy for {func} reductions.")

        if skipna or (skipna is None and isinstance(func, str) and array.dtype.kind in "cfO"):
            if "nan" not in func and func not in ["all", "any", "count"]:
                func = f"nan{func}"

        result, *groups = groupby_reduce(array, *by, func=func, **kwargs)

        # Transpose the new quantile dimension to the end. This is ugly.
        # but new core dimensions are expected at the end :/
        # but groupby_reduce inserts them at the beginning
        if func in ["quantile", "nanquantile"]:
            (newdim,) = quantile_new_dims_func(**finalize_kwargs)
            if not newdim.is_scalar:
                # NOTE: _restore_dim_order will move any new dims to the end anyway.
                # This transpose is simply makes it easy to specify output_core_dims
                # output dim order: (*broadcast_dims, *group_dims, quantile_dim)
                result = np.moveaxis(result, 0, -1)

        return result

    # These data variables do not have any of the core dimension,
    # take them out to prevent errors.
    # apply_ufunc can handle non-dim coordinate variables without core dimensions
    missing_dim = {}
    if isinstance(obj, xr.Dataset):
        # broadcasting means the group dim gets added to ds, so we check the original obj
        for k, v in obj.data_vars.items():
            is_missing_dim = not (any(d in v.dims for d in dim_tuple))
            if is_missing_dim:
                missing_dim[k] = v

    # dim_tuple contains dimensions we are reducing over. These need to be the last
    # core dimensions to be synchronized with axis.
    input_core_dims = [[d for d in grouper_dims if d not in dim_tuple] + list(dim_tuple)]
    input_core_dims += [list(b.dims) for b in by_da]

    newdims: tuple[Dim, ...] = (
        quantile_new_dims_func(**finalize_kwargs) if func in ["quantile", "nanquantile"] else ()
    )

    output_core_dims = [d for d in input_core_dims[0] if d not in dim_tuple]
    output_core_dims.extend(group_names)
    vector_dims = [dim.name for dim in newdims if not dim.is_scalar]
    output_core_dims.extend(vector_dims)

    output_sizes = group_sizes
    output_sizes.update({dim.name: dim.size for dim in newdims if dim.size != 0})

    actual = xr.apply_ufunc(
        wrapper,
        ds_broad.drop_vars(tuple(missing_dim)).transpose(..., *grouper_dims),
        *by_da,
        input_core_dims=input_core_dims,
        # for xarray's test_groupby_duplicate_coordinate_labels
        exclude_dims=set(dim_tuple),
        output_core_dims=[output_core_dims],
        dask="allowed",
        dask_gufunc_kwargs=dict(
            output_sizes=output_sizes,
            output_dtypes=[dtype] if dtype is not None else None,
        ),
        keep_attrs=keep_attrs,
        kwargs={
            "func": func,
            "axis": axis,
            "sort": sort,
            "fill_value": fill_value,
            "method": method,
            "min_count": min_count,
            "skipna": skipna,
            "engine": engine,
            "reindex": reindex,
            "expected_groups": tuple(expected_groups_valid_list),
            "isbin": isbins,
            "finalize_kwargs": finalize_kwargs,
            "dtype": dtype,
            "core_dims": input_core_dims,
        },
    )

    # restore non-dim coord variables without the core dimension
    # TODO: shouldn't apply_ufunc handle this?
    for var in set(ds_broad._coord_names) - set(ds_broad._indexes) - set(ds_broad.dims):
        if all(d not in ds_broad[var].dims for d in dim_tuple):
            actual[var] = ds_broad[var]

    for newdim in newdims:
        actual.coords[newdim.name] = newdim.values if newdim.is_scalar else np.array(newdim.values)

    expect3: T_ExpectIndex | np.ndarray
    for name, expect2, by_ in zip(group_names, expected_groups_valid_list, by_da):
        # Can't remove this until xarray handles IntervalIndex:
        if isinstance(expect2, pd.IntervalIndex):
            # TODO: Only place where expect3 is an ndarray, remove the type if xarray
            # starts supporting IntervalIndex.
            expect3 = expect2.to_numpy()
        else:
            expect3 = expect2
        if isinstance(actual, xr.Dataset) and name in actual:
            actual = actual.drop_vars(name)
        # When grouping by MultiIndex, expect is an pd.Index wrapping
        # an object array of tuples
        if (
            name in ds_broad.indexes
            and isinstance(ds_broad.indexes[name], pd.MultiIndex)
            and not isinstance(expect3, pd.RangeIndex)
        ):
            levelnames = ds_broad.indexes[name].names
            if isinstance(expect3, np.ndarray):
                # TODO: workaoround for IntervalIndex issue.
                raise NotImplementedError
            expect3 = pd.MultiIndex.from_tuples(expect3.values, names=levelnames)
            actual[name] = expect3
            if Version(xr.__version__) > Version("2022.03.0"):
                actual = actual.set_coords(levelnames)
        else:
            actual[name] = expect3
        if keep_attrs:
            actual[name].attrs = by_.attrs

    if unindexed_dims:
        actual = actual.drop_vars(unindexed_dims)

    if nby == 1:
        for var in actual:
            if isinstance(obj, xr.Dataset):
                template = obj[var]
            else:
                template = obj

            if actual[var].ndim > 1 + len(vector_dims):
                no_groupby_reorder = isinstance(obj, xr.Dataset)  # do not re-order dataarrays inside datasets
                actual[var] = _restore_dim_order(
                    actual[var].variable,
                    template,
                    by_da[0],
                    no_groupby_reorder=no_groupby_reorder,
                )

    if missing_dim:
        for k, v in missing_dim.items():
            missing_group_dims = {d: size for d, size in group_sizes.items() if d not in v.dims}
            # The expand_dims is for backward compat with xarray's questionable behaviour
            if missing_group_dims:
                actual[k] = v.expand_dims(missing_group_dims).variable
            else:
                actual[k] = v.variable

    if isinstance(obj, xr.DataArray):
        return obj._from_temp_dataset(actual)
    else:
        return actual


def rechunk_for_cohorts(
    obj: T_DataArray | T_Dataset,
    dim: str,
    labels: T_DataArray,
    force_new_chunk_at,
    chunksize: int | None = None,
    ignore_old_chunks: bool = False,
    debug: bool = False,
):
    """
    Rechunks array so that each new chunk contains groups that always occur together.

    Parameters
    ----------
    obj : DataArray or Dataset
        array to rechunk
    dim : str
        Dimension to rechunk
    labels : DataArray
        1D Group labels to align chunks with. This routine works
        well when ``labels`` has repeating patterns: e.g.
        ``1, 2, 3, 1, 2, 3, 4, 1, 2, 3`` though there is no requirement
        that the pattern must contain sequences.
    force_new_chunk_at : Sequence
        Labels at which we always start a new chunk. For
        the example ``labels`` array, this would be `1`.
    chunksize : int, optional
        nominal chunk size. Chunk size is exceeded when the label
        in ``force_new_chunk_at`` is less than ``chunksize//2`` elements away.
        If None, uses median chunksize along ``dim``.

    Returns
    -------
    DataArray or Dataset
        Xarray object with rechunked arrays.
    """
    return _rechunk(
        rechunk_array_for_cohorts,
        obj,
        dim,
        labels,
        force_new_chunk_at=force_new_chunk_at,
        chunksize=chunksize,
        ignore_old_chunks=ignore_old_chunks,
        debug=debug,
    )


def rechunk_for_blockwise(obj: T_DataArray | T_Dataset, dim: str, labels: T_DataArray):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    embarrassingly parallel group reductions.

    This only works when the groups are sequential
    (e.g. labels = ``[0,0,0,1,1,1,1,2,2]``).
    Such patterns occur when using ``.resample``.

    Parameters
    ----------
    obj : DataArray or Dataset
        Array to rechunk
    dim : hashable
        Name of dimension to rechunk
    labels : DataArray
        Group labels

    Returns
    -------
    DataArray or Dataset
        Xarray object with rechunked arrays.
    """
    return _rechunk(rechunk_array_for_blockwise, obj, dim, labels)


def _rechunk(func, obj, dim, labels, **kwargs):
    """Common logic for rechunking xarray objects."""
    obj = obj.copy(deep=True)

    if isinstance(obj, xr.Dataset):
        for var in obj:
            if obj[var].chunks is not None:
                obj[var] = obj[var].copy(
                    data=func(
                        obj[var].data,
                        axis=obj[var].get_axis_num(dim),
                        labels=labels.data,
                        **kwargs,
                    )
                )
    else:
        if obj.chunks is not None:
            obj = obj.copy(data=func(obj.data, axis=obj.get_axis_num(dim), labels=labels.data, **kwargs))

    return obj
