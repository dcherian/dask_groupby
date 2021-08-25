import itertools
from typing import TYPE_CHECKING, Hashable, Iterable, Sequence, Tuple, Union

import dask
import numpy as np
import numpy_groupies as npg
import pandas as pd
import xarray as xr

from .aggregations import Aggregation, _atleast_1d
from .core import factorize_, groupby_reduce, reindex_

if TYPE_CHECKING:
    from xarray import DataArray, Dataset, GroupBy, Resample


def _get_input_core_dims(group_names, dim, ds, to_group):
    input_core_dims = [[], []]
    for g in group_names:
        if g in dim:
            continue
        if g in ds.dims:
            input_core_dims[0].extend([g])
        if g in to_group.dims:
            input_core_dims[1].extend([g])
    input_core_dims[0].extend(dim)
    input_core_dims[1].extend(dim)
    return input_core_dims


def xarray_reduce(
    obj: Union["Dataset", "DataArray"],
    *by: Union["DataArray", Iterable[str], Iterable["DataArray"]],
    func: Union[str, Aggregation],
    expected_groups=None,
    isbin: Union[bool, Sequence[bool]] = False,
    dim: Hashable = None,
    split_out: int = 1,
    fill_value=None,
    blockwise: bool = False,
    keep_attrs: bool = True,
):
    """GroupBy reduce operations on xarray objects using numpy-groupies

    Parameters
    ----------
    obj : Union["Dataset", "DataArray"]
        Xarray object to reduce
    *by : Union["DataArray", Iterable[str], Iterable["DataArray"]]
        Variables with which to group by `obj`
    func : Union[str, Aggregation]
        Reduction method
    expected_groups : Dict[str, Sequence]
        expected group labels corresponding to each `by` variable
    isbin : If True, corresponding entry in `expected_groups` are bin edges. If False, the entry in `expected_groups` is treated as a simple label.
    dim : Hashable
        dimension name along which to reduce. If None, reduces across all
        dimensions of `by`
    split_out : int
        Number of output chunks along grouped dimension in output.
    fill_value :
        Value used for missing groups in the output i.e. when one of the labels
        in `expected_groups` is not actually present in `by`
    blockwise : bool
        If True, only apply the reduction blockwise. If False (default), we
        apply blockwise and then compute a tree reduction of those results.

    Raises
    ------
    NotImplementedError
    ValueError

    Examples
    --------
    FIXME: Add docs.
    """

    # TODO: handle this _DummyGroup stuff when dispatching from xarray
    from xarray.core.groupby import _DummyGroup

    unindexed_dims = tuple(b.name for b in by if isinstance(b, _DummyGroup))
    by = tuple(b.name if isinstance(b, _DummyGroup) else b for b in by)

    for b in by:
        if isinstance(b, xr.DataArray) and b.name is None:
            raise ValueError("Cannot group by unnamed DataArrays.")

    by: Tuple["DataArray"] = tuple(obj[g] if isinstance(g, str) else g for g in by)  # type: ignore

    if len(by) > 1 and any(dask.is_dask_collection(by_) for by_ in by):
        raise ValueError("Grouping by multiple variables will call compute dask variables.")

    if isinstance(isbin, bool):
        isbin = (isbin,) * len(by)

    grouper_dims = set(itertools.chain(*tuple(g.dims for g in by)))

    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
    else:
        ds = obj

    if dim is Ellipsis:
        dim = obj.dims

    # broadcast all variables against each other along all dimensions in `by` variables
    # don't exclude `dim` because it need not be a dimension in any of the `by` variables!
    exclude_dims = set(ds.dims) - grouper_dims
    if dim is not None:
        exclude_dims -= set(dim)
    ds, *by = xr.broadcast(ds, *by, exclude=exclude_dims)

    if dim is None:
        dim = by[0].dims
    else:
        dim = _atleast_1d(dim)

    if any(d not in grouper_dims and d not in obj.dims for d in dim):
        raise ValueError(f"cannot reduce over dimensions {dim}")

    dims_not_in_groupers = tuple(d for d in dim if d not in grouper_dims)
    if dims_not_in_groupers == dim:
        # reducing along a dimension along which groups do not vary
        # This is really just a normal reduction.
        result = getattr(ds, func)(dim=dim)
        return result

    axis = tuple(range(-len(dim), 0))

    group_names = tuple(g.name for g in by)
    # ds = ds.drop_vars(tuple(g for g in group_names))

    if len(by) > 1:
        group_idx, expected_groups, group_shape, _, _, _ = factorize_(
            tuple(g.data for g in by),
            axis,
            expected_groups,
            isbin,
        )
        to_group = xr.DataArray(group_idx, dims=dim, coords={d: by[0][d] for d in by[0].indexes})
    else:
        if expected_groups is None and isinstance(by[0].data, np.ndarray):
            expected_groups = (np.unique(by[0].data),)
        if expected_groups is None:
            raise NotImplementedError(
                "Please provided expected_groups if not grouping by a numpy-backed DataArray"
            )
        if isinstance(expected_groups, np.ndarray):
            expected_groups = (expected_groups,)
        if isbin[0]:
            group_shape = (len(expected_groups) - 1,)
        else:
            group_shape = (len(expected_groups),)
        to_group = by[0]

    group_sizes = dict(zip(group_names, group_shape))

    def wrapper(*args, **kwargs):
        result, groups = groupby_reduce(*args, **kwargs)
        if len(by) > 1:
            # all groups need not be present. reindex here
            # TODO: add test
            reindexed = reindex_(
                result,
                from_=groups,
                to=np.arange(np.prod(group_shape)),
                fill_value=fill_value,
                axis=-1,
            )
            result = reindexed.reshape(result.shape[:-1] + group_shape)
        return result

    # These data variables do not have the core dimension,
    # take them out to prevent errors.
    # apply_ufunc can handle non-dim coordinate variables without core dimensions
    missing_dim = {}
    for k, v in ds.data_vars.items():
        is_missing_dim = not (all(d in v.dims for d in dim))
        if is_missing_dim:
            missing_dim[k] = v

    # TODO: do this for specific reductions only
    bad_dtypes = tuple(k for k in ds.variables if ds[k].dtype.kind in ("S", "U"))

    actual = xr.apply_ufunc(
        wrapper,
        ds.drop_vars(tuple(missing_dim) + bad_dtypes),
        to_group,
        input_core_dims=_get_input_core_dims(group_names, dim, ds, to_group),
        # for xarray's test_groupby_duplicate_coordinate_labels
        exclude_dims=set(dim),
        output_core_dims=[group_names],
        dask="allowed",
        dask_gufunc_kwargs=dict(output_sizes=group_sizes),
        keep_attrs=keep_attrs,
        kwargs={
            "func": func,
            "axis": axis,
            "split_out": split_out,
            "fill_value": fill_value,
            "blockwise": blockwise,
            # The following mess exists becuase for multiple `by`s I factorize eagerly
            # here before passing it on; this means I have to handle the
            # "binning by single by variable" case explicitly where the factorization
            # happens later allowing `by` to  be a dask variable.
            "expected_groups": expected_groups[0] if len(by) == 1 and isbin[0] else None,
            "isbin": isbin[0] if len(by) == 1 else False,
        },
    )

    for name, expect, isbin_ in zip(group_names, expected_groups, isbin):
        if isbin_:
            expect = [pd.Interval(left, right) for left, right in zip(expect[:-1], expect[1:])]
        if isinstance(actual, xr.Dataset) and name in actual:
            actual = actual.drop_vars(name)
        actual[name] = expect

    # if grouping by multi-indexed variable, then restore it
    for name, index in ds.indexes.items():
        if name in actual.indexes and isinstance(index, pd.MultiIndex):
            actual[name] = index

    if missing_dim:
        actual = actual.update(missing_dim)

    if unindexed_dims:
        actual = actual.drop_vars(unindexed_dims)

    if isinstance(obj, xr.DataArray):
        return obj._from_temp_dataset(actual)
    else:
        return actual


def xarray_groupby_reduce(
    groupby: "GroupBy",
    func: Union[str, Aggregation],
    split_out: int = 1,
    blockwise: bool = False,
    keep_attrs: bool = True,
):
    """Apply on an existing Xarray groupby object for convenience."""

    def wrapper(*args, **kwargs):
        result, _ = groupby_reduce(*args, **kwargs)
        return result

    groups = list(groupby.groups.keys())
    outdim = groupby._unique_coord.name
    groupdim = groupby._group_dim

    actual = xr.apply_ufunc(
        wrapper,
        groupby._obj,
        groupby._group,
        input_core_dims=[[groupdim], [groupdim]],
        # for xarray's test_groupby_duplicate_coordinate_labels
        exclude_dims=set(groupdim),
        output_core_dims=[[outdim]],
        dask="allowed",
        dask_gufunc_kwargs=dict(output_sizes={outdim: len(groups)}),
        keep_attrs=keep_attrs,
        kwargs={
            "func": func,
            "axis": -1,
            "split_out": split_out,
            "expected_groups": groups,
            "blockwise": blockwise,
        },
    )
    actual[outdim] = groups

    return actual


def _get_optimal_chunks_for_groups(chunks, labels):
    chunkidx = np.cumsum(chunks) - 1
    # what are the groups at chunk boundaries
    labels_at_chunk_bounds = np.unique(labels[chunkidx])
    # what's the last index of all groups
    last_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="last")
    # what's the last index of groups at the chunk boundaries.
    lastidx = last_indexes[labels_at_chunk_bounds]

    if len(chunkidx) == len(lastidx) and (chunkidx == lastidx).all():
        return chunks

    first_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="first")
    firstidx = first_indexes[labels_at_chunk_bounds]

    newchunkidx = [0]
    for c, f, l in zip(chunkidx, firstidx, lastidx):
        Δf = abs(c - f)
        Δl = abs(c - l)
        if c == 0 or newchunkidx[-1] > l:
            continue
        if Δf < Δl and f > newchunkidx[-1]:
            newchunkidx.append(f)
        else:
            newchunkidx.append(l + 1)
    if newchunkidx[-1] != chunkidx[-1] + 1:
        newchunkidx.append(chunkidx[-1] + 1)
    newchunks = np.diff(newchunkidx)

    assert sum(newchunks) == sum(chunks)
    return tuple(newchunks)


def rechunk_to_group_boundaries(obj: Union["DataArray", "Dataset"], dim: str, labels: "DataArray"):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    parallel group reductions.

    This only works when the groups are sequential (e.g. labels = [0,0,0,1,1,1,1,2,2]).
    Such patterns occur when using ``.resample``.
    """

    obj = obj.copy(deep=True)

    def _rechunk(array):
        axis = array.get_axis_num(dim)
        chunks = array.chunks[axis]
        # TODO: lru_cache this?
        newchunks = _get_optimal_chunks_for_groups(chunks, labels.data)
        if newchunks == chunks:
            return array
        else:
            return array.chunk({dim: newchunks})

    if isinstance(obj, xr.Dataset):
        for var in obj:
            if obj[var].chunks is not None:
                obj[var] = _rechunk(obj[var])
    else:
        if obj.chunks is not None:
            obj = _rechunk(obj)

    return obj


def resample_reduce(
    resampler: "Resample",
    func: Union[str, Aggregation],
    keep_attrs: bool = True,
):

    obj = resampler._obj
    dim = resampler._group_dim

    # this creates a label DataArray since resample doesn't do that somehow
    tostack = []
    for idx, slicer in enumerate(resampler._group_indices):
        if slicer.stop is None:
            stop = resampler._obj.sizes[dim]
        else:
            stop = slicer.stop
        tostack.append(idx * np.ones((stop - slicer.start,), dtype=np.int32))
    by = xr.DataArray(np.hstack(tostack), dims=(dim,), name="__resample_dim__")

    obj = rechunk_to_group_boundaries(obj, dim, by)

    result = xarray_reduce(
        obj,
        by,
        func=func,
        blockwise=True,
        expected_groups=(resampler._unique_coord.data,),
        keep_attrs=keep_attrs,
    ).rename({"__resample_dim__": dim})
    return result
