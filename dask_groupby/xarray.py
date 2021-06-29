import itertools
from typing import Dict, Iterable, Sequence, Tuple, Union

import dask
import xarray as xr

from .aggregations import Aggregation, _atleast_1d
from .core import factorize_, groupby_reduce


def xarray_reduce(
    obj: Union[xr.Dataset, xr.DataArray],
    *by: Union[xr.DataArray, Iterable[str], Iterable[xr.DataArray]],
    func: Union[str, Aggregation],
    expected_groups: Dict[str, Sequence] = None,
    bins=None,
    dim=None,
    split_out=1,
    fill_value=None,
):

    by: Tuple[xr.DataArray] = tuple(obj[g] if isinstance(g, str) else g for g in by)  # type: ignore

    if len(by) > 1 and any(dask.is_dask_collection(by_) for by_ in by):
        raise ValueError("Grouping by multiple variables will call compute dask variables.")

    grouper_dims = set(itertools.chain(*tuple(g.dims for g in by)))
    obj, *by = xr.broadcast(obj, *by, exclude=set(obj.dims) - grouper_dims)
    obj = obj.transpose(..., *by[0].dims)

    if dim is None:
        dim = by[0].dims
    else:
        dim = _atleast_1d(dim)

    assert isinstance(obj, xr.DataArray)
    axis = tuple(obj.get_axis_num(d) for d in dim)

    if len(by) > 1:
        group_idx, expected_groups, group_shape, _, _, _ = factorize_(
            tuple(g.data for g in by), expected_groups, bins
        )
        to_group = xr.DataArray(group_idx, dims=dim)
    else:
        to_group = by[0]

    print(to_group)

    group_names = tuple(g.name for g in by)
    group_sizes = dict(zip(group_names, group_shape))
    indims = tuple(obj.dims)
    otherdims = tuple(d for d in indims if d not in dim)
    result_dims = otherdims + group_names

    def wrapper(*args, **kwargs):
        result, _ = groupby_reduce(*args, **kwargs)
        if len(by) > 1:
            result = result.reshape(result.shape[:-1] + group_shape)
        return result

    print(obj.dims, to_group.dims)
    actual = xr.apply_ufunc(
        wrapper,
        obj,
        to_group,
        input_core_dims=[indims, dim],
        dask="allowed",
        output_core_dims=[result_dims],
        dask_gufunc_kwargs=dict(output_sizes=group_sizes),
        kwargs={"func": func, "axis": axis, "split_out": split_out, "fill_value": fill_value},
    )

    for name, expect in zip(group_names, expected_groups):
        actual[name] = expect

    return actual


def xarray_groupby_reduce(
    groupby: xr.core.groupby.GroupBy,
    func: Union[str, Aggregation],
    split_out=1,
):
    """ Apply on an existing Xarray groupby object for convenience."""

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
        dask="allowed",
        output_core_dims=[[outdim]],
        dask_gufunc_kwargs=dict(output_sizes={outdim: len(groups)}),
        kwargs={
            "func": func,
            "axis": -1,
            "split_out": split_out,
            "expected_groups": groups,
        },
    )
    actual[outdim] = groups

    return actual
