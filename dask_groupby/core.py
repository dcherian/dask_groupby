import copy
import itertools
import operator
from collections import namedtuple
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy_groupies as npg
import pandas as pd

from . import aggregations
from .aggregations import Aggregation, _atleast_1d, _count, _get_fill_value
from .xrutils import is_duck_array, is_duck_dask_array

if TYPE_CHECKING:
    import dask.array.Array as DaskArray


IntermediateDict = Dict[Union[str, Callable], Any]
FinalResultsDict = Dict[str, Union["DaskArray", np.ndarray]]


def _get_chunk_reduction(reduction_type: str) -> Callable:
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def _move_reduce_dims_to_end(arr: np.ndarray, axis: Sequence) -> np.ndarray:
    """Transpose `arr` by moving `axis` to the end."""
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    arr = arr.transpose(order)
    return arr


def _collapse_axis(arr: np.ndarray, naxis: int) -> np.ndarray:
    """Reshape so that the last `naxis` axes are collapsed to one axis."""
    newshape = arr.shape[:-naxis] + (np.prod(arr.shape[-naxis:]),)
    return arr.reshape(newshape)


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


def chunks_maximize_cohorts(labels, chunksize, force_new_chunk_at):
    """
    Finds new chunks that align well with labels.

    Parameters
    ----------
    labels: np.array
        Group labels to align chunks with. This routine works
        well when the labels are repeating sequences: e.g.
        ``1, 2, 3, 1, 2, 3, 4, 1, 2, 3``
    chunksize: int
        nominal chunk size. Chunk size is exceded when the label
        in ``force_new_chunk_at`` is less than chunksize//2 elements away.
    force_new_chunk_at:
        label at which we must always start a new chunk. For
        the example ``labels`` array, this would be `1``.

    Returns
    -------
    newchunks: tuple of int
    """

    force_new_chunk_at = _atleast_1d(force_new_chunk_at)

    isbreak = labels == force_new_chunk_at
    divisions = []
    counter = 1
    for idx, lab in enumerate(labels):
        if lab in force_new_chunk_at:
            divisions.append(idx)
            counter = 1
            continue

        next_break = np.nonzero(isbreak[idx:])[0]
        if next_break.any():
            next_break_is_close = next_break[0] <= chunksize // 2
            # print(idx, next_break_is_close)
        else:
            next_break_is_close = False

        if counter >= chunksize and not next_break_is_close:
            divisions.append(idx)
            counter = 1
            continue
        counter += 1

    divisions.append(len(labels))
    newchunks = tuple(np.diff(divisions))
    assert sum(newchunks) == len(labels)
    return newchunks


def find_group_cohorts(labels, chunks, merge=False):
    """
    Finds groups labels that occur together: "cohorts"

    Parameters
    ----------
    labels: np.ndarray
        Array of group labels
    chunks: tuple
        chunks along grouping dimension for array that is being reduced
    merge: bool, optional
        Attempt to merge cohorts when one cohort's chunks are a subset
        of another cohort's chunks.

    Returns
    -------
    cohorts: dict_values
        Iterable of cohorts
    """
    import copy

    import toolz as tlz

    which_chunk = np.repeat(np.arange(len(chunks)), chunks)
    # these are chunks where a label is present
    label_chunks = {lab: tuple(np.unique(which_chunk[labels == lab])) for lab in labels}

    # These invert the label_chunks mapping so we know which labels occur together.
    chunks_cohorts = tlz.groupby(label_chunks.get, label_chunks.keys())
    # TODO: sort by length of values (i.e. cohort);
    # then loop in reverse and merge when keys are subsets of initial keys?
    if merge:
        items = tuple(chunks_cohorts.items())

        merged_cohorts = {}
        merged_keys = []

        for idx, (k1, v1) in enumerate(items):
            if k1 in merged_keys:
                continue
            merged_cohorts[k1] = copy.deepcopy(v1)
            for k2, v2 in items[idx + 1 :]:
                if k2 in merged_keys:
                    continue
                if set(k2).issubset(set(k1)):
                    print(f"merging {k1}, {k2}")
                    merged_cohorts[k1].extend(v2)
                    merged_keys.append(k2)

        return merged_cohorts.values()
    else:
        return chunks_cohorts.values()


def rechunk_for_cohorts(array, axis, labels, chunksize, force_new_chunk_at):

    newchunks = chunks_maximize_cohorts(labels, chunksize, force_new_chunk_at)
    if newchunks == array.chunks[axis]:
        return array
    else:
        return array.rechunk({axis: newchunks})


def rechunk_for_blockwise(array, axis, labels):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    parallel group reductions.

    This only works when the groups are sequential (e.g. labels = [0,0,0,1,1,1,1,2,2]).
    Such patterns occur when using ``.resample``.
    """
    chunks = array.chunks[axis]
    # TODO: lru_cache this?
    newchunks = _get_optimal_chunks_for_groups(chunks, labels)
    if newchunks == chunks:
        return array
    else:
        return array.rechunk({axis: newchunks})


def reindex_(array: np.ndarray, from_, to, fill_value=None, axis: int = -1) -> np.ndarray:

    assert axis in (0, -1)

    from_ = np.atleast_1d(from_)
    to = np.atleast_1d(to)
    # short-circuit for trivial case
    if len(from_) == len(to) and np.all(from_ == to):
        return array

    if array.shape[axis] == 0:
        # all groups were NaN
        reindexed = np.full(array.shape[:-1] + (len(to),), fill_value, dtype=array.dtype)
        return reindexed

    from_ = np.atleast_1d(from_)
    if from_.dtype.kind == "O" and isinstance(from_[0], tuple):
        raise NotImplementedError(
            "Currently does not support reindexing with object arrays of tuples. "
            "These occur when grouping by multi-indexed variables in xarray."
        )
    idx = np.array(
        [np.argwhere(np.array(from_) == label)[0, 0] if label in from_ else -1 for label in to]
    )
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx  # type: ignore
    reindexed = array[tuple(indexer)]
    if any(idx == -1):
        if fill_value is None:
            raise ValueError("Filling is required. fill_value cannot be None.")
        if axis == 0:
            loc = (idx == -1, ...)
        else:
            loc = (..., idx == -1)
        reindexed[loc] = fill_value
    return reindexed


def offset_labels(labels: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Offset group labels by dimension. This is used when we
    reduce over a subset of the dimensions of by. It assumes that the reductions
    dimensions have been flattened in the last dimension
    Copied from xhistogram &
    https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    ngroups: int = labels.max() + 1  # type: ignore
    offset: np.ndarray = (
        labels + np.arange(np.prod(labels.shape[:-1])).reshape((*labels.shape[:-1], -1)) * ngroups
    )
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[labels == -1] = -1
    size: int = np.prod(labels.shape[:-1]) * ngroups  # type: ignore
    return offset, ngroups, size


def factorize_(by: Tuple, axis, expected_groups: Tuple = None, isbin: Tuple = None):
    if not isinstance(by, tuple):
        raise ValueError(f"Expected `by` to be a tuple. Received {type(by)} instead")

    if isbin is None:
        isbin = (False,) * len(by)
    if expected_groups is None:
        expected_groups = (None,) * len(by)

    factorized = []
    found_groups = []
    for groupvar, expect, tobin in zip(by, expected_groups, isbin):
        if tobin:
            # when binning we change expected groups to integers marking the interval
            # this makes the reindexing logic simpler.
            if expect is None:
                raise ValueError("Please pass bin_edges in expected_groups.")
            idx = np.digitize(groupvar, expect) - 1
            expect = np.arange(idx.max() + 1)
            found_groups.append(expect)
        else:
            idx, groups = pd.factorize(groupvar.ravel())
            found_groups.append(np.array(groups))
        factorized.append(idx)

    grp_shape = tuple(len(grp) for grp in found_groups)
    ngroups = np.prod(grp_shape)
    if len(by) > 1:
        group_idx = np.ravel_multi_index(factorized, grp_shape).reshape(by[0].shape)
    else:
        group_idx = factorized[0]

    if np.isscalar(axis) and groupvar.ndim > 1:
        # Not reducing along all dimensions of by
        offset_group = True
        group_idx, ngroups, size = offset_labels(group_idx.reshape(by[0].shape))
        group_idx = group_idx.ravel()
    else:
        size = None
        offset_group = False

    # numpy_groupies cannot deal with group_idx = -1
    # so we'll add use (ngroups+1) as the sentinel
    # note we cannot simply remove the NaN locations;
    # that would mess up argmax, argmin
    nan_sentinel = size + 1 if offset_group else ngroups + 1
    group_idx[group_idx == -1] = nan_sentinel

    FactorProps = namedtuple("FactorProps", "offset_group nan_sentinel")
    props = FactorProps(offset_group, nan_sentinel)
    return group_idx, found_groups, grp_shape, ngroups, size, props


def chunk_argreduce(
    array_plus_idx: Tuple[np.ndarray, ...],
    by: np.ndarray,
    func: Sequence[str],
    expected_groups: Optional[Union[Sequence, np.ndarray]],
    axis: Union[int, Sequence[int]],
    fill_value: Mapping[Union[str, Callable], Any],
    dtype=None,
    reindex: bool = False,
    isbin: bool = False,
) -> IntermediateDict:
    """
    Per-chunk arg reduction.

    Expects a tuple of (array, index along reduction axis). Inspired by
    dask.array.reductions.argtopk
    """
    array, idx = array_plus_idx

    results = chunk_reduce(
        array,
        by,
        func,
        expected_groups=None,
        axis=axis,
        fill_value=fill_value,
        isbin=isbin,
        dtype=dtype,
    )
    if not np.isnan(results["groups"]).all():
        # will not work for empty groups...
        # glorious
        newidx = np.broadcast_to(idx, array.shape)[
            np.unravel_index(results["intermediates"][1], array.shape)
        ]
        results["intermediates"][1] = newidx

    if reindex and expected_groups is not None:
        results["intermediates"][1] = reindex_(
            results["intermediates"][1], results["groups"].squeeze(), expected_groups, fill_value=0
        )

    return results


def chunk_reduce(
    array: np.ndarray,
    by: np.ndarray,
    func: Union[str, Callable, Sequence[str], Sequence[Callable]],
    expected_groups: Union[Sequence, np.ndarray] = None,
    axis: Union[int, Sequence[int]] = None,
    fill_value: Mapping[Union[str, Callable], Any] = None,
    dtype=None,
    reindex: bool = False,
    isbin: bool = False,
) -> IntermediateDict:
    """
    Wrapper for numpy_groupies aggregate that supports nD ``array`` and
    mD ``by``.

    Core groupby reduction using numpy_groupies. Uses ``pandas.factorize`` to factorize
    ``by``. Offsets the groups if not reducing along all dimensions of ``by``.
    Always ravels ``by`` to 1D, flattens appropriate dimensions of array.

    When dask arrays are passed to groupby_reduce, this function is called on every
    block.

    Parameters
    ----------
    array: numpy.ndarray
        Array of values to reduced
    by: numpy.ndarray
        Array to group by.
    func: str or Callable or Sequence[str] or Sequence[Callable]
        Name of reduction or function, passed to numpy_groupies.
        Supports multiple reductions.
    axis: (optional) int or Sequence[int]
        If None, reduce along all dimensions of array.
        Else reduce along specified axes.

    Returns
    -------
    dict
    """

    if isinstance(func, str) or callable(func):
        func = (func,)  # type: ignore

    func: Union[Sequence[str], Sequence[Callable]]

    nax = len(axis) if isinstance(axis, Sequence) else by.ndim
    final_array_shape = array.shape[:-nax] + (1,) * (nax - 1)
    final_groups_shape = (1,) * (nax - 1)

    if isinstance(axis, Sequence) and len(axis) == 1:
        axis = next(iter(axis))

    if not isinstance(fill_value, Sequence):
        fill_value = (fill_value,)

    # when axis is a tuple
    # collapse and move reduction dimensions to the end
    if isinstance(axis, Sequence) and len(axis) < by.ndim:
        by = _collapse_axis(by, len(axis))
        array = _collapse_axis(array, len(axis))
        axis = -1

    if by.ndim == 1:
        # TODO: This assertion doesn't work with dask reducing across all dimensions
        # when by.ndim == array.ndim
        # the intermediates are 1D but axis=range(array.ndim)
        # assert axis in (0, -1, array.ndim - 1, None)
        axis = -1

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, groups, _, ngroups, size, props = factorize_(
        (by,), axis, expected_groups=(expected_groups,), isbin=(isbin,)
    )
    groups = groups[0]

    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - by.ndim] + (np.prod(array.shape[-by.ndim :]),)
    array = array.reshape(newshape)

    assert group_idx.ndim == 1
    mask = np.logical_not(group_idx == props.nan_sentinel)
    empty = np.all(~mask) or np.prod(by.shape) == 0

    results: IntermediateDict = {"groups": [], "intermediates": []}
    if reindex and expected_groups is not None:
        results["groups"] = np.array(expected_groups)
    else:
        if empty:
            results["groups"] = np.array([np.nan])
        else:
            sortidx = np.argsort(groups)
            results["groups"] = groups[sortidx]

    final_array_shape += results["groups"].shape
    final_groups_shape += results["groups"].shape

    for reduction, fv in zip(func, fill_value):
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fv)
        else:
            if callable(reduction):
                # passing a custom reduction for npg to apply per-group is really slow!
                # So this `reduction` has to do the groupby-aggregation
                result = reduction(
                    group_idx,
                    array,
                    size=size,
                    # important when reducing with "offset" groups
                    fill_value=fv,
                )
            else:
                result = npg.aggregate_numpy.aggregate(
                    group_idx,
                    array,
                    axis=-1,
                    func=reduction,
                    size=size,
                    # important when reducing with "offset" groups
                    fill_value=fv,
                    dtype=dtype,
                )
            if np.any(~mask):
                # remove NaN group label which should be last
                result = result[..., :-1]
            if props.offset_group:
                result = result.reshape(*final_array_shape[:-1], ngroups)
            if reindex and expected_groups is not None:
                result = reindex_(result, groups, expected_groups, fill_value=fv)
            else:
                result = result[..., sortidx]
            result = result.reshape(final_array_shape)
        results["intermediates"].append(result)
    if final_groups_shape:
        # This happens when to_group is broadcasted, and we reduce along the broadcast
        # dimension
        results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    return results


def _squeeze_results(results: IntermediateDict, axis: Sequence) -> IntermediateDict:
    # at the end we squeeze out extra dims
    groups = results["groups"]
    newresults: IntermediateDict = {"groups": [], "intermediates": []}
    newresults["groups"] = np.squeeze(
        groups, axis=tuple(ax for ax in range(groups.ndim - 1) if groups.shape[ax] == 1)
    )
    for v in results["intermediates"]:
        squeeze_ax = tuple(ax for ax in sorted(axis)[:-1] if v.shape[ax] == 1)
        newresults["intermediates"].append(np.squeeze(v, axis=squeeze_ax) if squeeze_ax else v)
    return newresults


def _split_groups(array, j, slicer):
    """Slices out chunks when split_out > 1"""
    results = {"groups": array["groups"][..., slicer]}
    results["intermediates"] = [v[..., slicer] for v in array["intermediates"]]
    return results


def _finalize_results(
    results: IntermediateDict,
    agg: Aggregation,
    axis: Sequence[int],
    expected_groups: Union[Sequence, np.ndarray, None],
    fill_value: Any,
    mask_counts: bool = True,
):
    """Finalize results by
    1. Squeezing out dummy dimensions
    2. Calling agg.finalize with intermediate results
    3. Mask using counts and fill with user-provided fill_value.
    4. reindex to expected_groups

    Parameters
    ----------

    mask_counts: bool
        Whether to mask out results using counts which is expected to be the last element in
        results["intermediates"]. Should be False when dask arrays are not involved.

    """
    squeezed = _squeeze_results(results, axis)

    if fill_value is not None and mask_counts:
        counts = squeezed["intermediates"][-1]
        squeezed["intermediates"] = squeezed["intermediates"][:-1]

    # finalize step
    result: Dict[str, Union["DaskArray", np.ndarray]] = {}
    if agg.finalize is None:
        result[agg.name] = squeezed["intermediates"][0]
    else:
        result[agg.name] = agg.finalize(*squeezed["intermediates"])

    if fill_value is not None and mask_counts:
        result[agg.name] = np.where(counts > 0, result[agg.name], fill_value)

    # Final reindexing has to be here to be lazy
    if expected_groups is not None:
        result[agg.name] = reindex_(
            result[agg.name], squeezed["groups"], expected_groups, fill_value=fill_value
        )

    # all groups need not have been present, so we don't use `squeezed["groups"]` here
    result["groups"] = expected_groups

    return result


def _npg_aggregate(
    x_chunk,
    agg: Aggregation,
    expected_groups: Union[Sequence, np.ndarray, None],
    axis: Sequence,
    keepdims,
    group_ndim: int,
    fill_value: Any = None,
) -> FinalResultsDict:
    """Final aggregation step of tree reduction"""
    results = _npg_combine(x_chunk, agg, axis, keepdims, group_ndim)
    return _finalize_results(results, agg, axis, expected_groups, fill_value)


def _npg_combine(
    x_chunk,
    agg: Aggregation,
    axis: Sequence,
    keepdims: bool,
    group_ndim: int,
) -> IntermediateDict:
    """Combine intermediates step of tree reduction."""
    from dask.array.core import _concatenate2
    from dask.base import flatten
    from dask.utils import deepmap

    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]

    unique_groups = np.unique(
        tuple(flatten(deepmap(lambda x: np.atleast_1d(x["groups"].squeeze()).tolist(), x_chunk)))
    )

    def reindex_intermediates(x):
        new_shape = x["groups"].shape[:-1] + (len(unique_groups),)
        newx = {"groups": np.broadcast_to(unique_groups, new_shape)}
        newx["intermediates"] = tuple(
            reindex_(v, from_=x["groups"].squeeze(), to=unique_groups, fill_value=f)
            for v, f in zip(x["intermediates"], agg.fill_value["intermediate"])
        )
        return newx

    def _conc2(key1, key2=None, axis=None) -> np.ndarray:
        """copied from dask.array.reductions.mean_combine"""
        if key2 is not None:
            mapped = deepmap(lambda x: x[key1][key2], x_chunk)
        else:
            mapped = deepmap(lambda x: x[key1], x_chunk)
        return _concatenate2(mapped, axes=axis)

    x_chunk = deepmap(reindex_intermediates, x_chunk)

    group_conc_axis: Iterable[int]
    if group_ndim == 1:
        group_conc_axis = (0,)
    else:
        group_conc_axis = sorted(group_ndim - ax - 1 for ax in axis)
    groups = _conc2("groups", axis=group_conc_axis)

    if agg.reduction_type == "argreduce":

        # If _count was added for masking later, we need to account for that
        if agg.chunk[-1] == _count:
            slicer = slice(None, -1)
        else:
            slicer = slice(None, None)

        # We need to send the intermediate array values & indexes at the same time
        # intermediates are (value e.g. max, index e.g. argmax, counts)
        array_idx = tuple(_conc2(key1="intermediates", key2=idx, axis=axis) for idx in (0, 1))
        results = chunk_argreduce(
            array_idx,
            groups,
            func=agg.combine[slicer],  # count gets treated specially next
            axis=axis,
            expected_groups=None,
            fill_value=agg.fill_value["intermediate"][slicer],
            dtype=agg.dtype,
        )

        if agg.chunk[-1] == _count:
            counts = _conc2(key1="intermediates", key2=2, axis=axis)
            # sum the counts
            results["intermediates"].append(
                chunk_reduce(
                    counts,
                    groups,
                    func="sum",
                    axis=axis,
                    expected_groups=None,
                    fill_value=(0,),
                    dtype=np.intp,
                )["intermediates"][0]
            )

    elif agg.reduction_type == "reduce":
        # Here we reduce the intermediates individually
        results = {"groups": None, "intermediates": []}
        for idx, (combine, fv) in enumerate(zip(agg.combine, agg.fill_value["intermediate"])):
            array = _conc2(key1="intermediates", key2=idx, axis=axis)
            if array.shape[-1] == 0:
                # all empty when combined
                results["intermediates"].append(
                    np.empty(shape=(1,) * (len(axis) - 1) + (0,), dtype=agg.dtype)
                )
                results["groups"] = np.empty(
                    shape=(1,) * (len(group_conc_axis) - 1) + (0,), dtype=groups.dtype
                )
            else:
                _results = chunk_reduce(
                    array,
                    groups,
                    func=combine,
                    axis=axis,
                    expected_groups=None,
                    fill_value=fv,
                )
                results["intermediates"].append(*_results["intermediates"])
                results["groups"] = _results["groups"]
    return results


def split_blocks(applied, split_out, expected_groups, split_name):
    import dask.array
    from dask.array.core import normalize_chunks
    from dask.highlevelgraph import HighLevelGraph

    chunk_tuples = tuple(itertools.product(*tuple(range(n) for n in applied.numblocks)))
    ngroups = len(expected_groups)
    group_chunks = normalize_chunks(np.ceil(ngroups / split_out), (ngroups,))[0]
    idx = tuple(np.cumsum((0,) + group_chunks))

    # split each block into `split_out` chunks
    dsk = {}
    for i in chunk_tuples:
        for j in range(split_out):
            dsk[(split_name, *i, j)] = (
                _split_groups,
                (applied.name, *i),
                j,
                slice(idx[j], idx[j + 1]),
            )

    # now construct an array that can be passed to _tree_reduce
    intergraph = HighLevelGraph.from_collections(split_name, dsk, dependencies=(applied,))
    intermediate = dask.array.Array(
        intergraph,
        name=split_name,
        chunks=applied.chunks + ((1,) * split_out,),
        meta=applied._meta,
    )
    return intermediate, group_chunks


def groupby_agg(
    array: "DaskArray",
    by: Union["DaskArray", np.ndarray],
    agg: Aggregation,
    expected_groups: Optional[Union[Sequence, np.ndarray]],
    axis: Sequence = None,
    split_out: int = 1,
    fill_value: Any = None,
    method: str = "mapreduce",
    isbin: bool = False,
) -> Tuple["DaskArray", Union[np.ndarray, "DaskArray"]]:

    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"
    token = dask.base.tokenize(array, by, agg, expected_groups, axis, split_out)

    # This is necessary for argreductions.
    # We need to rechunk before zipping up with the index
    # let's always do it anyway
    # but first save by if blockwise is True.
    if method == "blockwise":
        by_maybe_numpy = by
    _, (array, by) = dask.array.unify_chunks(array, inds, by, inds[-by.ndim :])

    # preprocess the array
    if agg.preprocess:
        array = agg.preprocess(array, axis=axis)

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,  # type: ignore
            axis=axis,
            # with the current implementation we want reindexing at the blockwise step
            # only reindex to groups present at combine stage
            expected_groups=expected_groups if split_out > 1 or isbin else None,
            fill_value=agg.fill_value["intermediate"],
            isbin=isbin,
            reindex=split_out > 1,
        ),
        inds,
        array,
        inds,
        by,
        inds[-by.ndim :],
        concatenate=False,
        dtype=array.dtype,
        meta=array._meta,
        align_arrays=False,
        token=f"{name}-chunk-{token}",
    )

    if split_out > 1:
        if expected_groups is None:
            # This could be implemented using the "hash_split" strategy
            # from dask.dataframe
            raise NotImplementedError

        intermediate, group_chunks = split_blocks(
            applied, split_out, expected_groups, split_name=f"{name}-split-{token}"
        )
        expected_agg = None
    else:
        intermediate = applied
        # from this point on, we just work with bin indexes when binning
        if isbin:
            expected_groups = np.arange(len(expected_groups) - 1)
        group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
        expected_agg = expected_groups

    if method == "mapreduce":
        # reduced is really a dict mapping reduction name to array
        # and "groups" to an array of group labels
        # Note: it does not make sense to interpret axis relative to
        # shape of intermediate results after the blockwise call
        reduced = dask.array.reductions._tree_reduce(
            intermediate,
            aggregate=partial(
                _npg_aggregate,
                agg=agg,
                expected_groups=expected_agg,
                group_ndim=by.ndim,
                fill_value=fill_value,
            ),
            combine=partial(_npg_combine, agg=agg, group_ndim=by.ndim),
            name=f"{name}-reduce",
            dtype=array.dtype,
            axis=axis,
            keepdims=True,
            concatenate=False,
        )
        output_chunks = reduced.chunks[: -(len(axis) + int(split_out > 1))] + (group_chunks,)
    else:
        # Blockwise apply the aggregation step so that one input chunk → one output chunk
        # TODO: We could combine this with the chunk reduction and do everything in one task.
        #       This would also optimize the single block along reduced-axis case.
        if (
            expected_groups is None
            or split_out > 1
            or len(axis) > 1
            or not isinstance(by_maybe_numpy, np.ndarray)
        ):
            raise NotImplementedError

        reduced = dask.array.blockwise(
            partial(
                _npg_aggregate,
                agg=agg,
                expected_groups=None,
                group_ndim=by.ndim,
                fill_value=fill_value,
                axis=axis,
                keepdims=True,
            ),
            inds,
            intermediate,
            inds,
            concatenate=False,
            dtype=array.dtype,
            meta=array._meta,
            align_arrays=False,
            name=f"{name}-blockwise-agg-{token}",
        )
        chunks = array.chunks[axis[0]]

        # find number of groups in each chunk, this is needed for output chunks
        # along the reduced axis
        bnds = np.insert(np.cumsum(chunks), 0, 0)
        groups_per_chunk = tuple(
            len(np.unique(by_maybe_numpy[i0:i1])) for i0, i1 in zip(bnds[:-1], bnds[1:])
        )
        output_chunks = reduced.chunks[: -(len(axis))] + (groups_per_chunk,)

    def _getitem(d, key1, key2):
        return d[key1][key2]

    # extract results from the dict
    result: Dict = {}
    layer: Dict[Tuple, Tuple] = {}
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    if expected_groups is None:
        groups_name = f"groups-{name}-{token}"
        # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
        first_block = len(ochunks) * (0,)
        layer[(groups_name, *first_block)] = (
            operator.getitem,
            (reduced.name, *first_block),
            "groups",
        )
        groups = (
            dask.array.Array(
                HighLevelGraph.from_collections(groups_name, layer, dependencies=[reduced]),
                groups_name,
                chunks=(group_chunks,),
                dtype=by.dtype,
            ),
        )
    else:
        groups = (expected_groups,)

    layer: Dict[Tuple, Tuple] = {}  # type: ignore
    agg_name = f"{name}-{token}"
    for ochunk in itertools.product(*ochunks):
        if method == "blockwise":
            inchunk = ochunk
        else:
            inchunk = ochunk[:-1] + (0,) * (len(axis)) + (ochunk[-1],) * int(split_out > 1)
        layer[(agg_name, *ochunk)] = (
            operator.getitem,
            (reduced.name, *inchunk),
            agg.name,
        )
    result = dask.array.Array(
        HighLevelGraph.from_collections(agg_name, layer, dependencies=[reduced]),
        agg_name,
        chunks=output_chunks,
        dtype=agg.dtype,
    )

    return (result, *groups)


def groupby_reduce(
    array: Union[np.ndarray, "DaskArray"],
    by: Union[np.ndarray, "DaskArray"],
    func: Union[str, Aggregation],
    expected_groups: Union[Sequence, np.ndarray] = None,
    axis=None,
    fill_value=None,
    split_out: int = 1,
    method="mapreduce",
    isbin: bool = False,
) -> Tuple["DaskArray", Union[np.ndarray, "DaskArray"]]:
    """
    GroupBy reductions using tree reductions for dask.array

    Parameters
    ----------
    array: numpy.ndarray, dask.array.Array
        Array to be reduced, nD
    by: numpy.ndarray, dask.array.Array
        Array of labels to group over. Must be aligned with `array` so that
            ``array.shape[-by.ndim :] == by.shape``
    func: str or Aggregation
        Single function name or an Aggregation instance
    expected_groups: (optional) Sequence
        Expected unique labels.
    axis: (optional) None or int or Sequence[int]
        If None, reduce across all dimensions of by
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim
    fill_value: Any
        Value when a label in `expected_groups` is not present
    split_out: int, optional
        Number of chunks along group axis in output (last axis)
    method: {"mapreduce", "blockwise", "cohorts"}, optional
        Strategy for reduction. Applies to dask arrays only

          * "mapreduce" : First apply the reduction blockwise on ``array``, then
                          combine a few newighbouring blocks, apply the reduction.
                          Continue until finalizing. Usually, ``func`` will need
                          to be an Aggregation instance for this method to work. Common
                          aggregations are implemented.
          * "blockwise" : Only reduce using blockwise and avoid aggregating blocks together.
                          Useful for resampling reductions where group members are always together.
                          The array is rechunked so that chunk boundaries line up with group boundaries
                          i.e. each block contains exactly one group.
          * "cohorts" : Finds group labels that tend to occur together ("cohorts"), indexes
                        out cohorts and reduces that subset using "mapreduce", repeat for all cohorts.
                        This works well for many time groupings where the group labels repeat
                        at regular intervals like 'hour', 'month', dayofyear' etc. Optimize
                        chunking ``array`` for this method by first rechunking using ``rechunk_for_cohorts``.
    isbin: bool, optional
        Are `expected_groups` bin edges?

    Returns
    -------
    result
        Aggregated result
    *groups
        Group labels
    """

    if not is_duck_array(by):
        by = np.asarray(by)
    if not is_duck_array(array):
        array = np.asarray(array)
    if array.shape[-by.ndim :] != by.shape:
        raise ValueError(
            "array and by must be aligned i.e. array.shape[-by.ndim :] == by.shape. "
            f"Received array of shape {array.shape} and by of shape {by.shape}"
        )

    if axis is None:
        axis = tuple(array.ndim + np.arange(-by.ndim, 0))
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, array.ndim)  # type: ignore

    if expected_groups is None and isinstance(by, np.ndarray):
        expected_groups = np.unique(by)
        if np.issubdtype(expected_groups.dtype, np.floating):  # type: ignore
            expected_groups = expected_groups[~np.isnan(expected_groups)]

    # TODO: make sure expected_groups is unique
    if len(axis) == 1 and by.ndim > 1 and expected_groups is None:
        # When we reduce along all axes, it guarantees that we will see all
        # groups in the final combine stage, so everything works.
        # This is not necessarily true when reducing along a subset of axes
        # (of by)
        # TODO: depends on chunking of by?
        # we could relax this if there is only one chunk along all
        # by dim != axis?
        raise NotImplementedError(
            "Please provide ``expected_groups`` when not reducing along all axes."
        )

    if isinstance(axis, Sequence) and len(axis) < by.ndim:
        by = _move_reduce_dims_to_end(by, -array.ndim + np.array(axis) + by.ndim)
        array = _move_reduce_dims_to_end(array, axis)
        axis = tuple(array.ndim + np.arange(-len(axis), 0))

    if not isinstance(func, Aggregation):
        try:
            # TODO: need better interface
            # we set dtype, fillvalue on reduction later. so deepcopy now
            reduction = copy.deepcopy(getattr(aggregations, func))
        except AttributeError:
            raise NotImplementedError(f"Reduction {func!r} not implemented yet")
    else:
        reduction = func

    if reduction.dtype is None:
        reduction.dtype = array.dtype
    # Replace sentinel fill values according to dtype
    reduction.fill_value["intermediate"] = tuple(
        _get_fill_value(reduction.dtype, fv) for fv in reduction.fill_value["intermediate"]
    )
    reduction.fill_value[func] = _get_fill_value(reduction.dtype, reduction.fill_value[func])

    # TODO: handle reduction being something custom not present in numpy_groupies
    if not is_duck_dask_array(array) and not is_duck_dask_array(by):
        fv = reduction.fill_value[func] if fill_value is None else fill_value
        # for pure numpy grouping, we just use npg directly and avoid "finalizing"
        # (agg.finalize = None). We still need to do the reindexing step in finalize
        # so that everything matches the dask version.
        # Also, npg's count counts the number of groups
        # we want to count the number of non-NaN array elements in each group
        # So we use our custom _count instead of "count"
        func = reduction.name if reduction.name != "count" else _count
        results = chunk_reduce(
            array,
            by,
            func=func,
            axis=axis,
            expected_groups=expected_groups if isbin else None,
            fill_value=fv,
            dtype=reduction.dtype,
            isbin=isbin,
        )  # type: ignore

        if reduction.name in ["argmin", "argmax", "nanargmax", "nanargmin"]:
            print(results["intermediates"][0])
            if array.ndim > 1 and by.ndim == 1:
                # Fix npg bug where argmax with nD array, 1D group_idx, axis=-1
                # will return wrong indices
                results["intermediates"][0] = np.unravel_index(
                    results["intermediates"][0], array.shape
                )[-1]
                print(results["intermediates"][0])

        if isbin:
            expected_groups = np.arange(len(expected_groups) - 1)
        reduction.finalize = None
        result = _finalize_results(
            results,
            reduction,
            axis,
            expected_groups,
            fill_value=fill_value,
            mask_counts=False,
        )
        groups = (result["groups"],)
        result = result[reduction.name]

    else:
        if func in ["first", "last"]:
            raise NotImplementedError("first, last not implemented for dask arrays")

        if fill_value is None:
            fill_value = reduction.fill_value[func]

        # we need to explicitly track counts so that we can mask at the end
        if fill_value is not None:
            reduction.chunk += (_count,)
            reduction.combine += ("sum",)
            reduction.fill_value["intermediate"] += (0,)

        if method == "cohorts":
            assert len(axis) == 1

            cohorts = find_group_cohorts(by, array.chunks[axis[0]], merge=True)
            idx = np.arange(len(by))

            results = []
            groups_ = []
            for cohort in cohorts:
                # indexes for a subset of groups
                subset_idx = idx[np.isin(by, cohort)]
                # get final result for these groups
                r, *g = groupby_agg(
                    array[..., subset_idx],
                    by[subset_idx],
                    reduction,
                    expected_groups=cohort,
                    axis=axis,
                    split_out=split_out,
                    fill_value=fill_value,
                    method="mapreduce",
                    isbin=isbin,
                )
                results.append(r)
                groups_.append(g)

            # concatenate results together
            groups = (np.hstack(groups_),)
            result = np.concatenate(results, axis=-1)

        else:
            if method == "blockwise":
                if by.ndim > 1:
                    raise ValueError(
                        "For method='blockwise', ``by`` must be 1D. "
                        f"Received {by.ndim} dimensions instead."
                    )
                array = rechunk_for_blockwise(array, axis=-1, labels=by)

            # TODO: test with mixed array kinds (numpy + dask; dask + numpy)
            result, *groups = groupby_agg(
                array,
                by,
                reduction,
                expected_groups,
                axis=axis,
                split_out=split_out,
                fill_value=fill_value,
                method=method,
                isbin=isbin,
            )

    return (result, *groups)
