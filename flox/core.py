from __future__ import annotations

import copy
import itertools
import logging
import math
import operator
import sys
import warnings
from collections import namedtuple
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import product
from numbers import Integral
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy_groupies as npg
import pandas as pd
import toolz as tlz
from scipy.sparse import csc_array, csr_array

from . import xrdtypes
from .aggregate_flox import _prepare_for_flox
from .aggregations import (
    AGGREGATIONS,
    Aggregation,
    AlignedArrays,
    Scan,
    ScanState,
    _atleast_1d,
    _initialize_aggregation,
    generic_aggregate,
    quantile_new_dims_func,
    topk_new_dims_func,
)
from .cache import memoize
from .lib import ArrayLayer
from .xrutils import (
    _contains_cftime_datetimes,
    _to_pytimedelta,
    datetime_to_numeric,
    is_chunked_array,
    is_duck_array,
    is_duck_cubed_array,
    is_duck_dask_array,
    isnull,
    module_available,
    notnull,
)

if module_available("numpy", minversion="2.0.0"):
    from numpy.lib.array_utils import normalize_axis_tuple
else:
    from numpy.core.numeric import normalize_axis_tuple  # type: ignore[no-redef]

HAS_NUMBAGG = module_available("numbagg", minversion="0.3.0")

if TYPE_CHECKING:
    try:
        if sys.version_info < (3, 11):
            from typing_extensions import Unpack
        else:
            from typing import Unpack
    except (ModuleNotFoundError, ImportError):
        Unpack: Any  # type: ignore[no-redef]
    from .types import CubedArray, DaskArray, Graph

    T_DuckArray: TypeAlias = np.ndarray | DaskArray | CubedArray  # Any ?
    T_By: TypeAlias = T_DuckArray
    T_Bys = tuple[T_By, ...]
    T_ExpectIndex = pd.Index
    T_ExpectIndexTuple = tuple[T_ExpectIndex, ...]
    T_ExpectIndexOpt = T_ExpectIndex | None
    T_ExpectIndexOptTuple = tuple[T_ExpectIndexOpt, ...]
    T_Expect = Sequence | np.ndarray | T_ExpectIndex
    T_ExpectTuple = tuple[T_Expect, ...]
    T_ExpectOpt = Sequence | np.ndarray | T_ExpectIndexOpt
    T_ExpectOptTuple = tuple[T_ExpectOpt, ...]
    T_ExpectedGroups = T_Expect | T_ExpectOptTuple
    T_ExpectedGroupsOpt = T_ExpectedGroups | None
    T_Func = str | Callable
    T_Funcs = T_Func | Sequence[T_Func]
    T_Agg = str | Aggregation
    T_Scan = str | Scan
    T_Axis = int
    T_Axes = tuple[T_Axis, ...]
    T_AxesOpt = T_Axis | T_Axes | None
    T_Dtypes = np.typing.DTypeLike | Sequence[np.typing.DTypeLike] | None
    T_FillValues = np.typing.ArrayLike | Sequence[np.typing.ArrayLike] | None
    T_Engine = Literal["flox", "numpy", "numba", "numbagg"]
    T_EngineOpt = None | T_Engine
    T_Method = Literal["map-reduce", "blockwise", "cohorts"]
    T_MethodOpt = None | Literal["map-reduce", "blockwise", "cohorts"]
    T_IsBins = bool | Sequence[bool]

T = TypeVar("T")

IntermediateDict = dict[str | Callable, Any]
FinalResultsDict = dict[str, Union["DaskArray", "CubedArray", np.ndarray]]
FactorProps = namedtuple("FactorProps", "offset_group nan_sentinel nanmask")

# This dummy axis is inserted using np.expand_dims
# and then reduced over during the combine stage by
# _simple_combine.
DUMMY_AXIS = -2

logger = logging.getLogger("flox")


class FactorizeKwargs(TypedDict, total=False):
    """Used in _factorize_multiple"""

    by: T_Bys
    axes: T_Axes
    fastpath: bool
    expected_groups: T_ExpectIndexOptTuple | None
    reindex: bool
    sort: bool


def _postprocess_numbagg(result, *, func, fill_value, size, seen_groups):
    """Account for numbagg not providing a fill_value kwarg."""
    from .aggregate_numbagg import DEFAULT_FILL_VALUE

    if not isinstance(func, str) or func not in DEFAULT_FILL_VALUE:
        return result
    # The condition needs to be
    # len(found_groups) < size; if so we mask with fill_value (?)
    default_fv = DEFAULT_FILL_VALUE[func]
    needs_masking = fill_value is not None and not np.array_equal(fill_value, default_fv, equal_nan=True)
    groups = np.arange(size)
    if needs_masking:
        mask = np.isin(groups, seen_groups, assume_unique=True, invert=True)
        if mask.any():
            result[..., groups[mask]] = fill_value
    return result


def identity(x: T) -> T:
    return x


def _issorted(arr: np.ndarray) -> bool:
    return bool((arr[:-1] <= arr[1:]).all())


def _is_arg_reduction(func: T_Agg) -> bool:
    if isinstance(func, str) and func in ["argmin", "argmax", "nanargmax", "nanargmin"]:
        return True
    if isinstance(func, Aggregation) and func.reduction_type == "argreduce":
        return True
    return False


def _is_minmax_reduction(func: T_Agg) -> bool:
    return not _is_arg_reduction(func) and (isinstance(func, str) and ("max" in func or "min" in func))


def _is_first_last_reduction(func: T_Agg) -> bool:
    if isinstance(func, Aggregation):
        func = func.name
    return func in ["nanfirst", "nanlast", "first", "last"]


def _is_bool_supported_reduction(func: T_Agg) -> bool:
    if isinstance(func, Aggregation):
        func = func.name
    return (
        func in ["all", "any", "topk"]
        # TODO: enable in npg
        # or _is_first_last_reduction(func)
        # or _is_minmax_reduction(func)
    )


def _get_expected_groups(by: T_By, sort: bool) -> T_ExpectIndex:
    if is_duck_dask_array(by):
        raise ValueError("Please provide expected_groups if not grouping by a numpy array.")
    flatby = by.reshape(-1)
    expected = pd.unique(flatby[notnull(flatby)])
    return _convert_expected_groups_to_index((expected,), isbin=(False,), sort=sort)[0]


def _get_chunk_reduction(reduction_type: Literal["reduce", "argreduce"]) -> Callable:
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def is_nanlen(reduction: T_Func) -> bool:
    return isinstance(reduction, str) and reduction == "nanlen"


def _move_reduce_dims_to_end(arr: np.ndarray, axis: T_Axes) -> np.ndarray:
    """Transpose `arr` by moving `axis` to the end."""
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    arr = arr.transpose(order)
    return arr


def _collapse_axis(arr: np.ndarray, naxis: int) -> np.ndarray:
    """Reshape so that the last `naxis` axes are collapsed to one axis."""
    newshape = arr.shape[:-naxis] + (math.prod(arr.shape[-naxis:]),)
    return arr.reshape(newshape)


@memoize
def _get_optimal_chunks_for_groups(chunks, labels):
    chunkidx = np.cumsum(chunks) - 1
    # what are the groups at chunk boundaries
    labels_at_chunk_bounds = _unique(labels[chunkidx])
    # what's the last index of all groups
    last_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="last")
    # what's the last index of groups at the chunk boundaries.
    lastidx = last_indexes[labels_at_chunk_bounds]

    if len(chunkidx) == len(lastidx) and (chunkidx == lastidx).all():
        return chunks

    first_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="first")
    firstidx = first_indexes[labels_at_chunk_bounds]

    newchunkidx = [0]
    for c, f, l in zip(chunkidx, firstidx, lastidx):  # noqa
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


def _unique(a: np.ndarray) -> np.ndarray:
    """Much faster to use pandas unique and sort the results.
    np.unique sorts before uniquifying and is slow."""
    return np.sort(pd.unique(a.reshape(-1)))


def slices_from_chunks(chunks):
    """slightly modified from dask.array.core.slices_from_chunks to be lazy"""
    cumdims = [tlz.accumulate(operator.add, bds, 0) for bds in chunks]
    slices = (
        (slice(s, s + dim) for s, dim in zip(starts, shapes)) for starts, shapes in zip(cumdims, chunks)
    )
    return product(*slices)


def _compute_label_chunk_bitmask(labels, chunks, nlabels):
    def make_bitmask(rows, cols):
        data = np.broadcast_to(np.array(1, dtype=np.uint8), rows.shape)
        return csc_array((data, (rows, cols)), dtype=bool, shape=(nchunks, nlabels))

    assert isinstance(labels, np.ndarray)
    shape = tuple(sum(c) for c in chunks)
    nchunks = math.prod(len(c) for c in chunks)
    approx_chunk_size = math.prod(c[0] for c in chunks)

    # Shortcut for 1D with size-1 chunks
    if shape == (nchunks,):
        rows_array = np.arange(nchunks)
        cols_array = labels
        mask = labels >= 0
        return make_bitmask(rows_array[mask], cols_array[mask])

    labels = np.broadcast_to(labels, shape[-labels.ndim :])
    cols = []
    ilabels = np.arange(nlabels)

    def chunk_unique(labels, slicer, nlabels, label_is_present=None):
        if label_is_present is None:
            label_is_present = np.empty((nlabels + 1,), dtype=bool)
        label_is_present[:] = False
        subset = labels[slicer]
        # This is a quite fast way to find unique integers, when we know how many there are
        # inspired by a similar idea in numpy_groupies for first, last
        # instead of explicitly finding uniques, repeatedly write True to the same location
        label_is_present[subset.reshape(-1)] = True
        # skip the -1 sentinel by slicing
        # Faster than np.argwhere by a lot
        uniques = ilabels[label_is_present[:-1]]
        return uniques

    # TODO: refine this heuristic.
    # The general idea is that with the threadpool, we repeatedly allocate memory
    # for `label_is_present`. We trade that off against the parallelism across number of chunks.
    # For large enough number of chunks (relative to number of labels), it makes sense to
    # suffer the extra allocation in exchange for parallelism.
    THRESHOLD = 2
    if nlabels < THRESHOLD * approx_chunk_size:
        logger.debug(
            "Using threadpool since num_labels %s < %d * chunksize %s",
            nlabels,
            THRESHOLD,
            approx_chunk_size,
        )
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(chunk_unique, labels, slicer, nlabels)
                for slicer in slices_from_chunks(chunks)
            ]
            cols = tuple(f.result() for f in futures)

    else:
        logger.debug(
            "Using serial loop since num_labels %s > %d * chunksize %s",
            nlabels,
            THRESHOLD,
            approx_chunk_size,
        )
        cols = []
        # Add one to handle the -1 sentinel value
        label_is_present = np.empty((nlabels + 1,), dtype=bool)
        for region in slices_from_chunks(chunks):
            uniques = chunk_unique(labels, region, nlabels, label_is_present)
            cols.append(uniques)
    rows_array = np.repeat(np.arange(nchunks), tuple(len(col) for col in cols))
    cols_array = np.concatenate(cols)

    return make_bitmask(rows_array, cols_array)


# @memoize
def find_group_cohorts(
    labels, chunks, expected_groups: None | pd.RangeIndex = None, merge: bool = False
) -> tuple[T_Method, dict]:
    """
    Finds groups labels that occur together aka "cohorts"

    If available, results are cached in a 1MB cache managed by `cachey`.
    This allows us to be quick when repeatedly calling groupby_reduce
    for arrays with the same chunking (e.g. an xarray Dataset).

    Parameters
    ----------
    labels : np.ndarray
        mD Array of integer group codes, factorized so that -1
        represents NaNs.
    chunks : tuple
        chunks of the array being reduced
    expected_groups: pd.RangeIndex (optional)
        Used to extract the largest label expected
    merge: bool (optional)
        Whether to merge cohorts or not. Set to True if a user
        specifies "cohorts" but other methods are preferable.

    Returns
    -------
    preferred_method: {"blockwise", cohorts", "map-reduce"}
    cohorts: dict_values
        Iterable of cohorts
    """
    # To do this, we must have values in memory so casting to numpy should be safe
    labels = np.asarray(labels)

    shape = tuple(sum(c) for c in chunks)
    nchunks = math.prod(len(c) for c in chunks)

    # assumes that `labels` are factorized
    if expected_groups is None:
        nlabels = labels.max() + 1
    else:
        nlabels = expected_groups[-1] + 1

    # 1. Single chunk, blockwise always
    if nchunks == 1:
        return "blockwise", {(0,): list(range(nlabels))}

    labels = np.broadcast_to(labels, shape[-labels.ndim :])
    bitmask = _compute_label_chunk_bitmask(labels, chunks, nlabels)

    CHUNK_AXIS, LABEL_AXIS = 0, 1
    chunks_per_label = bitmask.sum(axis=CHUNK_AXIS)

    # can happen when `expected_groups` is passed but not all labels are present
    # (binning, resampling)
    present_labels = np.arange(bitmask.shape[LABEL_AXIS])
    present_labels_mask = chunks_per_label != 0
    if not present_labels_mask.all():
        present_labels = present_labels[present_labels_mask]
        bitmask = bitmask[..., present_labels_mask]
        chunks_per_label = chunks_per_label[present_labels_mask]

    label_chunks = {
        present_labels[idx].item(): bitmask.indices[slice(bitmask.indptr[idx], bitmask.indptr[idx + 1])]
        for idx in range(bitmask.shape[LABEL_AXIS])
    }

    # Invert the label_chunks mapping so we know which labels occur together.
    def invert(x) -> tuple[np.ndarray, ...]:
        arr = label_chunks[x]
        return tuple(arr.tolist())

    chunks_cohorts = tlz.groupby(invert, label_chunks.keys())

    # 2. Every group is contained to one block, use blockwise here.
    if bitmask.shape[CHUNK_AXIS] == 1 or (chunks_per_label == 1).all():
        logger.debug("find_group_cohorts: blockwise is preferred.")
        return "blockwise", chunks_cohorts

    # 3. Perfectly chunked so there is only a single cohort
    if len(chunks_cohorts) == 1:
        logger.debug("Only found a single cohort. 'map-reduce' is preferred.")
        return "map-reduce", chunks_cohorts if merge else {}

    # 4. Our dataset has chunksize one along the axis,
    single_chunks = all(all(a == 1 for a in ac) for ac in chunks)
    # 5. Every chunk only has a single group, but that group might extend across multiple chunks
    one_group_per_chunk = (bitmask.sum(axis=LABEL_AXIS) == 1).all()
    # 6. Existing cohorts don't overlap, great for time grouping with perfect chunking
    no_overlapping_cohorts = (np.bincount(np.concatenate(tuple(chunks_cohorts.keys()))) == 1).all()
    if one_group_per_chunk or single_chunks or no_overlapping_cohorts:
        logger.debug("find_group_cohorts: cohorts is preferred, chunking is perfect.")
        return "cohorts", chunks_cohorts

    # We'll use containment to measure degree of overlap between labels.
    # Containment C = |Q & S| / |Q|
    #  - |X| is the cardinality of set X
    #  - Q is the query set being tested
    #  - S is the existing set
    # The bitmask matrix S allows us to calculate this pretty efficiently using a dot product.
    #       S.T @ S / chunks_per_label
    #
    # We treat the sparsity(C) = (nnz/size) as a summary measure of the net overlap.
    # 1. For high enough sparsity, there is a lot of overlap and we should use "map-reduce".
    # 2. When labels are uniformly distributed amongst all chunks
    #    (and number of labels < chunk size), sparsity is 1.
    # 3. Time grouping cohorts (e.g. dayofyear) appear as lines in this matrix.
    # 4. When there are no overlaps at all between labels, containment is a block diagonal matrix
    #    (approximately).
    #
    # However computing S.T @ S can still be the slowest step, especially if S
    # is not particularly sparse. Empirically the sparsity( S.T @ S ) > min(1, 2 x sparsity(S)).
    # So we use sparsity(S) as a shortcut.
    MAX_SPARSITY_FOR_COHORTS = 0.4  # arbitrary
    sparsity = bitmask.nnz / math.prod(bitmask.shape)
    preferred_method: Literal["map-reduce"] | Literal["cohorts"]
    logger.debug(
        "sparsity of bitmask is {}, threshold is {}".format(  # noqa
            sparsity, MAX_SPARSITY_FOR_COHORTS
        )
    )
    # 7. Groups seem fairly randomly distributed, use "map-reduce".
    if sparsity > MAX_SPARSITY_FOR_COHORTS:
        if not merge:
            logger.debug(
                "find_group_cohorts: bitmask sparsity={}, merge=False, choosing 'map-reduce'".format(  # noqa
                    sparsity
                )
            )
            return "map-reduce", {}
        preferred_method = "map-reduce"
    else:
        preferred_method = "cohorts"

    # Note: While A.T @ A is a symmetric matrix, the division by chunks_per_label
    #       makes it non-symmetric.
    asfloat = bitmask.astype(float)
    containment = csr_array(asfloat.T @ asfloat / chunks_per_label)

    logger.debug(
        "sparsity of containment matrix is {}".format(  # noqa
            containment.nnz / math.prod(containment.shape)
        )
    )

    # Next we for-loop over groups and merge those that are quite similar.
    # Use a threshold on containment to always force some merging.
    # Note that we do not use the filtered containment matrix for estimating "sparsity"
    # because it is a bit hard to reason about.
    MIN_CONTAINMENT = 0.75  # arbitrary
    mask = containment.data < MIN_CONTAINMENT

    # Now we also know "exact cohorts" -- cohorts whose constituent groups
    # occur in exactly the same chunks. We only need examine one member of each group.
    # Skip the others by first looping over the exact cohorts, and zero out those rows.
    repeated = np.concatenate([v[1:] for v in chunks_cohorts.values()]).astype(int)
    repeated_idx = np.searchsorted(present_labels, repeated)
    for i in repeated_idx:
        mask[containment.indptr[i] : containment.indptr[i + 1]] = True
    containment.data[mask] = 0
    containment.eliminate_zeros()

    # Figure out all the labels we need to loop over later
    n_overlapping_labels = containment.astype(bool).sum(axis=1)
    order = np.argsort(n_overlapping_labels, kind="stable")[::-1]
    # Order is such that we iterate over labels, beginning with those with most overlaps
    # Also filter out any "exact" cohorts
    order = order[n_overlapping_labels[order] > 0]

    logger.debug("find_group_cohorts: merging cohorts")
    merged_cohorts = {}
    merged_keys = set()
    for rowidx in order:
        if present_labels[rowidx] in merged_keys:
            continue
        cohidx = containment.indices[slice(containment.indptr[rowidx], containment.indptr[rowidx + 1])]
        cohort_ = present_labels[cohidx]
        cohort = [elem.item() for elem in cohort_ if elem not in merged_keys]
        if not cohort:
            continue
        merged_keys.update(cohort)
        allchunks = (label_chunks[member].tolist() for member in cohort)
        chunk = tuple(set(itertools.chain(*allchunks)))
        merged_cohorts[chunk] = cohort

    actual_ngroups = np.concatenate(tuple(merged_cohorts.values())).size
    expected_ngroups = present_labels.size
    assert len(merged_keys) == actual_ngroups
    assert expected_ngroups == actual_ngroups, (expected_ngroups, actual_ngroups)

    # sort by first label in cohort
    # This will help when sort=True (default)
    # and we have to resort the dask array
    as_sorted = dict(sorted(merged_cohorts.items(), key=lambda kv: kv[1][0]))
    return preferred_method, as_sorted


def rechunk_for_cohorts(
    array: DaskArray,
    axis: T_Axis,
    labels: np.ndarray,
    force_new_chunk_at: Sequence,
    chunksize: int | None = None,
    ignore_old_chunks: bool = False,
    debug: bool = False,
) -> DaskArray:
    """
    Rechunks array so that each new chunk contains groups that always occur together.

    Parameters
    ----------
    array : dask.array.Array
        array to rechunk
    axis : int
        Axis to rechunk
    labels : np.ndarray
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
        If None, uses median chunksize along axis.

    Returns
    -------
    dask.array.Array
        rechunked array
    """
    if chunksize is None:
        chunksize = np.median(array.chunks[axis]).astype(int)

    if len(labels) != array.shape[axis]:
        raise ValueError(
            "labels must be equal to array.shape[axis]. "
            f"Received length {len(labels)}.  Expected length {array.shape[axis]}"
        )

    force_new_chunk_at = _atleast_1d(force_new_chunk_at)
    oldchunks = array.chunks[axis]
    oldbreaks = np.insert(np.cumsum(oldchunks), 0, 0)
    if debug:
        labels_at_breaks = labels[oldbreaks[:-1]]
        print(labels_at_breaks[:40])

    isbreak = np.isin(labels, force_new_chunk_at)
    if not np.any(isbreak):
        raise ValueError("One or more labels in ``force_new_chunk_at`` not present in ``labels``.")

    divisions = []
    counter = 1
    for idx, lab in enumerate(labels):
        if lab in force_new_chunk_at or idx == 0:
            divisions.append(idx)
            counter = 1
            continue

        next_break = np.nonzero(isbreak[idx:])[0]
        if next_break.any():
            next_break_is_close = next_break[0] <= chunksize // 2
        else:
            next_break_is_close = False

        if (not ignore_old_chunks and idx in oldbreaks) or (counter >= chunksize and not next_break_is_close):
            divisions.append(idx)
            counter = 1
            continue

        counter += 1

    divisions.append(len(labels))
    if debug:
        labels_at_breaks = labels[divisions[:-1]]
        print(labels_at_breaks[:40])

    newchunks = tuple(np.diff(divisions))
    if debug:
        print(divisions[:10], newchunks[:10])
        print(divisions[-10:], newchunks[-10:])
    assert sum(newchunks) == len(labels)

    if newchunks == array.chunks[axis]:
        return array
    else:
        return array.rechunk({axis: newchunks})


def rechunk_for_blockwise(array: DaskArray, axis: T_Axis, labels: np.ndarray) -> DaskArray:
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    embarrassingly parallel group reductions.

    This only works when the groups are sequential
    (e.g. labels = ``[0,0,0,1,1,1,1,2,2]``).
    Such patterns occur when using ``.resample``.

    Parameters
    ----------
    array : DaskArray
        Array to rechunk
    axis : int
        Axis along which to rechunk the array.
    labels : np.ndarray
        Group labels

    Returns
    -------
    DaskArray
        Rechunked array
    """
    # TODO: this should be unnecessary?
    labels = factorize_((labels,), axes=())[0]
    chunks = array.chunks[axis]
    newchunks = _get_optimal_chunks_for_groups(chunks, labels)
    if newchunks == chunks:
        return array
    else:
        return array.rechunk({axis: newchunks})


def reindex_(
    array: np.ndarray,
    from_,
    to,
    fill_value: Any = None,
    axis: T_Axis = -1,
    promote: bool = False,
) -> np.ndarray:
    if not isinstance(to, pd.Index):
        if promote:
            to = pd.Index(to)
        else:
            raise ValueError("reindex requires a pandas.Index or promote=True")

    if to.ndim > 1:
        raise ValueError(f"Cannot reindex to a multidimensional array: {to}")

    if array.shape[axis] == 0:
        # all groups were NaN
        reindexed = np.full(array.shape[:-1] + (len(to),), fill_value, dtype=array.dtype)
        return reindexed

    from_ = pd.Index(from_)
    # short-circuit for trivial case
    if from_.equals(to):
        return array

    if from_.dtype.kind == "O" and isinstance(from_[0], tuple):
        raise NotImplementedError(
            "Currently does not support reindexing with object arrays of tuples. "
            "These occur when grouping by multi-indexed variables in xarray."
        )
    idx = from_.get_indexer(to)
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx
    reindexed = array[tuple(indexer)]
    if any(idx == -1):
        if fill_value is None:
            raise ValueError("Filling is required. fill_value cannot be None.")
        indexer[axis] = idx == -1
        # This allows us to match xarray's type promotion rules
        if fill_value is xrdtypes.NA or isnull(fill_value):
            new_dtype, fill_value = xrdtypes.maybe_promote(reindexed.dtype)
            reindexed = reindexed.astype(new_dtype, copy=False)
        reindexed[tuple(indexer)] = fill_value
    return reindexed


def offset_labels(labels: np.ndarray, ngroups: int) -> tuple[np.ndarray, int]:
    """
    Offset group labels by dimension. This is used when we
    reduce over a subset of the dimensions of by. It assumes that the reductions
    dimensions have been flattened in the last dimension
    Copied from xhistogram &
    https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    assert labels.ndim > 1
    offset: np.ndarray = (
        labels + np.arange(math.prod(labels.shape[:-1])).reshape((*labels.shape[:-1], -1)) * ngroups
    )
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[labels == -1] = -1
    size: int = math.prod(labels.shape[:-1]) * ngroups
    return offset, size


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    fastpath: Literal[True],
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, None]: ...


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: Literal[False] = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, FactorProps]: ...


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, FactorProps | None]: ...


def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, FactorProps | None]:
    """
    Returns an array of integer  codes  for groups (and associated data)
    by wrapping pd.cut and pd.factorize (depending on isbin).
    This method handles reindex and sort so that we don't spend time reindexing / sorting
    a possibly large results array. Instead we set up the appropriate integer codes (group_idx)
    so that the results come out in the appropriate order.
    """
    if expected_groups is None:
        expected_groups = (None,) * len(by)

    factorized = []
    found_groups = []
    for groupvar, expect in zip(by, expected_groups):
        flat = groupvar.reshape(-1)
        if isinstance(expect, pd.RangeIndex):
            # idx is a view of the original `by` array
            # copy here so we don't have a race condition with the
            # group_idx[nanmask] = nan_sentinel assignment later
            # this is important in shared-memory parallelism with dask
            # TODO: figure out how to avoid this
            idx = flat.copy()
            found_groups.append(np.array(expect))
            # TODO: fix by using masked integers
            idx[idx > expect[-1]] = -1

        elif isinstance(expect, pd.IntervalIndex):
            if expect.closed == "both":
                raise NotImplementedError
            bins = np.concatenate([expect.left.to_numpy(), expect.right.to_numpy()[[-1]]])

            # digitize is 0 or idx.max() for values outside the bounds of all intervals
            # make it behave like pd.cut which uses -1:
            if len(bins) > 1:
                right = expect.closed_right
                idx = np.digitize(
                    flat,
                    bins=bins.view(np.int64) if bins.dtype.kind == "M" else bins,
                    right=right,
                )
                idx -= 1
                within_bins = flat <= bins.max() if right else flat < bins.max()
                idx[~within_bins] = -1
            else:
                idx = np.zeros_like(flat, dtype=np.intp) - 1

            found_groups.append(np.array(expect))
        else:
            if expect is not None and reindex:
                sorter = np.argsort(expect)
                groups = expect[(sorter,)] if sort else expect
                idx = np.searchsorted(expect, flat, sorter=sorter)
                mask = ~np.isin(flat, expect) | isnull(flat) | (idx == len(expect))
                if not sort:
                    # idx is the index in to the sorted array.
                    # if we didn't want sorting, unsort it back
                    idx[(idx == len(expect),)] = -1
                    idx = sorter[(idx,)]
                idx[mask] = -1
            else:
                idx, groups = pd.factorize(flat, sort=sort)

            found_groups.append(np.array(groups))
        factorized.append(idx.reshape(groupvar.shape))

    grp_shape = tuple(len(grp) for grp in found_groups)
    ngroups = math.prod(grp_shape)
    if len(by) > 1:
        group_idx = np.ravel_multi_index(factorized, grp_shape, mode="wrap")
        # NaNs; as well as values outside the bins are coded by -1
        # Restore these after the raveling
        nan_by_mask = reduce(np.logical_or, [(f == -1) for f in factorized])
        group_idx[nan_by_mask] = -1
    else:
        group_idx = factorized[0]

    if fastpath:
        return group_idx, tuple(found_groups), grp_shape, ngroups, ngroups, None

    if len(axes) == 1 and groupvar.ndim > 1:
        # Not reducing along all dimensions of by
        # this is OK because for 3D by and axis=(1,2),
        # we collapse to a 2D by and axis=-1
        offset_group = True
        group_idx, size = offset_labels(group_idx.reshape(by[0].shape), ngroups)
    else:
        size = ngroups
        offset_group = False

    # numpy_groupies cannot deal with group_idx = -1
    # so we'll add use ngroups as the sentinel
    # note we cannot simply remove the NaN locations;
    # that would mess up argmax, argmin
    nan_sentinel = size if offset_group else ngroups
    nanmask = group_idx == -1
    if nanmask.any():
        # bump it up so there's a place to assign values to the nan_sentinel index
        size += 1
    group_idx[nanmask] = nan_sentinel

    props = FactorProps(offset_group, nan_sentinel, nanmask)
    return group_idx, tuple(found_groups), grp_shape, ngroups, size, props


def chunk_argreduce(
    array_plus_idx: tuple[np.ndarray, ...],
    by: np.ndarray,
    func: T_Funcs,
    expected_groups: pd.Index | None,
    axis: T_AxesOpt,
    fill_value: T_FillValues,
    dtype: T_Dtypes = None,
    reindex: bool = False,
    engine: T_Engine = "numpy",
    sort: bool = True,
    user_dtype=None,
) -> IntermediateDict:
    """
    Per-chunk arg reduction.

    Expects a tuple of (array, index along reduction axis). Inspired by
    dask.array.reductions.argtopk
    """
    array, idx = array_plus_idx
    by = np.broadcast_to(by, array.shape)

    results = chunk_reduce(
        array,
        by,
        func,
        expected_groups=None,
        axis=axis,
        fill_value=fill_value,
        dtype=dtype,
        engine=engine,
        sort=sort,
        user_dtype=user_dtype,
    )
    if not isnull(results["groups"]).all():
        idx = np.broadcast_to(idx, array.shape)

        # array, by get flattened to 1D before passing to npg
        # so the indexes need to be unraveled
        newidx = np.unravel_index(results["intermediates"][1], array.shape)

        # Now index into the actual "global" indexes `idx`
        results["intermediates"][1] = idx[newidx]

    if reindex and expected_groups is not None:
        results["intermediates"][1] = reindex_(
            results["intermediates"][1],
            results["groups"].squeeze(),
            expected_groups,
            fill_value=0,
        )

    assert results["intermediates"][0].shape == results["intermediates"][1].shape

    return results


def chunk_reduce(
    array: np.ndarray,
    by: np.ndarray,
    func: T_Funcs,
    expected_groups: pd.Index | None,
    axis: T_AxesOpt = None,
    fill_value: T_FillValues = None,
    dtype: T_Dtypes = None,
    reindex: bool = False,
    engine: T_Engine = "numpy",
    kwargs: Sequence[dict] | None = None,
    sort: bool = True,
    user_dtype=None,
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
    array : numpy.ndarray
        Array of values to reduced
    by : numpy.ndarray
        Array to group by.
    func : str or Callable or Sequence[str] or Sequence[Callable]
        Name of reduction or function, passed to numpy_groupies.
        Supports multiple reductions.
    axis : (optional) int or Sequence[int]
        If None, reduce along all dimensions of array.
        Else reduce along specified axes.

    Returns
    -------
    dict
    """

    funcs = _atleast_1d(func)
    nfuncs = len(funcs)
    dtypes = _atleast_1d(dtype, nfuncs)
    fill_values = _atleast_1d(fill_value, nfuncs)
    kwargss = _atleast_1d({} if kwargs is None else kwargs, nfuncs)

    if isinstance(axis, Sequence):
        axes: T_Axes = axis
        nax = len(axes)
    else:
        nax = by.ndim
        axes = () if axis is None else (axis,) * nax

    assert by.ndim <= array.ndim

    final_array_shape = array.shape[:-nax] + (1,) * (nax - 1)
    final_groups_shape = (1,) * (nax - 1)

    if 1 < nax < by.ndim:
        # when axis is a tuple
        # collapse and move reduction dimensions to the end
        by = _collapse_axis(by, nax)
        array = _collapse_axis(array, nax)
        axes = (-1,)
        nax = 1

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, grps, found_groups_shape, _, size, props = factorize_(
        (by,), axes, expected_groups=(expected_groups,), reindex=reindex, sort=sort
    )
    (groups,) = grps

    # do this *before* possible broadcasting below.
    # factorize_ has already taken care of offsetting
    if engine == "numbagg":
        seen_groups = _unique(group_idx)

    order = "C"
    if nax > 1:
        needs_broadcast = any(
            group_idx.shape[ax] != array.shape[ax] and group_idx.shape[ax] == 1 for ax in range(-nax, 0)
        )
        if needs_broadcast:
            # This is the dim=... case, it's a lot faster to ravel group_idx
            # in fortran order since group_idx is then sorted
            # I'm seeing 400ms -> 23ms for engine="flox"
            # Of course we are slower to ravel `array` but we avoid argsorting
            # both `array` *and* `group_idx` in _prepare_for_flox
            group_idx = np.broadcast_to(group_idx, array.shape[-by.ndim :])
            if engine == "flox":
                group_idx = group_idx.reshape(-1, order="F")
                order = "F"
    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - by.ndim] + (math.prod(array.shape[-by.ndim :]),)
    array = array.reshape(newshape, order=order)  # type: ignore[call-overload]
    group_idx = group_idx.reshape(-1)

    assert group_idx.ndim == 1

    empty = np.all(props.nanmask)
    hasnan = np.any(props.nanmask)

    results: IntermediateDict = {"groups": [], "intermediates": []}
    if reindex and expected_groups is not None:
        # TODO: what happens with binning here?
        results["groups"] = expected_groups.to_numpy()
    else:
        if empty:
            results["groups"] = np.array([np.nan])
        else:
            results["groups"] = groups

    # npg's argmax ensures that index of first "max" is returned assuming there
    # are many elements equal to the "max". Sorting messes this up totally.
    # so we skip this for argreductions
    if engine == "flox":
        # is_arg_reduction = any("arg" in f for f in func if isinstance(f, str))
        # if not is_arg_reduction:
        group_idx, array, _ = _prepare_for_flox(group_idx, array)

    final_array_shape += results["groups"].shape
    final_groups_shape += results["groups"].shape

    # we commonly have func=(..., "nanlen", "nanlen") when
    # counts are needed for the final result as well as for masking
    # optimize that out.
    previous_reduction: T_Func = ""
    for reduction, fv, kw, dt in zip(funcs, fill_values, kwargss, dtypes):
        # TODO: Figure out how to generalize this
        if reduction in ("quantile", "nanquantile"):
            new_dims_shape = tuple(dim.size for dim in quantile_new_dims_func(**kw) if not dim.is_scalar)
        elif reduction == "topk":
            new_dims_shape = tuple(dim.size for dim in topk_new_dims_func(**kw) if not dim.is_scalar)
        else:
            new_dims_shape = tuple()

        if empty:
            result = np.full(shape=new_dims_shape + final_array_shape, fill_value=fv)
        elif is_nanlen(reduction) and is_nanlen(previous_reduction):
            result = results["intermediates"][-1]
        else:
            # fill_value here is necessary when reducing with "offset" groups
            kw_func = dict(size=size, dtype=dt, fill_value=fv)
            kw_func.update(kw)

            if callable(reduction):
                # passing a custom reduction for npg to apply per-group is really slow!
                # So this `reduction` has to do the groupby-aggregation
                result = reduction(group_idx, array, **kw_func)
            else:
                result = generic_aggregate(
                    group_idx, array, axis=-1, engine=engine, func=reduction, **kw_func
                ).astype(dt, copy=False)
            if engine == "numbagg":
                result = _postprocess_numbagg(
                    result,
                    func=reduction,
                    size=size,
                    fill_value=fv,
                    # Unfortunately, we cannot reuse found_groups, it has not
                    # been "offset" and is really expected_groups in nearly all cases
                    seen_groups=seen_groups,
                )
            if hasnan:
                # remove NaN group label which should be last
                result = result[..., :-1]
            result = result.reshape(new_dims_shape + final_array_shape[:-1] + found_groups_shape)
        results["intermediates"].append(result)
        previous_reduction = reduction

    results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    return results


def _squeeze_results(results: IntermediateDict, axis: T_Axes) -> IntermediateDict:
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


def _finalize_results(
    results: IntermediateDict,
    agg: Aggregation,
    axis: T_Axes,
    expected_groups: pd.Index | None,
    reindex: bool,
) -> FinalResultsDict:
    """Finalize results by
    1. Squeezing out dummy dimensions
    2. Calling agg.finalize with intermediate results
    3. Mask using counts and fill with user-provided fill_value.
    4. reindex to expected_groups
    """
    squeezed = _squeeze_results(results, tuple(agg.num_new_vector_dims + ax for ax in axis))

    min_count = agg.min_count
    if min_count > 0:
        counts = squeezed["intermediates"][-1]
        squeezed["intermediates"] = squeezed["intermediates"][:-1]

    # finalize step
    finalized: FinalResultsDict = {}
    if agg.finalize is None:
        finalized[agg.name] = squeezed["intermediates"][0]
    else:
        finalized[agg.name] = agg.finalize(*squeezed["intermediates"], **agg.finalize_kwargs)

    fill_value = agg.fill_value["user"]
    if min_count > 0:
        count_mask = counts < min_count
        if count_mask.any():
            # For one count_mask.any() prevents promoting bool to dtype(fill_value) unless
            # necessary
            fill_value = fill_value or agg.fill_value[agg.name]
            if fill_value is None:
                raise ValueError("Filling is required but fill_value is None.")
            # This allows us to match xarray's type promotion rules
            if fill_value is xrdtypes.NA:
                new_dtype, fill_value = xrdtypes.maybe_promote(finalized[agg.name].dtype)
                finalized[agg.name] = finalized[agg.name].astype(new_dtype)
            finalized[agg.name] = np.where(count_mask, fill_value, finalized[agg.name])

    # Final reindexing has to be here to be lazy
    if not reindex and expected_groups is not None:
        finalized[agg.name] = reindex_(
            finalized[agg.name],
            squeezed["groups"],
            expected_groups,
            fill_value=fill_value,
        )
        finalized["groups"] = expected_groups.to_numpy()
    else:
        finalized["groups"] = squeezed["groups"]

    finalized[agg.name] = finalized[agg.name].astype(agg.dtype["final"], copy=False)
    return finalized


def _aggregate(
    x_chunk,
    combine: Callable,
    agg: Aggregation,
    expected_groups: pd.Index | None,
    axis: T_Axes,
    keepdims: bool,
    fill_value: Any,
    reindex: bool,
) -> FinalResultsDict:
    """Final aggregation step of tree reduction"""
    results = combine(x_chunk, agg, axis, keepdims, is_aggregate=True)
    return _finalize_results(results, agg, axis, expected_groups, reindex)


def _expand_dims(results: IntermediateDict) -> IntermediateDict:
    results["intermediates"] = tuple(np.expand_dims(array, DUMMY_AXIS) for array in results["intermediates"])
    return results


def _find_unique_groups(x_chunk) -> np.ndarray:
    from dask.base import flatten
    from dask.utils import deepmap

    unique_groups = _unique(np.asarray(tuple(flatten(deepmap(listify_groups, x_chunk)))))
    unique_groups = unique_groups[notnull(unique_groups)]

    if len(unique_groups) == 0:
        unique_groups = np.array([np.nan])
    return unique_groups


def _simple_combine(
    x_chunk,
    agg: Aggregation,
    axis: T_Axes,
    keepdims: bool,
    reindex: bool,
    is_aggregate: bool = False,
) -> IntermediateDict:
    """
    'Simple' combination of blockwise results.

    1. After the blockwise groupby-reduce, all blocks contain a value for all possible groups,
       and are of the same shape; i.e. reindex must have been True
    2. _expand_dims was used to insert an extra axis DUMMY_AXIS
    3. Here we concatenate along DUMMY_AXIS, and then call the combine function along
       DUMMY_AXIS
    4. At the final aggregate step, we squeeze out DUMMY_AXIS
    """
    from dask.array.core import deepfirst
    from dask.utils import deepmap

    if not reindex:
        # We didn't reindex at the blockwise step
        # So now reindex before combining by reducing along DUMMY_AXIS
        unique_groups = _find_unique_groups(x_chunk)
        x_chunk = deepmap(
            partial(reindex_intermediates, agg=agg, unique_groups=unique_groups),
            x_chunk,
        )
    else:
        unique_groups = deepfirst(x_chunk)["groups"]

    results: IntermediateDict = {"groups": unique_groups}
    results["intermediates"] = []
    axis_ = axis[:-1] + (DUMMY_AXIS,)
    for idx, combine in enumerate(agg.simple_combine):
        array = _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis_)
        assert array.ndim >= 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            assert callable(combine)
            result = combine(array, axis=axis_, keepdims=True)
        if is_aggregate:
            # squeeze out DUMMY_AXIS if this is the last step i.e. called from _aggregate
            result = result.squeeze(axis=DUMMY_AXIS)
        results["intermediates"].append(result)
    return results


def _conc2(x_chunk, key1, key2=slice(None), axis: T_Axes | None = None) -> np.ndarray:
    """copied from dask.array.reductions.mean_combine"""
    from dask.array.core import _concatenate2
    from dask.utils import deepmap

    mapped = deepmap(lambda x: x[key1][key2], x_chunk)
    return _concatenate2(mapped, axes=axis)

    # This doesn't seem to improve things at all; and some tests fail...
    # from dask.array.core import concatenate3
    # for _ in range(mapped[0].ndim-1):
    #    mapped = [mapped]
    # return concatenate3(mapped)


def reindex_intermediates(x: IntermediateDict, agg: Aggregation, unique_groups) -> IntermediateDict:
    new_shape = x["groups"].shape[:-1] + (len(unique_groups),)
    newx: IntermediateDict = {"groups": np.broadcast_to(unique_groups, new_shape)}
    newx["intermediates"] = tuple(
        reindex_(
            v,
            from_=np.atleast_1d(x["groups"].squeeze()),
            to=pd.Index(unique_groups),
            fill_value=f,
        )
        for v, f in zip(x["intermediates"], agg.fill_value["intermediate"])
    )
    return newx


def listify_groups(x: IntermediateDict):
    return list(np.atleast_1d(x["groups"].squeeze()))


def _grouped_combine(
    x_chunk,
    agg: Aggregation,
    axis: T_Axes,
    keepdims: bool,
    engine: T_Engine,
    is_aggregate: bool = False,
    sort: bool = True,
) -> IntermediateDict:
    """Combine intermediates step of tree reduction."""
    from dask.utils import deepmap

    combine = agg.combine

    if isinstance(x_chunk, dict):
        # Only one block at final step; skip one extra groupby
        return x_chunk

    if len(axis) != 1:
        # when there's only a single axis of reduction, we can just concatenate later,
        # reindexing is unnecessary
        # I bet we can minimize the amount of reindexing for mD reductions too, but it's complicated
        unique_groups = _find_unique_groups(x_chunk)
        x_chunk = deepmap(
            partial(reindex_intermediates, agg=agg, unique_groups=unique_groups),
            x_chunk,
        )

    # these are negative axis indices useful for concatenating the intermediates
    neg_axis = tuple(range(-len(axis), 0))

    groups = _conc2(x_chunk, "groups", axis=neg_axis)

    if agg.reduction_type == "argreduce":
        # If "nanlen" was added for masking later, we need to account for that
        if agg.chunk[-1] == "nanlen":
            slicer = slice(None, -1)
        else:
            slicer = slice(None, None)

        # We need to send the intermediate array values & indexes at the same time
        # intermediates are (value e.g. max, index e.g. argmax, counts)
        array_idx = tuple(_conc2(x_chunk, key1="intermediates", key2=idx, axis=axis) for idx in (0, 1))

        # for a single element along axis, we don't want to run the argreduction twice
        # This happens when we are reducing along an axis with a single chunk.
        avoid_reduction = array_idx[0].shape[axis[0]] == 1
        if avoid_reduction:
            results: IntermediateDict = {
                "groups": groups,
                "intermediates": list(array_idx),
            }
        else:
            results = chunk_argreduce(
                array_idx,
                groups,
                # count gets treated specially next
                func=combine[slicer],  # type: ignore[arg-type]
                axis=axis,
                expected_groups=None,
                fill_value=agg.fill_value["intermediate"][slicer],
                dtype=agg.dtype["intermediate"][slicer],
                engine=engine,
                sort=sort,
            )

        if agg.chunk[-1] == "nanlen":
            counts = _conc2(x_chunk, key1="intermediates", key2=2, axis=axis)

            if avoid_reduction:
                results["intermediates"].append(counts)
            else:
                # sum the counts
                results["intermediates"].append(
                    chunk_reduce(
                        counts,
                        groups,
                        func="sum",
                        axis=axis,
                        expected_groups=None,
                        fill_value=(0,),
                        dtype=(np.intp,),
                        engine=engine,
                        sort=sort,
                        user_dtype=agg.dtype["user"],
                    )["intermediates"][0]
                )

    elif agg.reduction_type == "reduce":
        # Here we reduce the intermediates individually
        results = {"groups": None, "intermediates": []}
        for idx, (combine_, fv, dtype) in enumerate(
            zip(combine, agg.fill_value["intermediate"], agg.dtype["intermediate"])
        ):
            assert combine_ is not None
            array = _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis)
            if array.shape[-1] == 0:
                # all empty when combined
                results["intermediates"].append(np.empty(shape=(1,) * (len(axis) - 1) + (0,), dtype=dtype))
                results["groups"] = np.empty(shape=(1,) * (len(neg_axis) - 1) + (0,), dtype=groups.dtype)
            else:
                _results = chunk_reduce(
                    array,
                    groups,
                    func=combine_,
                    axis=axis,
                    expected_groups=None,
                    fill_value=(fv,),
                    dtype=(dtype,),
                    engine=engine,
                    sort=sort,
                    user_dtype=agg.dtype["user"],
                )
                results["intermediates"].append(*_results["intermediates"])
                results["groups"] = _results["groups"]
    return results


def _reduce_blockwise(
    array,
    by,
    agg: Aggregation,
    *,
    axis: T_Axes,
    expected_groups,
    fill_value: Any,
    engine: T_Engine,
    sort: bool,
    reindex: bool,
) -> FinalResultsDict:
    """
    Blockwise groupby reduction that produces the final result. This code path is
    also used for non-dask array aggregations.
    """
    # for pure numpy grouping, we just use npg directly and avoid "finalizing"
    # (agg.finalize = None). We still need to do the reindexing step in finalize
    # so that everything matches the dask version.
    agg.finalize = None

    assert agg.finalize_kwargs is not None
    finalize_kwargs_: tuple[dict[Any, Any], ...] = (agg.finalize_kwargs,) + ({},) + ({},)

    results = chunk_reduce(
        array,
        by,
        func=agg.numpy,
        axis=axis,
        expected_groups=expected_groups,
        # This fill_value should only apply to groups that only contain NaN observations
        # BUT there is funkiness when axis is a subset of all possible values
        # (see below)
        fill_value=agg.fill_value["numpy"],
        dtype=agg.dtype["numpy"],
        kwargs=finalize_kwargs_,
        engine=engine,
        sort=sort,
        reindex=reindex,
        user_dtype=agg.dtype["user"],
    )

    if _is_arg_reduction(agg):
        results["intermediates"][0] = np.unravel_index(results["intermediates"][0], array.shape)[-1]

    result = _finalize_results(results, agg, axis, expected_groups, reindex=reindex)
    return result


def _normalize_indexes(ndim: int, flatblocks: Sequence[int], blkshape: tuple[int, ...]) -> tuple:
    """
    .blocks accessor can only accept one iterable at a time,
    but can handle multiple slices.
    To minimize tasks and layers, we normalize to produce slices
    along as many axes as possible, and then repeatedly apply
    any remaining iterables in a loop.

    TODO: move this upstream
    """
    unraveled = np.unravel_index(flatblocks, blkshape)

    normalized: list[int | slice | list[int]] = []
    for ax, idx in enumerate(unraveled):
        i = _unique(idx).squeeze()
        if i.ndim == 0:
            normalized.append(i.item())
        else:
            if len(i) == blkshape[ax] and np.array_equal(i, np.arange(blkshape[ax])):
                normalized.append(slice(None))
            elif _issorted(i) and np.array_equal(i, np.arange(i[0], i[-1] + 1)):
                start = None if i[0] == 0 else i[0]
                stop = i[-1] + 1
                stop = None if stop == blkshape[ax] else stop
                normalized.append(slice(start, stop))
            else:
                normalized.append(list(i))
    full_normalized = (slice(None),) * (ndim - len(normalized)) + tuple(normalized)

    # has no iterables
    noiter = list(i if not hasattr(i, "__len__") else slice(None) for i in full_normalized)
    # has all iterables
    alliter = {ax: i for ax, i in enumerate(full_normalized) if hasattr(i, "__len__")}

    mesh = dict(zip(alliter.keys(), np.ix_(*alliter.values())))  # type: ignore[arg-type, var-annotated]

    full_tuple = tuple(i if ax not in mesh else mesh[ax] for ax, i in enumerate(noiter))

    return full_tuple


def subset_to_blocks(
    array: DaskArray,
    flatblocks: Sequence[int],
    blkshape: tuple[int, ...] | None = None,
    reindexer=identity,
    chunks_as_array: tuple[np.ndarray, ...] | None = None,
) -> ArrayLayer:
    """
    Advanced indexing of .blocks such that we always get a regular array back.

    Parameters
    ----------
    array : dask.array
    flatblocks : flat indices of blocks to extract
    blkshape : shape of blocks with which to unravel flatblocks

    Returns
    -------
    dask.array
    """
    from dask.base import tokenize

    if blkshape is None:
        blkshape = array.blocks.shape

    if chunks_as_array is None:
        chunks_as_array = tuple(np.array(c) for c in array.chunks)

    index = _normalize_indexes(array.ndim, flatblocks, blkshape)

    # These rest is copied from dask.array.core.py with slight modifications
    index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)

    name = "groupby-cohort-" + tokenize(array, index)
    new_keys = array._key_array[index]

    squeezed = tuple(np.squeeze(i) if isinstance(i, np.ndarray) else i for i in index)
    chunks = tuple(tuple(c[i].tolist()) for c, i in zip(chunks_as_array, squeezed))

    keys = itertools.product(*(range(len(c)) for c in chunks))
    layer: Graph = {(name,) + key: (reindexer, tuple(new_keys[key].tolist())) for key in keys}
    return ArrayLayer(layer=layer, chunks=chunks, name=name)


def _extract_unknown_groups(reduced, dtype) -> tuple[DaskArray]:
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    groups_token = f"group-{reduced.name}"
    first_block = reduced.ndim * (0,)
    layer: Graph = {(groups_token, 0): (operator.getitem, (reduced.name, *first_block), "groups")}
    groups: tuple[DaskArray] = (
        dask.array.Array(
            HighLevelGraph.from_collections(groups_token, layer, dependencies=[reduced]),
            groups_token,
            chunks=((np.nan,),),
            meta=np.array([], dtype=dtype),
        ),
    )

    return groups


def _unify_chunks(array, by):
    from dask.array import from_array, unify_chunks

    inds = tuple(range(array.ndim))

    # Unifying chunks is necessary for argreductions.
    # We need to rechunk before zipping up with the index
    # let's always do it anyway
    if not is_duck_dask_array(by):
        # chunk numpy arrays like the input array
        # This removes an extra rechunk-merge layer that would be
        # added otherwise
        chunks = tuple(array.chunks[ax] if by.shape[ax] != 1 else (1,) for ax in range(-by.ndim, 0))

        by = from_array(by, chunks=chunks)
    _, (array, by) = unify_chunks(array, inds, by, inds[-by.ndim :])

    return array, by


def dask_groupby_agg(
    array: DaskArray,
    by: T_By,
    agg: Aggregation,
    expected_groups: pd.RangeIndex | None,
    axis: T_Axes = (),
    fill_value: Any = None,
    method: T_Method = "map-reduce",
    reindex: bool = False,
    engine: T_Engine = "numpy",
    sort: bool = True,
    chunks_cohorts=None,
) -> tuple[DaskArray, tuple[np.ndarray | DaskArray]]:
    import dask.array
    from dask.array.core import slices_from_chunks
    from dask.highlevelgraph import HighLevelGraph

    from .dask_array_ops import _tree_reduce

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"

    if expected_groups is None and reindex:
        raise ValueError
    if method == "cohorts":
        assert reindex is False

    by_input = by

    array, by = _unify_chunks(array, by)

    # tokenize here since by has already been hashed if its numpy
    token = dask.base.tokenize(array, by, agg, expected_groups, axis, method)

    # preprocess the array:
    #   - for argreductions, this zips the index together with the array block
    #   - not necessary for blockwise with argreductions
    #   - if this is needed later, we can fix this then
    if agg.preprocess and method != "blockwise":
        array = agg.preprocess(array, axis=axis)

    # 1. We first apply the groupby-reduction blockwise to generate "intermediates"
    # 2. These intermediate results are combined to generate the final result using a
    #    "map-reduce" or "tree reduction" approach.
    #    There are two ways:
    #    a. "_simple_combine": Where it makes sense, we tree-reduce the reduction,
    #        NOT the groupby-reduction for a speed boost. This is what xhistogram does (effectively),
    #        It requires that all blocks contain all groups after the initial blockwise step (1) i.e.
    #        reindex=True, and we must know expected_groups
    #    b. "_grouped_combine": A more general solution where we tree-reduce the groupby reduction.
    #       This allows us to discover groups at compute time, support argreductions, lower intermediate
    #       memory usage (but method="cohorts" would also work to reduce memory in some cases)
    labels_are_unknown = is_duck_dask_array(by_input) and expected_groups is None
    do_grouped_combine = (
        _is_arg_reduction(agg)
        or labels_are_unknown
        or (_is_first_last_reduction(agg) and array.dtype.kind != "f")
    )
    do_simple_combine = not do_grouped_combine

    if method == "blockwise":
        #  use the "non dask" code path, but applied blockwise
        blockwise_method = partial(_reduce_blockwise, agg=agg, fill_value=fill_value, reindex=reindex)
    else:
        extra = {}
        if agg.name == "topk":
            extra["kwargs"] = (agg.finalize_kwargs, *(({},) * (len(agg.chunk) - 1)))
        # choose `chunk_reduce` or `chunk_argreduce`
        blockwise_method = partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,
            fill_value=agg.fill_value["intermediate"],
            dtype=agg.dtype["intermediate"],
            reindex=reindex,
            user_dtype=agg.dtype["user"],
            **extra,
        )
        if do_simple_combine:
            # Add a dummy dimension that then gets reduced over
            blockwise_method = tlz.compose(_expand_dims, blockwise_method)

    # apply reduction on chunk
    intermediate = dask.array.blockwise(
        partial(
            blockwise_method,
            axis=axis,
            expected_groups=expected_groups if reindex else None,
            engine=engine,
            sort=sort,
        ),
        # output indices are the same as input indices
        # Unlike xhistogram, we don't always know what the size of the group
        # dimension will be unless reindex=True
        inds,
        array,
        inds,
        by,
        inds[-by.ndim :],
        concatenate=False,
        dtype=array.dtype,  # this is purely for show
        meta=array._meta,
        align_arrays=False,
        name=f"{name}-chunk-{token}",
    )

    group_chunks: tuple[tuple[int | float, ...]]

    if method in ["map-reduce", "cohorts"]:
        combine: Callable[..., IntermediateDict] = (
            partial(_simple_combine, reindex=reindex)
            if do_simple_combine
            else partial(_grouped_combine, engine=engine, sort=sort)
        )

        tree_reduce = partial(
            dask.array.reductions._tree_reduce,
            name=f"{name}-simple-reduce",
            dtype=array.dtype,
            axis=axis,
            keepdims=True,
            concatenate=False,
        )
        aggregate = partial(_aggregate, combine=combine, agg=agg, fill_value=fill_value)

        # Each chunk of `reduced`` is really a dict mapping
        # 1. reduction name to array
        # 2. "groups" to an array of group labels
        # Note: it does not make sense to interpret axis relative to
        # shape of intermediate results after the blockwise call
        if method == "map-reduce":
            reduced = tree_reduce(
                intermediate,
                combine=partial(combine, agg=agg),
                aggregate=partial(aggregate, expected_groups=expected_groups, reindex=reindex),
            )
            if labels_are_unknown:
                groups = _extract_unknown_groups(reduced, dtype=by.dtype)
                group_chunks = ((np.nan,),)
            else:
                assert expected_groups is not None
                groups = (expected_groups.to_numpy(),)
                group_chunks = ((len(expected_groups),),)

        elif method == "cohorts":
            assert chunks_cohorts
            block_shape = array.blocks.shape[-len(axis) :]

            out_name = f"{name}-reduce-{method}-{token}"
            groups_ = []
            chunks_as_array = tuple(np.array(c) for c in array.chunks)
            dsk: Graph = {}
            for icohort, (blks, cohort) in enumerate(chunks_cohorts.items()):
                cohort_index = pd.Index(cohort)
                reindexer = (
                    partial(reindex_intermediates, agg=agg, unique_groups=cohort_index)
                    if do_simple_combine
                    else identity
                )
                subset = subset_to_blocks(intermediate, blks, block_shape, reindexer, chunks_as_array)
                dsk |= subset.layer  # type: ignore[operator]
                # now that we have reindexed, we can set reindex=True explicitlly
                _tree_reduce(
                    subset,
                    out_dsk=dsk,
                    name=out_name,
                    block_index=icohort,
                    axis=axis,
                    combine=partial(combine, agg=agg, reindex=do_simple_combine, keepdims=True),
                    aggregate=partial(
                        aggregate, expected_groups=cohort_index, reindex=do_simple_combine, keepdims=True
                    ),
                )
                # This is done because pandas promotes to 64-bit types when an Index is created
                # So we use the index to generate the return value for consistency with "map-reduce"
                # This is important on windows
                groups_.append(cohort_index.values)

            graph = HighLevelGraph.from_collections(out_name, dsk, dependencies=[intermediate])

            out_chunks = list(array.chunks)
            out_chunks[axis[-1]] = tuple(len(c) for c in chunks_cohorts.values())
            for ax in axis[:-1]:
                out_chunks[ax] = (1,)
            reduced = dask.array.Array(graph, out_name, out_chunks, meta=array._meta)

            groups = (np.concatenate(groups_),)
            group_chunks = (tuple(len(cohort) for cohort in groups_),)

    elif method == "blockwise":
        reduced = intermediate
        if reindex:
            if TYPE_CHECKING:
                assert expected_groups is not None
            # TODO: we could have `expected_groups` be a dask array with appropriate chunks
            # for now, we have a numpy array that is interpreted as listing all group labels
            # that are present in every chunk
            groups = (expected_groups,)
            group_chunks = ((len(expected_groups),),)
        else:
            # TODO: use chunks_cohorts here; hard because chunks_cohorts does not include all-NaN blocks
            #       but the array after applying the blockwise op; does. We'd have to insert a subsetting op.
            # Here one input chunk → one output chunks
            # find number of groups in each chunk, this is needed for output chunks
            # along the reduced axis
            # TODO: this logic is very specialized for the resampling case
            slices = slices_from_chunks(tuple(array.chunks[ax] for ax in axis))
            groups_in_block = tuple(_unique(by_input[slc]) for slc in slices)
            groups = (np.concatenate(groups_in_block),)
            ngroups_per_block = tuple(len(grp) for grp in groups_in_block)
            group_chunks = (ngroups_per_block,)
    else:
        raise ValueError(f"Unknown method={method}.")

    # Adjust output for any new dimensions added, example for multiple quantiles
    new_dims_shape = tuple(dim.size for dim in agg.new_dims if not dim.is_scalar)
    new_inds = tuple(range(-len(new_dims_shape), 0))
    out_inds = new_inds + inds[: -len(axis)] + (inds[-1],)
    output_chunks = new_dims_shape + reduced.chunks[: -len(axis)] + group_chunks
    new_axes = dict(zip(new_inds, new_dims_shape))

    if method == "blockwise" and len(axis) > 1:
        # The final results are available but the blocks along axes
        # need to be reshaped to axis=-1
        # I don't know that this is possible with blockwise
        # All other code paths benefit from an unmaterialized Blockwise layer
        reduced = _collapse_blocks_along_axes(reduced, axis, group_chunks)

    # Can't use map_blocks because it forces concatenate=True along drop_axes,
    result = dask.array.blockwise(
        _extract_result,
        out_inds,
        reduced,
        inds,
        adjust_chunks=dict(zip(out_inds, output_chunks)),
        dtype=agg.dtype["final"],
        key=agg.name,
        name=f"{name}-{token}",
        concatenate=False,
        new_axes=new_axes,
    )

    return (result, groups)


def cubed_groupby_agg(
    array: CubedArray,
    by: T_By,
    agg: Aggregation,
    expected_groups: pd.Index | None,
    axis: T_Axes = (),
    fill_value: Any = None,
    method: T_Method = "map-reduce",
    reindex: bool = False,
    engine: T_Engine = "numpy",
    sort: bool = True,
    chunks_cohorts=None,
) -> tuple[CubedArray, tuple[np.ndarray | CubedArray]]:
    import cubed
    import cubed.core.groupby

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    if method == "blockwise":
        assert by.ndim == 1
        assert expected_groups is not None

        def _reduction_func(a, by, axis, start_group, num_groups):
            # adjust group labels to start from 0 for each chunk
            by_for_chunk = by - start_group
            expected_groups_for_chunk = pd.RangeIndex(num_groups)

            axis = (axis,)  # convert integral axis to tuple

            blockwise_method = partial(
                _reduce_blockwise,
                agg=agg,
                axis=axis,
                expected_groups=expected_groups_for_chunk,
                fill_value=fill_value,
                engine=engine,
                sort=sort,
                reindex=reindex,
            )
            out = blockwise_method(a, by_for_chunk)
            return out[agg.name]

        num_groups = len(expected_groups)
        result = cubed.core.groupby.groupby_blockwise(
            array, by, axis=axis, func=_reduction_func, num_groups=num_groups
        )
        groups = (expected_groups.to_numpy(),)
        return (result, groups)

    else:
        inds = tuple(range(array.ndim))

        by_input = by

        # Unifying chunks is necessary for argreductions.
        # We need to rechunk before zipping up with the index
        # let's always do it anyway
        if not is_chunked_array(by):
            # chunk numpy arrays like the input array
            chunks = tuple(array.chunks[ax] if by.shape[ax] != 1 else (1,) for ax in range(-by.ndim, 0))

            by = cubed.from_array(by, chunks=chunks, spec=array.spec)
        _, (array, by) = cubed.core.unify_chunks(array, inds, by, inds[-by.ndim :])

        # Cubed's groupby_reduction handles the generation of "intermediates", and the
        # "map-reduce" combination step, so we don't have to do that here.
        # Only the equivalent of "_simple_combine" is supported, there is no
        # support for "_grouped_combine".
        labels_are_unknown = is_chunked_array(by_input) and expected_groups is None
        do_simple_combine = not _is_arg_reduction(agg) and not labels_are_unknown

        assert do_simple_combine
        assert method == "map-reduce"
        assert expected_groups is not None
        assert reindex is True
        assert len(axis) == 1  # one axis/grouping

        def _groupby_func(a, by, axis, intermediate_dtype, num_groups):
            blockwise_method = partial(
                _get_chunk_reduction(agg.reduction_type),
                func=agg.chunk,
                fill_value=agg.fill_value["intermediate"],
                dtype=agg.dtype["intermediate"],
                reindex=reindex,
                user_dtype=agg.dtype["user"],
                axis=axis,
                expected_groups=expected_groups,
                engine=engine,
                sort=sort,
            )
            out = blockwise_method(a, by)
            # Convert dict to one that cubed understands, dropping groups since they are
            # known, and the same for every block.
            return {f"f{idx}": intermediate for idx, intermediate in enumerate(out["intermediates"])}

        def _groupby_combine(a, axis, dummy_axis, dtype, keepdims):
            # this is similar to _simple_combine, except the dummy axis and concatenation is handled by cubed
            # only combine over the dummy axis, to preserve grouping along 'axis'
            dtype = dict(dtype)
            out = {}
            for idx, combine in enumerate(agg.simple_combine):
                field = f"f{idx}"
                out[field] = combine(a[field], axis=dummy_axis, keepdims=keepdims)
            return out

        def _groupby_aggregate(a, **kwargs):
            # Convert cubed dict to one that _finalize_results works with
            results = {"groups": expected_groups, "intermediates": a.values()}
            out = _finalize_results(results, agg, axis, expected_groups, reindex)
            return out[agg.name]

        # convert list of dtypes to a structured dtype for cubed
        intermediate_dtype = [(f"f{i}", dtype) for i, dtype in enumerate(agg.dtype["intermediate"])]
        dtype = agg.dtype["final"]
        num_groups = len(expected_groups)

        result = cubed.core.groupby.groupby_reduction(
            array,
            by,
            func=_groupby_func,
            combine_func=_groupby_combine,
            aggregate_func=_groupby_aggregate,
            axis=axis,
            intermediate_dtype=intermediate_dtype,
            dtype=dtype,
            num_groups=num_groups,
        )

        groups = (expected_groups.to_numpy(),)

        return (result, groups)


def _collapse_blocks_along_axes(reduced: DaskArray, axis: T_Axes, group_chunks) -> DaskArray:
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    nblocks = tuple(reduced.numblocks[ax] for ax in axis)
    output_chunks = reduced.chunks[: -len(axis)] + ((1,) * (len(axis) - 1),) + group_chunks

    # extract results from the dict
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    layer2: dict[tuple, tuple] = {}
    name = f"reshape-{reduced.name}"

    for ochunk in itertools.product(*ochunks):
        inchunk = ochunk[: -len(axis)] + np.unravel_index(ochunk[-1], nblocks)
        layer2[(name, *ochunk)] = (reduced.name, *inchunk)

    layer2: Graph
    return dask.array.Array(
        HighLevelGraph.from_collections(name, layer2, dependencies=[reduced]),
        name,
        chunks=output_chunks,
        dtype=reduced.dtype,
    )


def _extract_result(result_dict: FinalResultsDict, key) -> np.ndarray:
    from dask.array.core import deepfirst

    # deepfirst should be not be needed here but sometimes we receive a list of dict?
    return deepfirst(result_dict)[key]


def _validate_reindex(
    reindex: bool | None,
    func,
    method: T_MethodOpt,
    expected_groups,
    any_by_dask: bool,
    is_dask_array: bool,
    array_dtype: Any,
) -> bool | None:
    # logger.debug("Entering _validate_reindex: reindex is {}".format(reindex))  # noqa
    def first_or_last():
        return func in ["first", "last"] or (_is_first_last_reduction(func) and array_dtype.kind != "f")

    all_numpy = not is_dask_array and not any_by_dask
    if reindex is True and not all_numpy:
        if _is_arg_reduction(func):
            raise NotImplementedError
        if method == "cohorts" or (method == "blockwise" and not any_by_dask):
            raise ValueError("reindex=True is not a valid choice for method='blockwise' or method='cohorts'.")
        if first_or_last():
            raise ValueError("reindex must be None or False when func is 'first' or 'last.")

    if reindex is None:
        if method is None:
            # logger.debug("Leaving _validate_reindex: method = None, returning None")
            return None

        if all_numpy:
            return True

        if first_or_last():
            # have to do the grouped_combine since there's no good fill_value
            # Also needed for nanfirst, nanlast with no-NaN dtypes
            return False

        if method == "blockwise":
            # for grouping by dask arrays, we set reindex=True
            reindex = any_by_dask

        elif _is_arg_reduction(func):
            reindex = False

        elif method == "cohorts":
            reindex = False

        elif method == "map-reduce":
            if expected_groups is None and any_by_dask:
                reindex = False
            else:
                reindex = True

    assert isinstance(reindex, bool)
    # logger.debug("Leaving _validate_reindex: reindex is {}".format(reindex))  # noqa

    return reindex


def _assert_by_is_aligned(shape: tuple[int, ...], by: T_Bys) -> None:
    assert all(b.ndim == by[0].ndim for b in by[1:])
    for idx, b in enumerate(by):
        if not all(j in [i, 1] for i, j in zip(shape[-b.ndim :], b.shape)):
            raise ValueError(
                "`array` and `by` arrays must be 'aligned' "
                "so that such that by_ is broadcastable to array.shape[-by.ndim:] "
                "for every array `by_` in `by`. "
                "Either array.shape[-by_.ndim :] == by_.shape or the only differences "
                "should be size-1 dimensions in by_."
                f"Received array of shape {shape} but "
                f"array {idx} in `by` has shape {b.shape}."
            )


@overload
def _convert_expected_groups_to_index(
    expected_groups: tuple[None, ...], isbin: Sequence[bool], sort: bool
) -> tuple[None, ...]: ...


@overload
def _convert_expected_groups_to_index(
    expected_groups: T_ExpectTuple, isbin: Sequence[bool], sort: bool
) -> T_ExpectIndexTuple: ...


def _convert_expected_groups_to_index(
    expected_groups: T_ExpectOptTuple, isbin: Sequence[bool], sort: bool
) -> T_ExpectIndexOptTuple:
    out: list[T_ExpectIndexOpt] = []
    for ex, isbin_ in zip(expected_groups, isbin):
        if isinstance(ex, pd.IntervalIndex) or (isinstance(ex, pd.Index) and not isbin_):
            if sort:
                out.append(ex.sort_values())
            else:
                out.append(ex)
        elif ex is not None:
            if isbin_:
                out.append(pd.IntervalIndex.from_breaks(ex))
            else:
                if sort:
                    ex = np.sort(ex)
                out.append(pd.Index(ex))
        else:
            assert ex is None
            out.append(None)
    return tuple(out)


def _lazy_factorize_wrapper(*by: T_By, **kwargs) -> np.ndarray:
    group_idx, *_ = factorize_(by, **kwargs)
    return group_idx


def _factorize_multiple(
    by: T_Bys,
    expected_groups: T_ExpectIndexOptTuple,
    any_by_dask: bool,
    sort: bool = True,
) -> tuple[tuple[np.ndarray], tuple[np.ndarray, ...], tuple[int, ...]]:
    kwargs: FactorizeKwargs = dict(
        axes=(),  # always (), we offset later if necessary.
        expected_groups=expected_groups,
        fastpath=True,
        # This is the only way it makes sense I think.
        # reindex controls what's actually allocated in chunk_reduce
        # At this point, we care about an accurate conversion to codes.
        reindex=True,
        sort=sort,
    )
    if any_by_dask:
        import dask.array

        # unifying chunks will make sure all arrays in `by` are dask arrays
        # with compatible chunks, even if there was originally a numpy array
        inds = tuple(range(by[0].ndim))
        chunks, by_ = dask.array.unify_chunks(*itertools.chain(*zip(by, (inds,) * len(by))))

        group_idx = dask.array.map_blocks(
            _lazy_factorize_wrapper,
            *by_,
            chunks=tuple(chunks.values()),
            meta=np.array((), dtype=np.int64),
            **kwargs,
        )

        fg, gs = [], []
        for by_, expect in zip(by, expected_groups):
            if expect is None:
                if is_duck_dask_array(by_):
                    raise ValueError("Please provide expected_groups when grouping by a dask array.")

                found_group = pd.unique(by_.reshape(-1))
            else:
                found_group = expect.to_numpy()

            fg.append(found_group)
            gs.append(len(found_group))

        found_groups = tuple(fg)
        grp_shape = tuple(gs)
    else:
        kwargs["by"] = by
        group_idx, found_groups, grp_shape, *_ = factorize_(**kwargs)

    return (group_idx,), found_groups, grp_shape


@overload
def _validate_expected_groups(nby: int, expected_groups: None) -> tuple[None, ...]: ...


@overload
def _validate_expected_groups(nby: int, expected_groups: T_ExpectedGroups) -> T_ExpectTuple: ...


def _validate_expected_groups(nby: int, expected_groups: T_ExpectedGroupsOpt) -> T_ExpectOptTuple:
    if expected_groups is None:
        return (None,) * nby

    if nby == 1 and not isinstance(expected_groups, tuple):
        if isinstance(expected_groups, pd.Index | np.ndarray):
            return (expected_groups,)
        else:
            array = np.asarray(expected_groups)
            if np.issubdtype(array.dtype, np.integer):
                # preserve default dtypes
                # on pandas 1.5/2, on windows
                # when a list is passed
                array = array.astype(np.int64)
            return (array,)

    if nby > 1 and not isinstance(expected_groups, tuple):  # TODO: test for list
        raise ValueError(
            "When grouping by multiple variables, expected_groups must be a tuple "
            "of either arrays or objects convertible to an array (like lists). "
            "For example `expected_groups=(np.array([1, 2, 3]), ['a', 'b', 'c'])`."
            f"Received a {type(expected_groups).__name__} instead. "
            "When grouping by a single variable, you can pass an array or something "
            "convertible to an array for convenience: `expected_groups=['a', 'b', 'c']`."
        )

    if TYPE_CHECKING:
        assert isinstance(expected_groups, tuple)

    if len(expected_groups) != nby:
        raise ValueError(
            f"Must have same number of `expected_groups` (received {len(expected_groups)}) "
            f" and variables to group by (received {nby})."
        )

    return expected_groups


def _choose_method(
    method: T_MethodOpt, preferred_method: T_Method, agg: Aggregation, by, nax: int
) -> T_Method:
    if method is None:
        logger.debug("_choose_method: method is None")
        if agg.chunk == (None,):
            if preferred_method != "blockwise":
                raise ValueError(
                    f"Aggregation {agg.name} is only supported for `method='blockwise'`, "
                    "but the chunking is not right."
                )
            logger.debug("_choose_method: choosing 'blockwise'")
            return "blockwise"

        if nax != by.ndim:
            logger.debug("_choose_method: choosing 'map-reduce'")
            return "map-reduce"

        if _is_arg_reduction(agg) and preferred_method == "blockwise":
            return "cohorts"

        logger.debug(f"_choose_method: choosing preferred_method={preferred_method}")  # noqa
        return preferred_method
    else:
        return method


def _choose_engine(by, agg: Aggregation):
    dtype = agg.dtype["user"]

    not_arg_reduce = not _is_arg_reduction(agg)

    if agg.name in ["quantile", "nanquantile", "median", "nanmedian", "topk"]:
        logger.debug(f"_choose_engine: Choosing 'flox' since {agg.name}")
        return "flox"

    # numbagg only supports nan-skipping reductions
    # without dtype specified
    has_blockwise_nan_skipping = (agg.chunk[0] is None and "nan" in agg.name) or any(
        (isinstance(func, str) and "nan" in func) for func in agg.chunk
    )
    if HAS_NUMBAGG:
        if agg.name in ["all", "any"] or (not_arg_reduce and has_blockwise_nan_skipping and dtype is None):
            logger.debug("_choose_engine: Choosing 'numbagg'")
            return "numbagg"

    if not_arg_reduce and (not is_duck_dask_array(by) and _issorted(by)):
        logger.debug("_choose_engine: Choosing 'flox'")
        return "flox"
    else:
        logger.debug("_choose_engine: Choosing 'numpy'")
        return "numpy"


def groupby_reduce(
    array: np.ndarray | DaskArray,
    *by: T_By,
    func: T_Agg,
    expected_groups: T_ExpectedGroupsOpt = None,
    sort: bool = True,
    isbin: T_IsBins = False,
    axis: T_AxesOpt = None,
    fill_value=None,
    dtype: np.typing.DTypeLike = None,
    min_count: int | None = None,
    method: T_MethodOpt = None,
    engine: T_EngineOpt = None,
    reindex: bool | None = None,
    finalize_kwargs: dict[Any, Any] | None = None,
) -> tuple[DaskArray, Unpack[tuple[np.ndarray | DaskArray, ...]]]:
    """
    GroupBy reductions using tree reductions for dask.array

    Parameters
    ----------
    array : ndarray or DaskArray
        Array to be reduced, possibly nD
    *by : ndarray or DaskArray
        Array of labels to group over. Must be aligned with ``array`` so that
        ``array.shape[-by.ndim :] == by.shape`` or any disagreements in that
        equality check are for dimensions of size 1 in `by`.
    func : {"all", "any", "count", "sum", "nansum", "mean", "nanmean", \
            "max", "nanmax", "min", "nanmin", "argmax", "nanargmax", "argmin", "nanargmin", \
            "quantile", "nanquantile", "median", "nanmedian", "topk", "mode", "nanmode", \
            "first", "nanfirst", "last", "nanlast"} or Aggregation
        Single function name or an Aggregation instance
    expected_groups : (optional) Sequence
        Expected unique labels.
    isbin : bool, optional
        Are ``expected_groups`` bin edges?
    sort : bool, optional
        Whether groups should be returned in sorted order. Only applies for dask
        reductions when ``method`` is not ``"map-reduce"``. For ``"map-reduce"``, the groups
        are always sorted.
    axis : None or int or Sequence[int], optional
        If None, reduce across all dimensions of by
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim
    fill_value : Any
        Value to assign when a label in ``expected_groups`` is not present.
    dtype : data-type , optional
        DType for the output. Can be anything that is accepted by ``np.dtype``.
    min_count : int, default: None
        The required number of valid values to perform the operation. If
        fewer than min_count non-NA values are present the result will be
        NA. Only used if skipna is set to True or defaults to True for the
        array's dtype.
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
    engine : {"flox", "numpy", "numba", "numbagg"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
            This is the default choice because it works for most array types.
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
    reindex : bool, optional
        Whether to "reindex" the blockwise results to ``expected_groups`` (possibly automatically detected).
        If True, the intermediate result of the blockwise groupby-reduction has a value for all expected groups,
        and the final result is a simple reduction of those intermediates. In nearly all cases, this is a significant
        boost in computation speed. For cases like time grouping, this may result in large intermediates relative to the
        original block size. Avoid that by using ``method="cohorts"``. By default, it is turned off for argreductions.
    finalize_kwargs : dict, optional
        Kwargs passed to finalize the reduction such as ``ddof`` for var, std or ``q`` for quantile.

    Notes
    -----
    ``topk`` and ``quantile`` are implemented by converting to a complex number and so are limited to values between +-``2**53-1``
    i.e. the limit of a ``float64`` dtype. Offset your data appropriately if you need the larger range.

    Returns
    -------
    result
        Aggregated result
    *groups
        Group labels

    See Also
    --------
    xarray.xarray_reduce
    """

    if engine == "flox" and _is_arg_reduction(func):
        raise NotImplementedError(
            "argreductions not supported for engine='flox' yet. Try engine='numpy' or engine='numba' instead."
        )

    if engine == "numbagg" and dtype is not None:
        raise NotImplementedError(
            "numbagg does not support the `dtype` kwarg. Either cast your "
            "input arguments to `dtype` or use a different `engine`: "
            "'flox' or 'numpy' or 'numba'. "
            "See https://github.com/numbagg/numbagg/issues/121."
        )

    if func in ["quantile", "nanquantile"]:
        if finalize_kwargs is None or "q" not in finalize_kwargs:
            raise ValueError("Please pass `q` for quantile calculations.")
        else:
            nq = len(_atleast_1d(finalize_kwargs["q"]))
            if nq > 1 and engine == "numpy":
                raise ValueError(
                    "Multiple quantiles not supported with engine='numpy'."
                    "Use engine='flox' instead (it is also much faster), "
                    "or set engine=None to use the default."
                )
    if func == "topk" and (finalize_kwargs is None or "k" not in finalize_kwargs):
        raise ValueError("Please pass `k` in ``finalize_kwargs`` for topk calculations.")

    bys: T_Bys = tuple(np.asarray(b) if not is_duck_array(b) else b for b in by)
    nby = len(bys)
    by_is_dask = tuple(is_duck_dask_array(b) for b in bys)
    any_by_dask = any(by_is_dask)
    provided_expected = expected_groups is not None

    if engine == "numbagg" and _is_arg_reduction(func) and (any_by_dask or is_duck_dask_array(array)):
        # There is only one test that fails, but I can't figure
        # out why without deep debugging.
        # just disable for now.
        # test_groupby_reduce_axis_subset_against_numpy
        # for array is 3D dask, by is 3D dask, axis=2
        # We are falling back to numpy for the arg reduction,
        # so presumably something is going wrong
        raise NotImplementedError(
            "argreductions not supported for engine='numbagg' yet."
            "Try engine='numpy' or engine='numba' instead."
        )

    if method == "cohorts" and any_by_dask:
        raise ValueError(f"method={method!r} can only be used when grouping by numpy arrays.")

    if not is_duck_array(array):
        array = np.asarray(array)

    reindex = _validate_reindex(
        reindex,
        func,
        method,
        expected_groups,
        any_by_dask,
        is_duck_dask_array(array),
        array.dtype,
    )

    is_bool_array = np.issubdtype(array.dtype, bool) and not _is_bool_supported_reduction(func)
    array = array.astype(np.int_) if is_bool_array else array

    isbins = _atleast_1d(isbin, nby)

    _assert_by_is_aligned(array.shape, bys)

    expected_groups = _validate_expected_groups(nby, expected_groups)

    for idx, (expect, is_dask) in enumerate(zip(expected_groups, by_is_dask)):
        if is_dask and (reindex or nby > 1) and expect is None:
            raise ValueError(
                f"`expected_groups` for array {idx} in `by` cannot be None since it is a dask.array."
            )

    # We convert to pd.Index since that lets us know if we are binning or not
    # (pd.IntervalIndex or not)
    expected_groups = _convert_expected_groups_to_index(expected_groups, isbins, sort)

    # Don't factorize early only when
    # grouping by dask arrays, and not having expected_groups
    factorize_early = not (
        # can't do it if we are grouping by dask array but don't have expected_groups
        any(is_dask and ex_ is None for is_dask, ex_ in zip(by_is_dask, expected_groups))
    )
    expected_: pd.RangeIndex | None
    if factorize_early:
        bys, final_groups, grp_shape = _factorize_multiple(
            bys,
            expected_groups,
            any_by_dask=any_by_dask,
            sort=sort,
        )
        expected_ = pd.RangeIndex(math.prod(grp_shape))
    else:
        assert expected_groups == (None,)
        expected_ = None

    assert len(bys) == 1
    (by_,) = bys

    if axis is None:
        axis_ = tuple(array.ndim + np.arange(-by_.ndim, 0))
    else:
        axis_ = normalize_axis_tuple(axis, array.ndim)
    nax = len(axis_)

    has_dask = is_duck_dask_array(array) or is_duck_dask_array(by_)
    has_cubed = is_duck_cubed_array(array) or is_duck_cubed_array(by_)

    is_first_last = _is_first_last_reduction(func)
    if is_first_last:
        if has_dask and nax != 1:
            raise ValueError(
                "For dask arrays: first, last, nanfirst, nanlast reductions are "
                "only supported along a single axis. Please reshape appropriately."
            )

        elif nax not in [1, by_.ndim]:
            raise ValueError(
                "first, last, nanfirst, nanlast reductions are only supported "
                "along a single axis or when reducing across all dimensions of `by`."
            )

    is_npdatetime = array.dtype.kind in "Mm"
    is_cftime = _contains_cftime_datetimes(array)
    requires_numeric = (
        (func not in ["count", "any", "all"] and not is_first_last)
        # Flox's count works with non-numeric and its faster than converting.
        or (func == "count" and engine != "flox")
        or (is_first_last and is_cftime)
    )
    if requires_numeric:
        if is_npdatetime:
            datetime_dtype = array.dtype
            array = array.view(np.int64)
        elif is_cftime:
            offset = array.min()
            array = datetime_to_numeric(array, offset, datetime_unit="us")

    if nax == 1 and by_.ndim > 1 and expected_ is None:
        # When we reduce along all axes, we are guaranteed to see all
        # groups in the final combine stage, so everything works.
        # This is not necessarily true when reducing along a subset of axes
        # (of by)
        # TODO: Does this depend on chunking of by?
        # For e.g., we could relax this if there is only one chunk along all
        # by dim != axis?
        raise NotImplementedError("Please provide ``expected_groups`` when not reducing along all axes.")

    assert nax <= by_.ndim
    if nax < by_.ndim:
        by_ = _move_reduce_dims_to_end(by_, tuple(-array.ndim + ax + by_.ndim for ax in axis_))
        array = _move_reduce_dims_to_end(array, axis_)
        axis_ = tuple(array.ndim + np.arange(-nax, 0))
        nax = len(axis_)

    # When axis is a subset of possible values; then npg will
    # apply the fill_value to groups that don't exist along a particular axis (for e.g.)
    # since these count as a group that is absent. thoo!
    # fill_value applies to all-NaN groups as well as labels in expected_groups that are not found.
    #     The only way to do this consistently is mask out using min_count
    #     Consider np.sum([np.nan]) = np.nan, np.nansum([np.nan]) = 0
    if min_count is None:
        if nax < by_.ndim or (fill_value is not None and provided_expected):
            min_count_: int = 1
        else:
            min_count_ = 0
    else:
        min_count_ = min_count

    # TODO: set in xarray?
    if min_count_ > 0 and func in ["nansum", "nanprod"] and fill_value is None:
        # nansum, nanprod have fill_value=0, 1
        # overwrite than when min_count is set
        fill_value = np.nan

    kwargs = dict(axis=axis_, fill_value=fill_value)
    agg = _initialize_aggregation(func, dtype, array.dtype, fill_value, min_count_, finalize_kwargs)

    # Need to set this early using `agg`
    # It cannot be done in the core loop of chunk_reduce
    # since we "prepare" the data for flox.
    kwargs["engine"] = _choose_engine(by_, agg) if engine is None else engine

    groups: tuple[np.ndarray | DaskArray, ...]
    if has_cubed:
        if method is None:
            method = "map-reduce"

        if method not in ("map-reduce", "blockwise"):
            raise NotImplementedError(
                "Reduction for Cubed arrays is only implemented for methods 'map-reduce' and 'blockwise'."
            )

        partial_agg = partial(cubed_groupby_agg, **kwargs)

        result, groups = partial_agg(
            array=array,
            by=by_,
            expected_groups=expected_,
            agg=agg,
            reindex=bool(reindex),
            method=method,
            sort=sort,
        )

        return (result, groups)

    elif not has_dask:
        results = _reduce_blockwise(
            array,
            by_,
            agg,
            expected_groups=expected_,
            reindex=bool(reindex),
            sort=sort,
            **kwargs,
        )
        groups = (results["groups"],)
        result = results[agg.name]

    else:
        if TYPE_CHECKING:
            # TODO: How else to narrow that array.chunks is there?
            assert isinstance(array, DaskArray)

        if (not any_by_dask and method is None) or method == "cohorts":
            preferred_method, chunks_cohorts = find_group_cohorts(
                by_,
                [array.chunks[ax] for ax in range(-by_.ndim, 0)],
                expected_groups=expected_,
                # when provided with cohorts, we *always* 'merge'
                merge=(method == "cohorts"),
            )
        else:
            preferred_method = "map-reduce"
            chunks_cohorts = {}

        method = _choose_method(method, preferred_method, agg, by_, nax)

        if agg.chunk[0] is None and method != "blockwise":
            raise NotImplementedError(
                f"Aggregation {agg.name!r} is only implemented for dask arrays when method='blockwise'."
                f"Received method={method!r}"
            )

        if (
            _is_arg_reduction(agg)
            and method == "blockwise"
            and not all(nchunks == 1 for nchunks in array.numblocks[-nax:])
        ):
            raise NotImplementedError(
                "arg-reductions are not supported with method='blockwise', use 'cohorts' instead."
            )

        if nax != by_.ndim and method in ["blockwise", "cohorts"]:
            raise NotImplementedError(
                "Must reduce along all dimensions of `by` when method != 'map-reduce'."
                f"Received method={method!r}"
            )

        # TODO: clean this up
        reindex = _validate_reindex(
            reindex,
            func,
            method,
            expected_,
            any_by_dask,
            is_duck_dask_array(array),
            array.dtype,
        )

        if TYPE_CHECKING:
            assert method is not None

        # TODO: just do this in dask_groupby_agg
        # we always need some fill_value (see above) so choose the default if needed
        if kwargs["fill_value"] is None:
            kwargs["fill_value"] = agg.fill_value[agg.name]

        partial_agg = partial(dask_groupby_agg, **kwargs)

        # if preferred method is already blockwise, no need to rechunk
        if preferred_method != "blockwise" and method == "blockwise" and by_.ndim == 1:
            array = rechunk_for_blockwise(array, axis=-1, labels=by_)

        result, groups = partial_agg(
            array=array,
            by=by_,
            expected_groups=expected_,
            agg=agg,
            reindex=bool(reindex),
            method=method,
            chunks_cohorts=chunks_cohorts,
            sort=sort,
        )

        if sort and method != "map-reduce":
            assert len(groups) == 1
            sorted_idx = np.argsort(groups[0])
            # This optimization helps specifically with resampling
            if not _issorted(sorted_idx):
                result = result[..., sorted_idx]
                groups = (groups[0][sorted_idx],)

    if factorize_early:
        assert len(groups) == 1
        (groups_,) = groups
        # nan group labels are factorized to -1, and preserved
        # now we get rid of them by reindexing
        # First, for "blockwise", we can have -1 repeated in different blocks
        # This breaks the reindexing so remove those first.
        if method == "blockwise" and (mask := groups_ == -1).sum(axis=-1) > 1:
            result = result[..., ~mask]
            groups_ = groups_[..., ~mask]

        # This reindex also handles bins with no data
        result = reindex_(result, from_=groups_, to=expected_, fill_value=fill_value).reshape(
            result.shape[:-1] + grp_shape
        )
        groups = final_groups

    if is_bool_array and (_is_minmax_reduction(func) or _is_first_last_reduction(func)):
        result = result.astype(bool)

    # Output of count has an int dtype.
    if requires_numeric and func != "count":
        if is_npdatetime:
            result = result.astype(datetime_dtype)
        elif is_cftime:
            result = _to_pytimedelta(result, unit="us") + offset

    return (result, *groups)


def groupby_scan(
    array: np.ndarray | DaskArray,
    *by: T_By,
    func: T_Scan,
    expected_groups: T_ExpectedGroupsOpt = None,
    axis: int | tuple[int] = -1,
    dtype: np.typing.DTypeLike = None,
    method: T_MethodOpt = None,
    engine: T_EngineOpt = None,
) -> np.ndarray | DaskArray:
    """
    GroupBy reductions using parallel scans for dask.array

    Parameters
    ----------
    array : ndarray or DaskArray
        Array to be reduced, possibly nD
    *by : ndarray or DaskArray
        Array of labels to group over. Must be aligned with ``array`` so that
        ``array.shape[-by.ndim :] == by.shape`` or any disagreements in that
        equality check are for dimensions of size 1 in `by`.
    func : {"nancumsum", "ffill", "bfill"} or Scan
        Single function name or a Scan instance
    expected_groups : (optional) Sequence
        Expected unique labels.
    axis : None or int or Sequence[int], optional
        If None, reduce across all dimensions of by
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim.
    fill_value : Any
        Value to assign when a label in ``expected_groups`` is not present.
    dtype : data-type , optional
        DType for the output. Can be anything that is accepted by ``np.dtype``.
    method : {"blockwise", "cohorts"}, optional
        Strategy for reduction of dask arrays only:
          * ``"blockwise"``:
            Only scan using blockwise and avoid aggregating blocks
            together. Useful for resampling-style groupby problems where group
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
    engine : {"flox", "numpy", "numba", "numbagg"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
            This is the default choice because it works for most array types.
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

    Returns
    -------
    result
        Aggregated result

    See Also
    --------
    xarray.xarray_reduce
    """

    axis = _atleast_1d(axis)
    if len(axis) > 1:
        raise NotImplementedError("Scans are only supported along a single dimension.")

    bys: T_Bys = tuple(np.asarray(b) if not is_duck_array(b) else b for b in by)
    nby = len(by)
    by_is_dask = tuple(is_duck_dask_array(b) for b in bys)
    any_by_dask = any(by_is_dask)

    axis_ = normalize_axis_tuple(axis, array.ndim)

    if engine is not None:
        raise NotImplementedError("Setting `engine` is not supported for scans yet.")
    if method is not None:
        raise NotImplementedError("Setting `method` is not supported for scans yet.")
    if engine is None:
        engine = "flox"
    assert engine == "flox"

    if not is_duck_array(array):
        array = np.asarray(array)

    if isinstance(func, str):
        agg = AGGREGATIONS[func]
    assert isinstance(agg, Scan)
    agg = copy.deepcopy(agg)

    if (agg == AGGREGATIONS["ffill"] or agg == AGGREGATIONS["bfill"]) and array.dtype.kind != "f":
        # nothing to do, no NaNs!
        return array

    is_bool_array = np.issubdtype(array.dtype, bool)
    array = array.astype(np.int_) if is_bool_array else array

    if expected_groups is not None:
        raise NotImplementedError("Setting `expected_groups` and binning is not supported yet.")
    expected_groups = _validate_expected_groups(nby, expected_groups)
    expected_groups = _convert_expected_groups_to_index(expected_groups, isbin=(False,) * nby, sort=False)

    # Don't factorize early only when
    # grouping by dask arrays, and not having expected_groups
    factorize_early = not (
        # can't do it if we are grouping by dask array but don't have expected_groups
        any(is_dask and ex_ is None for is_dask, ex_ in zip(by_is_dask, expected_groups))
    )
    if factorize_early:
        bys, final_groups, grp_shape = _factorize_multiple(
            bys,
            expected_groups,
            any_by_dask=any_by_dask,
            sort=False,
        )
    else:
        raise NotImplementedError

    assert len(bys) == 1
    by_: np.ndarray
    (by_,) = bys
    has_dask = is_duck_dask_array(array) or is_duck_dask_array(by_)

    if array.dtype.kind in "Mm":
        cast_to = array.dtype
        array = array.view(np.int64)
    else:
        cast_to = None

    # TODO: move to aggregate_npg.py
    if agg.name in ["cumsum", "nancumsum"] and array.dtype.kind in ["i", "u"]:
        # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        # it defaults to the dtype of a, unless a
        # has an integer dtype with a precision less than that of the default platform integer.
        if array.dtype.kind == "i":
            agg.dtype = np.result_type(array.dtype, np.int_)
        elif array.dtype.kind == "u":
            agg.dtype = np.result_type(array.dtype, np.uint)
    else:
        agg.dtype = array.dtype if dtype is None else dtype

    (single_axis,) = axis_  # type: ignore[misc]
    # avoid some roundoff error when we can.
    if by_.shape[-1] == 1 or by_.shape == grp_shape:
        array = array.astype(agg.dtype)
        if cast_to is not None:
            array = array.astype(cast_to)
        return array

    # Made a design choice here to have `preprocess` handle both array and group_idx
    # Example: for reversing, we need to reverse the whole array, not just reverse
    #          each block independently
    inp = AlignedArrays(array=array, group_idx=by_)
    if agg.preprocess:
        inp = agg.preprocess(inp)

    if not has_dask:
        final_state = chunk_scan(inp, axis=single_axis, agg=agg, dtype=agg.dtype)
        result = _finalize_scan(final_state)
    else:
        result = dask_groupby_scan(inp.array, inp.group_idx, axes=axis_, agg=agg)

    # Made a design choice here to have `postprocess` handle both array and group_idx
    out = AlignedArrays(array=result, group_idx=by_)
    if agg.finalize:
        out = agg.finalize(out)

    if cast_to is not None:
        return out.array.astype(cast_to)
    return out.array


def chunk_scan(inp: AlignedArrays, *, axis: int, agg: Scan, dtype=None, keepdims=None) -> ScanState:
    assert axis == inp.array.ndim - 1

    # I don't think we need to re-factorize here unless we are grouping by a dask array
    accumulated = generic_aggregate(
        inp.group_idx,
        inp.array,
        axis=axis,
        engine="flox",
        func=agg.scan,
        dtype=dtype,
        fill_value=agg.identity,
    )
    result = AlignedArrays(array=accumulated, group_idx=inp.group_idx)
    return ScanState(result=result, state=None)


def grouped_reduce(inp: AlignedArrays, *, agg: Scan, axis: int, keepdims=None) -> ScanState:
    assert axis == inp.array.ndim - 1
    reduced = chunk_reduce(
        inp.array,
        inp.group_idx,
        func=(agg.reduction,),
        axis=axis,
        engine="flox",
        dtype=inp.array.dtype,
        fill_value=agg.identity,
        expected_groups=None,
    )
    return ScanState(
        state=AlignedArrays(array=reduced["intermediates"][0], group_idx=reduced["groups"]),
        result=None,
    )


def _zip(group_idx: np.ndarray, array: np.ndarray) -> AlignedArrays:
    return AlignedArrays(group_idx=group_idx, array=array)


def _finalize_scan(block: ScanState) -> np.ndarray:
    assert block.result is not None
    return block.result.array


def dask_groupby_scan(array, by, axes: T_Axes, agg: Scan) -> DaskArray:
    from dask.array import map_blocks
    from dask.array.reductions import cumreduction as scan

    from flox.aggregations import scan_binary_op

    if len(axes) > 1:
        raise NotImplementedError("Scans are only supported along a single axis.")
    (axis,) = axes

    array, by = _unify_chunks(array, by)

    # 1. zip together group indices & array
    zipped = map_blocks(
        _zip,
        by,
        array,
        dtype=array.dtype,
        meta=array._meta,
        name="groupby-scan-preprocess",
    )

    scan_ = partial(chunk_scan, agg=agg)
    # dask tokenizing error workaround
    scan_.__name__ = scan_.func.__name__  # type: ignore[attr-defined]

    # 2. Run the scan
    accumulated = scan(
        func=scan_,
        binop=partial(scan_binary_op, agg=agg),
        ident=agg.identity,
        x=zipped,
        axis=axis,
        # TODO: support method="sequential" here.
        method="blelloch",
        preop=partial(grouped_reduce, agg=agg),
        dtype=agg.dtype,
    )

    # 3. Unzip and extract the final result array, discard groups
    result = map_blocks(_finalize_scan, accumulated, dtype=agg.dtype)

    assert result.chunks == array.chunks

    return result
