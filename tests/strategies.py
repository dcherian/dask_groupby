from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cftime
import dask
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

from . import ALL_FUNCS, SCIPY_STATS_FUNCS

Chunks = tuple[tuple[int, ...], ...]

calendars = st.sampled_from(
    [
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "noleap",
        "365_day",
        "360_day",
        "julian",
        "all_leap",
        "366_day",
    ]
)


@st.composite
def units(draw, *, calendar: str) -> str:
    choices = ["days", "hours", "minutes", "seconds", "milliseconds", "microseconds"]
    if calendar == "360_day":
        choices += ["months"]
    elif calendar == "noleap":
        choices += ["common_years"]
    time_units = draw(st.sampled_from(choices))

    dt = draw(st.datetimes())
    year, month, day = dt.year, dt.month, dt.day
    if calendar == "360_day":
        day = min(day, 30)
    if calendar in ["360_day", "365_day", "noleap"] and month == 2 and day == 29:
        day = 28

    return f"{time_units} since {year}-{month}-{day}"


@st.composite
def cftime_arrays(
    draw: st.DrawFn,
    *,
    shape: st.SearchStrategy[tuple[int, ...]] = npst.array_shapes(),
    calendars: st.SearchStrategy[str] = calendars,
    elements: dict[str, Any] | None = None,
) -> np.ndarray[Any, Any]:
    if elements is None:
        elements = {}
    elements.setdefault("min_value", -10_000)
    elements.setdefault("max_value", 10_000)
    cal = draw(calendars)
    values = draw(npst.arrays(dtype=np.int64, shape=shape, elements=elements))
    unit = draw(units(calendar=cal))
    return cftime.num2date(values, units=unit, calendar=cal)


numeric_dtypes = (
    npst.integer_dtypes(endianness="=")
    | npst.unsigned_integer_dtypes(endianness="=")
    | npst.floating_dtypes(endianness="=", sizes=(32, 64))
    # TODO: add complex here not in supported_dtypes
)
numeric_like_dtypes = (
    npst.boolean_dtypes()
    | numeric_dtypes
    | npst.datetime64_dtypes(endianness="=")
    | npst.timedelta64_dtypes(endianness="=")
)
supported_dtypes = (
    numeric_like_dtypes
    | npst.unicode_string_dtypes(endianness="=")
    | npst.complex_number_dtypes(endianness="=")
)
by_dtype_st = supported_dtypes

NON_NUMPY_FUNCS = [
    "first",
    "last",
    "nanfirst",
    "nanlast",
    "count",
    "any",
    "all",
] + list(SCIPY_STATS_FUNCS)
SKIPPED_FUNCS = ["var", "std", "nanvar", "nanstd"]

func_st = st.sampled_from([f for f in ALL_FUNCS if f not in NON_NUMPY_FUNCS and f not in SKIPPED_FUNCS])
numeric_arrays = npst.arrays(
    elements={"allow_subnormal": False}, shape=npst.array_shapes(), dtype=numeric_dtypes
)
numeric_like_arrays = npst.arrays(
    elements={"allow_subnormal": False}, shape=npst.array_shapes(), dtype=numeric_like_dtypes
)
all_arrays = (
    npst.arrays(
        elements={"allow_subnormal": False},
        shape=npst.array_shapes(),
        dtype=numeric_like_dtypes,
    )
    | cftime_arrays()
)


def by_arrays(
    shape: st.SearchStrategy[tuple[int, ...]], *, elements: dict[str, Any] | None = None
) -> st.SearchStrategy[np.ndarray[Any, Any]]:
    if elements is None:
        elements = {}
    elements.setdefault("alphabet", st.characters(exclude_categories=["C"]))
    return st.one_of(
        npst.arrays(
            dtype=npst.integer_dtypes(endianness="=") | npst.unicode_string_dtypes(endianness="="),
            shape=shape,
            elements=elements,
        ),
        cftime_arrays(shape=shape, elements=elements),
    )


@st.composite
def chunks(draw: st.DrawFn, *, shape: tuple[int, ...]) -> Chunks:
    chunks = []
    for size in shape:
        if size > 1:
            nchunks = draw(st.integers(min_value=1, max_value=size - 1))
            dividers = sorted(
                set(draw(st.integers(min_value=1, max_value=size - 1)) for _ in range(nchunks - 1))
            )
            chunks.append(tuple(a - b for a, b in zip(dividers + [size], [0] + dividers)))
        else:
            chunks.append((1,))
    return tuple(chunks)


@st.composite
def chunked_arrays(
    draw: st.DrawFn,
    *,
    chunks: Callable[..., st.SearchStrategy[Chunks]] = chunks,
    arrays=all_arrays,
    from_array: Callable = dask.array.from_array,
) -> dask.array.Array:
    array = draw(arrays)
    chunks = draw(chunks(shape=array.shape))

    if array.dtype.kind in "cf":
        nan_idx = draw(
            st.lists(
                st.integers(min_value=0, max_value=array.shape[-1] - 1),
                max_size=array.shape[-1] - 1,
                unique=True,
            )
        )
        if nan_idx:
            array[..., nan_idx] = np.nan

    return from_array(array, chunks=chunks)
