import numpy as np
import pandas as pd
from asv_runner.benchmarks.mark import parameterize, skip_for_params

import flox
import flox.aggregations

N = 3000
funcs = ["sum", "nansum", "mean", "nanmean", "max", "nanmax", "var", "count", "all"]
engines = ["flox", "numpy", "numbagg"]
expected_groups = {
    "None": None,
    "RangeIndex": pd.RangeIndex(5),
    "bins": pd.IntervalIndex.from_breaks([1, 2, 4]),
}
expected_names = tuple(expected_groups)

NUMBAGG_FUNCS = ["nansum", "nanmean", "nanmax", "count", "all"]

numbagg_skip = [
    (func, expected_names[0], "numbagg") for func in funcs if func not in NUMBAGG_FUNCS
] + [(func, expected_names[1], "numbagg") for func in funcs if func not in NUMBAGG_FUNCS]


def setup_jit():
    # pre-compile jitted funcs
    labels = np.ones((N), dtype=int)
    array1 = np.ones((N), dtype=float)
    array2 = np.ones((N, N), dtype=float)

    if "numba" in engines:
        for func in funcs:
            method = getattr(flox.aggregate_npg, func)
            method(labels, array1, engine="numba")
    if "numbagg" in engines:
        for func in set(NUMBAGG_FUNCS) & set(funcs):
            flox.groupby_reduce(array1, labels, func=func, engine="numbagg")
            flox.groupby_reduce(array2, labels, func=func, engine="numbagg")


class ChunkReduce:
    """Time the core reduction function."""

    min_run_count = 5
    warmup_time = 1

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @skip_for_params(numbagg_skip)
    @parameterize({"func": funcs, "expected_name": expected_names, "engine": engines})
    def time_reduce(self, func, expected_name, engine):
        flox.groupby_reduce(
            self.array,
            self.labels,
            func=func,
            engine=engine,
            axis=self.axis,
            expected_groups=expected_groups[expected_name],
        )

    @parameterize({"func": ["nansum", "nanmean", "nanmax", "count"], "engine": engines})
    def time_reduce_bare(self, func, engine):
        flox.aggregations.generic_aggregate(
            self.labels,
            self.array,
            axis=-1,
            size=5,
            func=func,
            engine=engine,
            fill_value=0,
        )

    @skip_for_params(numbagg_skip)
    @parameterize({"func": funcs, "expected_name": expected_names, "engine": engines})
    def peakmem_reduce(self, func, expected_name, engine):
        flox.groupby_reduce(
            self.array,
            self.labels,
            func=func,
            engine=engine,
            axis=self.axis,
            expected_groups=expected_groups[expected_name],
        )


class ChunkReduce1D(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N,))
        self.labels = np.repeat(np.arange(5), repeats=N // 5)
        self.axis = -1
        if "numbagg" in args:
            setup_jit()


class ChunkReduce1DUnsorted(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N,))
        self.labels = np.random.permutation(np.repeat(np.arange(5), repeats=N // 5))
        self.axis = -1
        setup_jit()


class ChunkReduce2D(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.repeat(np.arange(N // 5), repeats=5)
        self.axis = -1
        setup_jit()


class ChunkReduce2DUnsorted(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.random.permutation(np.repeat(np.arange(N // 5), repeats=5))
        self.axis = -1
        setup_jit()


class ChunkReduce2DAllAxes(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.repeat(np.arange(N // 5), repeats=5)
        self.axis = None
        setup_jit()


class ChunkReduce2DAllAxesUnsorted(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.random.permutation(np.repeat(np.arange(N // 5), repeats=5))
        self.axis = None
        setup_jit()
