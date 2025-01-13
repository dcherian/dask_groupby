#!/usr/bin/env python3

import numpy as np
import pandas as pd
from asv_runner.benchmarks.mark import parameterize

import flox

Nsmall = 4
Nlarge = 2000


class Factorize:
    """Time the core factorize_ function."""

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @parameterize(
        {
            "expected": (None, (pd.Index([1, 3]),), (pd.RangeIndex(Nsmall),)),
            "reindex": [True, False],
            "sort": [True, False],
        }
    )
    def time_factorize_small(self, expected, reindex, sort):
        flox.core.factorize_(
            self.by_small,
            axes=(-1,),
            expected_groups=expected,
            reindex=reindex,
            sort=sort,
        )

    @parameterize(
        {
            "expected": (None, (pd.Index([1, 3]),), (pd.RangeIndex(Nsmall),)),
            "reindex": [True, False],
            "sort": [True, False],
        }
    )
    def time_factorize_large(self, expected, reindex, sort):
        flox.core.factorize_(
            self.by_large,
            axes=(-1,),
            expected_groups=None,
            reindex=reindex,
            sort=sort,
        )


class SingleGrouper1D(Factorize):
    def setup(self, *args, **kwargs):
        self.by_small = (np.repeat(np.arange(Nsmall), 250),)
        self.by_large = (np.random.permutation(np.arange(Nlarge)),)


class SingleGrouper3D(Factorize):
    def setup(self, *args, **kwargs):
        self.by_small = (np.broadcast_to(np.repeat(np.arange(Nsmall), 250), (5, 5, 1000)),)
        self.by_large = (np.broadcast_to(np.random.permutation(np.arange(Nlarge)), (5, 5, Nlarge)),)


# class Multiple(Factorize):
#     def setup(self, *args, **kwargs):
#         pass

# class CFTimeFactorize(Factorize):
#     pass
