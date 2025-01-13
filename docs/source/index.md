# flox: fast & furious GroupBy reductions for dask, cubed, & Xarray!

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/xarray-contrib/flox/ci.yaml?branch=main&logo=github&style=flat)](https://github.com/xarray-contrib/flox/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/xarray-contrib/flox/main.svg)](https://results.pre-commit.ci/latest/github/xarray-contrib/flox/main)
[![image](https://img.shields.io/codecov/c/github/xarray-contrib/flox.svg?style=flat)](https://codecov.io/gh/xarray-contrib/flox)
[![Documentation Status](https://readthedocs.org/projects/flox/badge/?version=latest)](https://flox.readthedocs.io/en/latest/?badge=latest)

[![PyPI](https://img.shields.io/pypi/v/flox.svg?style=flat)](https://pypi.org/project/flox/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/flox.svg?style=flat)](https://anaconda.org/conda-forge/flox)

[![NASA-80NSSC18M0156](https://img.shields.io/badge/NASA-80NSSC18M0156-blue)](https://earthdata.nasa.gov/esds/competitive-programs/access/pangeo-ml)
[![NASA-80NSSC22K0345](https://img.shields.io/badge/NASA-80NSSC22K0345-blue)](https://science.nasa.gov/open-science-overview)

## Overview

`flox` mainly provides strategies for fast GroupBy reductions and scans with chunked arrays like [dask](<>) and [cubed](<>).
For reductions, `flox` uses a "tree reduction" to run the GroupBy operation in a parallel-native way totally avoiding a sort or shuffle operation. It was motivated by

1. Dask Dataframe GroupBy
   [blogpost](https://blog.dask.org/2019/10/08/df-groupby)
1. numpy_groupies in Xarray
   [issue](https://github.com/pydata/xarray/issues/4473)

See a presentation ([video](https://discourse.pangeo.io/t/november-17-2021-flox-fast-furious-groupby-reductions-with-dask-at-pangeo-scale/2016), [slides](https://docs.google.com/presentation/d/1YubKrwu9zPHC_CzVBhvORuQBW-z148BvX3Ne8XcvWsQ/edit?usp=sharing)) about this package, from the Pangeo Showcase.

## Why flox?

1. {py:func}`flox.groupby_reduce` [wraps](engines.md) the `numpy-groupies` package for performant Groupby reductions on nD arrays.
1. {py:func}`flox.groupby_reduce` provides [parallel-friendly strategies](implementation.md) for GroupBy reductions by wrapping `numpy-groupies` for dask arrays.
1. `flox` [integrates with xarray](xarray.md) to provide more performant Groupby and Resampling operations.
1. {py:func}`flox.xarray.xarray_reduce` [extends](xarray.md) Xarray's GroupBy operations allowing lazy grouping by dask arrays, grouping by multiple arrays,
   as well as combining categorical grouping and histogram-style binning operations using multiple variables.
1. `flox` also provides utility functions for rechunking both dask arrays and Xarray objects along a single dimension using the group labels as a guide:
   1. To rechunk for blockwise operations: {py:func}`flox.rechunk_for_blockwise`, {py:func}`flox.xarray.rechunk_for_blockwise`.
   1. To rechunk so that "cohorts", or groups of labels, tend to occur in the same chunks: {py:func}`flox.rechunk_for_cohorts`, {py:func}`flox.xarray.rechunk_for_cohorts`.

## Installing

```shell
$ pip install flox
```

```shell
$ conda install -c conda-forge flox
```

## Acknowledgements

This work was funded in part by

1. NASA-ACCESS 80NSSC18M0156 "Community tools for analysis of NASA Earth Observing System
   Data in the Cloud" (PI J. Hamman),
1. NASA-OSTFL 80NSSC22K0345 "Enhancing analysis of NASA data with the open-source Python Xarray Library" (PIs Scott Henderson, University of Washington;
   Deepak Cherian, NCAR; Jessica Scheick, University of New Hampshire), and
1. [NCAR's Earth System Data Science Initiative](https://ncar.github.io/esds/).

It was motivated by many discussions in the [Pangeo](https://pangeo.io) community.

## Contents

```{eval-rst}
.. toctree::
   :maxdepth: 1

   intro.md
   aggregations.md
   engines.md
   arrays.md
   implementation.md
   xarray.md
   user-stories.md
   api.rst
```
