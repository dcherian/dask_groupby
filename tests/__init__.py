import importlib
from contextlib import contextmanager
from distutils import version

import numpy as np
import pytest

try:
    import dask
    import dask.array as da
except ImportError:
    pass


try:
    import xarray as xr
except ImportError:
    pass


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


def LooseVersion(vstring):
    # Our development version is something like '0.10.9+aac7bfc'
    # This function just ignored the git commit id.
    vstring = vstring.split("+")[0]
    return version.LooseVersion(vstring)


has_dask, requires_dask = _importorskip("dask")
has_xarray, requires_xarray = _importorskip("xarray")


class CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    Reference: https://stackoverflow.com/questions/53289286/"""

    def __init__(self, max_computes=0):
        self.total_computes = 0
        self.max_computes = max_computes

    def __call__(self, dsk, keys, **kwargs):
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError(
                "Too many computes. Total: %d > max: %d." % (self.total_computes, self.max_computes)
            )
        return dask.get(dsk, keys, **kwargs)


@contextmanager
def dummy_context():
    yield None


def raise_if_dask_computes(max_computes=0):
    # return a dummy context manager so that this can be used for non-dask objects
    if not has_dask:
        return dummy_context()
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)


def assert_equal(a, b):
    __tracebackhide__ = True

    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    if (
        has_xarray
        and isinstance(a, (xr.DataArray, xr.Dataset))
        or isinstance(b, (xr.DataArray, xr.Dataset))
    ):
        xr.testing.assert_identical(a, b)
    elif has_dask and isinstance(a, da.Array) or isinstance(b, da.Array):
        # does some validation of the dask graph
        da.utils.assert_eq(a, b, equal_nan=True)
    else:
        np.testing.assert_allclose(a, b, equal_nan=True)
