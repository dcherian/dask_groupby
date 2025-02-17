# Aggregations

`flox` implements all common reductions provided by `numpy_groupies` in `aggregations.py`. Control this by passing
the `func` kwarg:

- `"sum"`, `"nansum"`
- `"prod"`, `"nanprod"`
- `"count"` - number of non-NaN elements by group
- `"mean"`, `"nanmean"`
- `"var"`, `"nanvar"`
- `"std"`, `"nanstd"`
- `"argmin"`, `"nanargmax"`
- `"argmax"`, `"nanargmin"`
- `"first"`, `"nanfirst"`
- `"last"`, `"nanlast"`
- `"median"`, `"nanmedian"`
- `"mode"`, `"nanmode"`
- `"quantile"`, `"nanquantile"`
- `"topk"`

## Custom Reductions

`flox` also allows you to specify a custom Aggregation (again inspired by dask.dataframe),
though this might not be fully functional at the moment. See `aggregations.py` for examples.

See the ["Custom Aggregations"](user-stories/custom-aggregations.ipynb) user story for a more user-friendly example.

```python
mean = Aggregation(
    # name used for dask tasks
    name="mean",
    # operation to use for pure-numpy inputs
    numpy="mean",
    # blockwise reduction
    chunk=("sum", "count"),
    # combine intermediate results: sum the sums, sum the counts
    combine=("sum", "sum"),
    # generate final result as sum / count
    finalize=lambda sum_, count: sum_ / count,
    # Used when "reindexing" at combine-time
    fill_value=0,
    # Used when any member of `expected_groups` is not found
    final_fill_value=np.nan,
)
```

## Custom Scans

Coming soon!
