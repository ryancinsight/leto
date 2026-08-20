# Reductions and Statistics

Reductions turn an array into a scalar or a lower-dimensional array. Leto's
core exposes `sum_all`, `mean_all`, `median_all`, `quantile_all`, and
`pearson_correlation` over borrowed views. Axis reductions and the generic
scalar kernels live in `leto-ops`.

## Empty and degenerate inputs

`sum_all` has a neutral element for supported numeric types, but `mean_all`
and the statistical operations need a non-empty, well-defined domain. They
return `Result` so an empty input or an invalid shape is visible to the caller.
The examples use `expect` only after constructing a known non-empty fixture;
library code should propagate the error with `?` or map it to its surrounding
typed error.

For a vector `x`, the mean is

`mean(x) = (1/n) * sum_i x_i`.

For a matrix, `mean_axis` keeps the reduced axis at length one. This shape
policy matches tensor consumers: reducing `[N, C]` along the column axis
produces `[N, 1]` rather than silently changing rank.

## Pearson correlation

For columns `x` and `y`, Pearson's coefficient is

`r(x,y) = sum_i (x_i-x_bar)(y_i-y_bar) /
          sqrt(sum_i (x_i-x_bar)^2 * sum_i (y_i-y_bar)^2)`.

`pearson_correlation` treats rows as observations and returns the full square
correlation matrix. A column is perfectly correlated with itself, and an
affine relation with positive scale has coefficient one. Degenerate zero
variance is handled by the provider's typed error or documented result rather
than by dividing by zero.

The independent checks in the example cover both the reduction value and the
correlation value. A test that only checks that the call returned `Ok` would
not detect a wrong traversal or a wrong normalization.
