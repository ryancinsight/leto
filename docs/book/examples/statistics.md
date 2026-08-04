# Example: Statistics

**Crate**: `leto`
**Source**: `crates/leto/examples/book_statistics.rs`

Compute `sum_all`, `mean_all`, and `pearson_correlation` over typed `Array1`
and `Array2` values.

## Source

```rust
{{#include ../../../crates/leto/examples/book_statistics.rs}}
```

## Output

```text
sum([1..5]) = 15
mean([1..5]) = 3
pearson(x, 2x+1) = 1.000000
pearson diagonal = 1.000000
all statistics assertions passed
```

## What to notice

- `sum_all` and `mean_all` return `Result<T, LetoError>` because an
  empty array has no well-defined mean.  The `?` / `.expect()` pattern
  is idiomatic.

- `pearson_correlation` takes a rank-2 array where each **column** is a
  variable and each **row** is an observation.  It returns the full N×N
  correlation matrix, so `corr[[0, 1]]` is the correlation between columns 0
  and 1.

- Perfectly linearly related columns (`y = 2x + 1`) yield `r = 1.0` because
  Pearson's `r` measures linear dependence, not scaling.

- `corr[[i, i]] = 1.0` always — each variable is perfectly correlated with
  itself.
