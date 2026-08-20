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

- `sum_all` and `mean_all` return `Result<T, LetoError>` because an empty
  array has no well-defined mean. The example uses `expect` only after
  constructing a known non-empty fixture.
- `pearson_correlation` treats each column as a variable and each row as an
  observation. It returns the full correlation matrix.
- Perfectly linearly related columns (`y = 2x + 1`) yield `r = 1.0` because
  Pearson's coefficient measures linear dependence, not scale.
- The diagonal check is value-semantic: it would fail if the implementation
  returned a merely well-formed but incorrect matrix.
