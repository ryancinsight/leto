# Example: Array Basics

**Crate**: `leto`
**Source**: `crates/leto/examples/book_array_basics.rs`

Construct `Array1` and `Array2` values, index into them, apply `mapv`, and
verify sums and structure with `sum_all`.

## Source

```rust
{{#include ../../../crates/leto/examples/book_array_basics.rs}}
```

## Output

```text
zeros(8) sum = 0
ones(8) sum  = 8
a = Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
a[3]*2 = 8
3×3 zeros sum = 0
eye(4) sum = 4
all array-basics assertions passed
```

## What to notice

- `Array1::zeros(8)` allocates a heap-backed array of 8 `f64` zeros.
  The shape is passed as a `usize`; `Array2::zeros((3, 3))` takes a tuple.

- `a[[3]]` is a rank-1 index using a one-element array literal.
  `eye[[0, 1]]` is a rank-2 index. Both return `T` by copy (no borrow).

- `mapv(|x| x * 2.0)` allocates a new array.  For in-place updates use
  `mapv_inplace(|x| x * 2.0)`.

- `Array2::eye(4)` produces the 4×4 identity; `sum_all` over it equals 4.0
  (the trace for an identity matrix).
