# leto-ops

Math, reduction, SIMD, and parallel kernels for [`leto`](../leto/README.md)
arrays.

`leto-ops` is the compute half of the [Leto workspace](../../README.md). It
adds elementwise arithmetic, reductions, matrix multiplication, dense linear
algebra, sparse formats and solvers, and CPU attention on top of the layout and
storage types `leto` defines.

```rust
use leto::Array2;
use leto_ops::MatrixDecompose;

let a = Array2::<f64>::from_shape_fn([3, 3], |[i, j]| if i == j { 2.0 } else { 0.5 });
let qr = a.qr()?;
# Ok::<(), leto::LetoError>(())
```

## What is here

- One generic `binary_map::<Op, T, N>` traversal behind `add`, `sub`, `mul`,
  and `div`, with broadcasting into a caller-owned output shape.
- Native `f32`/`f64` paths route through Hermes SIMD, which runtime-dispatches
  AVX-512/AVX2/NEON with a scalar fallback. SIMD is not a build feature.
- Keep-dim axis reductions (`sum_axis`, `mean_axis`, `min_axis`, `max_axis`)
  and their caller-owned `*_into` forms, sharing one ZST-selected traversal.
- Unary math markers (`ExpOp`, `LnOp`, `SqrtOp`, …) over a `RealScalar` bound,
  plus `map_into` / `mapv` / `map` / `map_inplace`.
- `matmul` and `batched_matmul` writing into caller-owned output, over
  contiguous and strided/transposed inputs.
- Dense decompositions: LU (partial and full pivot), QR (Householder and
  column-pivoted), Cholesky, UDU, Bunch-Kaufman, Hessenberg, bidiagonalization,
  SVD (thin and rank-revealing), symmetric and Hermitian eigensolvers, the
  general non-symmetric `eigenvalues`, and real `schur`.
- Direct and iterative solvers: `solve`, `solve_least_squares`, `pinv`, plus
  `ConjugateGradient`, `BiCGSTAB`, `GMRES`, and `LsqrSolver` with Jacobi, SOR,
  SSOR, and ILU preconditioners.
- Sparse `CooMatrix` / `CsrMatrix` / `CscMatrix` with `spmv`, `spmm`, `spgemm`,
  and a sparse LU (`SparseLuSolver`).
- A fluent rank-2 trait layer (`MatrixProduct`, `MatrixNorm`,
  `MatrixDecompose`, `MatrixSolve`, `MatrixProperties`, `MatrixFunction`) whose
  methods are zero-cost delegators to the free-function kernels.

## Runnable examples

```sh
cargo run --locked -p leto-ops --example scalar_reference_parity
cargo run --locked -p leto-ops --example poisson_sparse_lu_accuracy
```

Both are deterministic, CI-safe accuracy harnesses checked against
analytically derived bounds, not benchmarks.

## Documentation

API docs: <https://docs.rs/leto-ops>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
