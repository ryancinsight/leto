# ADR index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-elementwise-operator-overloading.md) | Elementwise operator overloading on `Array` | — |
| [0002](0002-coeus-rank-boundary.md) | Const-rank Coeus provider boundary | Accepted |
| [0003](0003-matrix-linalg-trait-layer.md) | Fluent rank-2 linear-algebra trait layer | — |
| [0004](0004-array-elementwise-operators.md) | Elementwise operator overloading on `Array` | — |
| [0005](0005-rank-revealing-svd.md) | Rank-revealing SVD via one-sided Jacobi | — |
| [0006](0006-nonsymmetric-eigensolver-track.md) | Non-symmetric eigensolver track (Hessenberg → Francis QR) | — |
| [0007](0007-dynamic-rank-boundary.md) | Dynamic rank (`IxDyn`) as a boundary carrier with a zero-copy rank bridge | — |
| [0008](0008-parity-scope-boundary.md) | Parity scope boundary — fixed-size storage, compile-time shape, geometry | — |
| [0009](0009-automatic-sparsity.md) | Automatic sparsity support (CSR, SpMV/SpMM, density dispatch) | — |
| [0010](0010-blocked-reflector-vectorization.md) | Blocked-reflector (compact-WY) vectorization for eig/SVD | — |
| [0011](0011-blocked-bidiagonalization.md) | Blocked bidiagonal reduction (`dgebrd`/`dlabrd`) for singular values | — |
| [0011](0011-num-complex-removal.md) | Atlas-native `Complex<T>`, removing the `num-complex` dependency | — |
| [0012](0012-dqds-values-only-singular-values.md) | Values-only singular values: dqds vs the bidiagonal Givens QR sweep | — |
| [0013](0013-provider-default-msrv.md) | Provider-default MSRV alignment | Accepted |
| [0014](0014-athena-cg-extraction.md) | Move conjugate gradient orchestration to Athena | — |
| [0015](0015-athena-gmres-extraction.md) | Move restarted GMRES orchestration to Athena | — |
| [0016](0016-typed-laplacian-stencil.md) | Own the typed Cartesian Laplacian stencil | — |
| [0017](0017-retire-ndarray-compatibility.md) | Retire public ndarray compatibility | — |
| [0018](0018-finite-difference-3d-extension.md) | Adopt the typed Cartesian 3-D finite-difference provider in leto-ops | — |
| [0019](0019-convolution-provider-contract.md) | Own CPU convolution in leto-ops | Accepted |
| [0020](0020-generic-zip-sources.md) | Generic tuple source sets for multi-input zips | Accepted |
| [—](sparse-support-design.md) | Sparse array support in Leto and Hephaestus | Accepted |
