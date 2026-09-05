# leto-python

NumPy-compatible array kernels and dense linear algebra for Python, backed by
the [Leto](https://github.com/ryancinsight/leto) Rust array substrate.

Every operation takes and returns ordinary NumPy arrays. The bindings convert
at the boundary, release the GIL around the computation, and raise
`ValueError` where Leto reports a domain error — so a call is a normal Python
call that happens to run in Rust.

## Install

```sh
pip install leto-python
```

Wheels are published for CPython 3.9 through 3.13 on Linux, Windows, and
macOS. NumPy is the only runtime requirement.

## Use

```python
import numpy as np
import leto_python as leto

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

print(leto.add(a, b))       # elementwise, float32
print(leto.matmul(a, b))    # dense matrix product, float32
print(leto.sum(a))          # 10.0

m = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
print(leto.det(m))          # 11.0
print(leto.inv(m))          # dense inverse, float64
print(leto.cholesky(m))     # lower factor of a symmetric positive definite m
```

## The dtype contract

It is not uniform, and reading it wrongly is the most likely first error:

| Group | Functions | dtype |
| --- | --- | --- |
| Elementwise and matrix product | `add`, `sub`, `mul`, `div`, `matmul`, `batched_matmul`, `dot`, `sum`, `sum_dyn` | `float32` |
| Linear algebra | `det`, `inv`, `solve`, `norm`, `trace`, `kron`, `eigenvalues`, `singular_values` | `float64` |
| Decompositions | `cholesky`, `cholesky_solve`, `cholesky_inv`, `qr`, `col_piv_qr`, `svd`, `symmetric_eigen`, `schur`, `hessenberg`, `bidiagonalize`, `full_piv_lu`, `udu`, `bunch_kaufman`, `matexp` | `float64` |

Inputs must be C-contiguous. A non-contiguous or wrongly typed array raises
rather than being silently copied or reinterpreted.

## What this package is

A binding surface and nothing else. It holds no algorithms: the numerics live
in the `leto` and `leto-ops` Rust crates, and this package converts types, maps
errors, and gets out of the way. That is deliberate — one implementation, one
place to verify it.

Correctness is checked against NumPy and SciPy on identical inputs. The
sign- and order-ambiguous decompositions (QR, Cholesky, eigendecomposition,
SVD) are checked by their mathematical invariants — reconstruction,
orthonormality, sorted spectra — because a factorization is unique only up to
signs and permutations, so raw factor equality would be the wrong oracle.

## Links

- [Source and issues](https://github.com/ryancinsight/leto)
- [Rust API documentation](https://docs.rs/leto)

## Licence

MIT or Apache-2.0, at your option.
