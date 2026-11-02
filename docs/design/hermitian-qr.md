# Hermitian QR convergence repair [patch]

The public `hermitian_eigen_qr` entry retains its existing complex-double API.
Orders at most 32 still delegate to the unchanged Jacobi implementation.
Above that boundary, finite Hermitian input with a real diagonal is reduced
once to real symmetric tridiagonal form, then diagonalized by the existing
generic QL kernel. No dense QR factorization runs inside the iteration loop.
The private reduction, iteration and eigenvector composition execute in their
scalar type, with native-precision controls for all four real scalar formats.

## Numerical contract

Each self-adjoint reflector is `H = I - 2 u uᴴ`, with `uᴴu = 1` and a
leading complex phase chosen to avoid cancellation. A Hermitian rank-two
update computes the similarity; right multiplication accumulates the basis.
Diagonal unit phases turn the final complex tridiagonal edges real. With
`Qᴴ A Q = T` and the real QL vectors stored as rows of `Sᵀ`, output columns
are `V = Q S`, so the required identity is `A V = V diag(λ)`.

This is the reduction/iteration structure documented in Reference-LAPACK
v3.12.1 [ZHETD2](https://raw.githubusercontent.com/Reference-LAPACK/lapack/v3.12.1/SRC/zhetd2.f)
and [ZSTEQR](https://raw.githubusercontent.com/Reference-LAPACK/lapack/v3.12.1/SRC/zsteqr.f).
Our self-adjoint reflector convention differs from LAPACK's complex-tau
storage convention; their reflector formulas are not mixed.

The real symmetric caller retains its rounding-based deflation criterion and
arithmetic. The Hermitian caller selects an absolute policy: each deflated
edge is at most `tolerance / (2n)`. Removing both symmetric entries costs
`sqrt(2) |edge|`; summing those costs across at most `n-1` deflations bounds
the accumulated perturbation by the configured tolerance. At the final
allowed shift, the full remaining Frobenius norm plus previously discarded
cost can also satisfy this contract. It is tested before requiring another
shift. A zero shift budget therefore accepts a diagonal matrix and rejects
an unconverged one.

`off_diagonal_norm` is this conservative discarded-norm bound, not a measured
original-matrix residual and not a roundoff bound. Reduction/rotation roundoff
is assessed separately by `A V - V diag(λ)` and `Vᴴ V - I` assertions.
Power-of-two scaling transforms the absolute tolerance and restores both
results and error diagnostics to input units. An absolute policy may retain
edges below the relative rounding floor: its shift and closing correction
avoid overflow/underflow without substituting a looser convergence threshold.
Exhaustion reports the global residual bound and tolerance in the same units.
Reflectors and complex phases are normalized at their own column/component
scale; absolute-policy Givens ratios are formed before restoring a physical
radius. This preserves unitary transformations for minimum-subnormal
couplings beside normal diagonals. Global norms accumulate scaled squares,
so a sequence of subnormal edges cannot stagnate at one rounded edge norm.

Sorting enabled returns descending eigenpairs. With sorting disabled, the
iterative path is ascending and the diagonal shortcut retains input order.
The large-matrix condition estimate uses absolute spectral extrema. Invalid
tolerances, nonfinite input, imaginary diagonal entries and nonfinite or
unrepresentable results are errors; direct Jacobi semantics are unchanged.

## Executable evidence

At baseline `32d11f7ab6ef29ce4830486e457f48a1ccac85e2`, the 33-by-33 complex
Toeplitz matrix unitarily equivalent to `tridiag(1,4,1)` returned success after
128 allowed steps, reporting 127. Its largest absolute eigenvalue error was
1.2447575231383148 and relative residual was 0.4336969091139751. Zero-budget
non-diagonal input also returned success. The new tests require the analytic
eigenvalues `4 + 2 cos(kπ/34)`, small residual and unitary eigenvectors.

`tests/hermitian_qr_contract.rs` also preserves the exact ordered sinusoidal
Gram construction from [Kwavers #1012](https://github.com/ryancinsight/kwavers/commit/e3e0f2aae79d89e56f370ec986ccb9667ae1114c).
It checks eigenpairs and positive spectrum, with unchanged Jacobi as a
second numerical comparison. The author's reported 80% error and benchmark
duration are not independently established by these correctness tests.

Further controls cover a dense analytic rank-two 120-by-120 Gram spectrum,
repeated/indefinite/zero eigenvalues, complex phases, scales `2^-600` and
`2^600`, a `1e-280` block beside unit-scale entries, and a `1e-310` edge whose
naive shift ratio overflows. Zero/subnormal tolerances, exact-last-step
success, exhaustion, input rejection, sorting and unchanged Jacobi/real-QL
behavior are exercised without extending a test or solver iteration limit.

## Delivery boundary

This patch advances only `leto-ops` and its local workspace edge to 0.47.1.
Provider identities, core 0.47.1, private Python 0.46.2, and the unarmed
automatic publisher remain as on the base. The existing provider-first
package-release, independent-review and hosted gates still apply. Native
CPU numerical acceptance does not establish accelerator or downstream
Kwavers integration acceptance.
