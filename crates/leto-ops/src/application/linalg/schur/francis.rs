//! Francis double-shift implicit QR iteration on a real upper-Hessenberg matrix,
//! accumulating the orthogonal similarity.
//!
//! Operates entirely in real arithmetic: complex eigenvalues surface as isolated
//! 2×2 diagonal blocks (standardized later), so no complex type is needed here.
//!
//! # Theorem (Francis step = one implicit double-shift QR step)
//! Let `H` be unreduced upper Hessenberg and `μ₁, μ₂` a shift pair that is either
//! two reals or a complex-conjugate pair. One Francis step computes an orthogonal
//! `Z` with `Zᵀ H Z` again upper Hessenberg, and `Zᵀ H Z` equals the matrix
//! produced by one explicit double-shift QR step — i.e. `Q` from the QR
//! factorization `(H − μ₁I)(H − μ₂I) = Q R`, applied as `Qᵀ H Q` — without ever
//! forming the product or leaving real arithmetic.
//!
//! *Proof (implicit-Q).* `M = (H − μ₁I)(H − μ₂I)` is real: a conjugate pair gives
//! `M = H² − (μ₁+μ₂)H + μ₁μ₂I` with real coefficients `s = μ₁+μ₂`, `t = μ₁μ₂`.
//! The step builds the Householder `P₀` mapping `M e₁` (its first column,
//! computed directly from `s, t` and the top-left of `H` — the `x, y, zz` below)
//! to a multiple of `e₁`, then forms `P₀ᵀ H P₀`, which bulges `H` just below the
//! subdiagonal, and chases the bulge with Householders `P₁ … P_{n-2}` that
//! restore Hessenberg form. Set `Z = P₀ P₁ … P_{n-2}`; then `Zᵀ H Z` is upper
//! Hessenberg and `Z e₁ = P₀ e₁ ∝ M e₁`. The implicit-Q theorem states that for
//! unreduced `H`, an orthogonal `Z` with `Zᵀ H Z` unreduced Hessenberg is
//! determined, up to column signs, by `Z e₁`. The explicit step's `Q` satisfies
//! `Q e₁ ∝ M e₁` as well (first column of `Q R = M`), so `Z` and `Q` agree up to
//! signs and yield the same Hessenberg form. Hence the bulge chase realizes the
//! double-shift QR step in real arithmetic. ∎
//!
//! # Corollary (convergence and deflation)
//! With Wilkinson-type shifts (eigenvalues of the trailing 2×2 block) the bottom
//! subdiagonal entry converges quadratically to zero; the iteration zeroes it
//! (LAPACK `dlahqr`'s small-subdiagonal test with the Ahues–Tisseur
//! refinement), splitting off a 1×1 (real eigenvalue) or 2×2 (conjugate pair)
//! block, and recurses on the leading submatrix. `dlahqr`'s exceptional
//! shifts every ten stalled iterations break the rare non-convergent cycles.
//!
//! # Theorem (eigenvalues-only within-block apply window — LAPACK `dlahqr`)
//! For the spectrum it suffices to apply each bulge-chasing reflector `Pₖ` only on
//! the window columns `[k, hi]` (left) and rows `[lo, k+len]` (right), provided the
//! annihilated bulge column `k−1` is set to its known image `(α, 0, 0)ᵀ`. The
//! eigenvalues read off the converged quasi-triangular `H` are unchanged.
//!
//! *Proof.* `H` stays similar to the original under every two-sided reflector, so
//! the spectrum is preserved regardless of which entries are stored. The
//! eigenvalues are read from the **diagonal blocks** only. An entry skipped by the
//! window is one of: (i) the bulge subdiagonal in column `k−1`, whose post-reflector
//! value is exactly `(α, 0, 0)` — written explicitly, so no information is lost; or
//! (ii) an entry with row `< lo` or column `> hi`, which is strictly above the
//! active diagonal block (`row < lo ≤ col` or `row ≤ hi < col`) and hence never
//! lies on a diagonal block, is never a shift source (shifts come from the trailing
//! 2×2 of `[lo, hi]`), and never enters the bulge band. Because `hi` is
//! non-increasing and `lo` is non-decreasing for fixed `hi` (deflation sets
//! `h[lo][lo−1]` to exact zero, a hard floor), such an entry is never read by a
//! future active block either. The window thus omits only never-read entries, so
//! the diagonal blocks — hence the eigenvalues — match the full sweep. ∎
//!
//! *Numerical note (evidence tier: differential + empirical).* The window reorders
//! the floating-point updates relative to a full sweep, so on a **defective**
//! eigenvalue (perturbation `O(√(ε‖A‖))`) the computed value can differ from a full
//! sweep — and from a backward-stable reference — by `O(√(ε‖A‖))`. This is within
//! backward stability, not an error; the eigenvalue battery asserts the derived
//! `8·√(ε‖A‖)` tolerance accordingly. The `ACCUMULATE_Q` (Schur) path applies
//! each reflector where `dlahqr`'s `WANTT` form does — columns `k ..` to the
//! right edge and rows `0 ..= min(k + 3, hi)` — because `T` and the Schur
//! vectors are outputs.

use crate::domain::real::RealScalar;
use bulge::{francis_step, step_window, StepWorkspace};
use deflation::{negligible_subdiagonal, run_floor};
use leto::{LetoError, Result};
use shift::Shift;

mod bulge;
mod deflation;
mod shift;

/// Iteration cap per deflation before declaring non-convergence (Wilkinson +
/// exceptional shifts converge in `O(n)` steps; this is a safety bound, and
/// the backward-error bounds of the test suite are taken at it).
const MAX_ITER: usize = 2000;

/// The subnormal part of the deflation threshold, `safmin`: a subdiagonal
/// driven into the subnormals, where no relative test can be met short of an
/// exact zero, still deflates. At most `n − 1` such deflations perturb by at
/// most `√n·safmin` jointly, which the matrix-tier gate (`schur/mod.rs`,
/// [`thresholds::deflation_count_log2`](crate::application::linalg::thresholds::deflation_count_log2))
/// keeps below `ε·‖A‖_F`.
pub(super) fn deflation_floor<T: RealScalar>() -> T {
    crate::application::linalg::thresholds::safe_min::<T>()
}

#[inline]
fn at<T: Copy>(h: &[T], i: usize, j: usize, n: usize) -> T {
    h[i * n + j]
}

/// Drive the Francis iteration to convergence: `h` becomes real
/// quasi-upper-triangular (real Schur form `T`) and `z` accumulates the
/// orthogonal similarity so that `H₀ = z T zᵀ`.
///
/// # Errors
/// [`LetoError::StorageError`] if a block fails to converge within [`MAX_ITER`].
pub(super) fn run<T: RealScalar, const ACCUMULATE_Q: bool>(
    h: &mut [T],
    z: &mut [T],
    n: usize,
) -> Result<()> {
    if n < 3 {
        return Ok(()); // 0/1: trivial; 2: a single block, standardized later.
    }
    // Reusable left-apply accumulator `w`, sized to the widest possible column
    // span (`n`); reused across every reflector so the hot path allocates once.
    let mut scratch_stack = [T::ZERO; 128];
    let mut scratch_vec = Vec::new();
    let scratch = if n <= 128 {
        &mut scratch_stack[..n]
    } else {
        scratch_vec.resize(n, T::ZERO);
        &mut scratch_vec[..]
    };
    let floor = run_floor(h, n);
    let ulp = crate::application::linalg::thresholds::machine_epsilon::<T>();
    let mut workspace = StepWorkspace {
        scratch,
        window: step_window(),
    };
    let mut hi = n - 1;
    let mut iter = 0usize;
    loop {
        // Bottom-most unreduced block: scan up while the subdiagonal is
        // non-negligible ([`negligible_subdiagonal`]).
        let mut lo = hi;
        while lo > 0 {
            if negligible_subdiagonal(h, n, lo, ulp, floor) {
                h[lo * n + (lo - 1)] = T::ZERO;
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            // 1×1 deflation (real eigenvalue).
            if hi == 0 {
                break;
            }
            hi -= 1;
            iter = 0;
            continue;
        }
        if lo == hi - 1 {
            // 2×2 deflation (real pair or complex conjugate pair).
            if lo == 0 {
                break;
            }
            hi = lo - 1;
            iter = 0;
            continue;
        }

        iter += 1;
        if iter > MAX_ITER {
            return Err(LetoError::StorageError {
                reason: "Schur QR iteration failed to converge".to_string(),
            });
        }
        francis_step::<T, ACCUMULATE_Q>(
            h,
            z,
            lo,
            hi,
            n,
            Shift::for_iteration(iter),
            &mut workspace,
        );
    }
    Ok(())
}
