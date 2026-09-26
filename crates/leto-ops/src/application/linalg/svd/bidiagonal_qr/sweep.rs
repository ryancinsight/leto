//! The shifted sweep: the Wilkinson shift, one implicit-shift Golub–Kahan step,
//! and the iteration driving the bidiagonal to diagonal form.

use super::super::triangular_pair::triangular_svd;
use super::deflation::{
    chase_negligible_diagonal_column, chase_negligible_diagonal_row, diagonal_is_negligible,
    Deflation,
};
use super::rotation::{givens, rotate_row_pair};
use super::zero_shift::zero_shift_sweep;
use crate::application::linalg::scaling::KernelWindow;
use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;
use leto::Result;

/// Iteration cap before declaring non-convergence (a safety bound; shifted QR
/// converges in `O(n)` sweeps).
const MAX_ITER: usize = 4000;

/// Wilkinson shift: the eigenvalue of the trailing 2×2 of `T = BᵀB` (rows
/// `q-1, q`) nearest the corner `T[q,q]`, from `dq = d[q]`, `dq1 = d[q-1]`,
/// `eq1 = e[q-1]`, `eq2 = e[q-2]` (zero when the block has two rows).
fn wilkinson_shift<T: RealScalar>(dq: T, dq1: T, eq1: T, eq2: T) -> T {
    let t11 = dq1.mul(dq1).add(eq2.mul(eq2));
    let t22 = dq.mul(dq).add(eq1.mul(eq1));
    let t12 = dq1.mul(eq1);

    let half = T::from_f64(0.5);
    // δ = (t11 − t22)/2; μ = t22 − sign(δ)·t12² / (|δ| + √(δ²+t12²)) — the
    // cancellation-avoiding form of "eigenvalue of the 2×2 nearest t22".
    let delta = t11.sub(t22).mul(half);
    let denom = delta.abs().add(delta.mul(delta).add(t12.mul(t12)).sqrt());
    if denom == T::ZERO {
        return t22;
    }
    let sign = if delta < T::ZERO {
        T::ONE.neg()
    } else {
        T::ONE
    };
    t22.sub(sign.mul(t12.mul(t12)).div(denom))
}

/// The kernel windows of one bidiagonal QR run, computed once per call.
#[derive(Clone, Copy)]
pub(super) struct SweepWindows<T> {
    /// [`givens`]: `a² + b² ≤ 2m²` (degree 2, bound `2¹`).
    pub(super) rotation: KernelWindow<T>,
    /// [`qr_step`]'s shift and first column: the degree-4 discriminant
    /// `≤ 2m⁴` kept normal and finite.
    shift: KernelWindow<T>,
}

impl<T: RealScalar> SweepWindows<T> {
    pub(super) fn new() -> Self {
        Self {
            rotation: KernelWindow::new(2, 2, 1),
            shift: KernelWindow::new(4, 4, 1),
        }
    }
}

/// Drive the bidiagonal `(d, e)` to diagonal form (all superdiagonals deflated).
///
/// When `VEC`, the left/right Givens rotations are accumulated into `u`
/// (`m × m`) and `v` (`n × n`); otherwise those updates are DCE'd
/// (`u`/`v` may be empty) — a zero-cost specialization for the values-only path.
pub(super) fn qr_iterate<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    k: usize,
    u: &mut [T],
    m: usize,
    v: &mut [T],
    n: usize,
) -> Result<()> {
    if k <= 1 {
        return Ok(());
    }
    let windows = SweepWindows::new();
    let deflation = Deflation::new(d, e, k);
    let mut q = k - 1;
    let mut iter = 0usize;
    loop {
        // Peel converged singular values off the bottom. Only the active
        // region near the bottom is touched — already-converged blocks above
        // are not re-scanned each iteration (LAPACK/leto `delimit_subproblem`).
        while q > 0 && e[q - 1] == T::ZERO {
            q -= 1;
        }
        if q == 0 {
            return Ok(());
        }
        // Top of the bottom-most unreduced block: scan up, splitting at the
        // first `|e| ≤ thresh` (`dbdsqr`'s scan); a split at the bottom itself
        // leaves `p = q`, which the bottom test below then peels.
        let mut p = q;
        while p > 0 {
            if e[p - 1].abs() <= deflation.thresh {
                e[p - 1] = T::ZERO;
                break;
            }
            p -= 1;
        }
        // A 2×2 block is diagonalized directly (`dbdsqr` with `dlasv2`): shifted
        // steps on it cycle when its smaller singular value is subnormal.
        if q == p + 1 {
            let pair = triangular_svd(d[p], e[p], d[q]);
            d[p] = pair.ssmax;
            e[p] = T::ZERO;
            d[q] = pair.ssmin;
            if VEC {
                rotate_row_pair(v, n, p, q, pair.csr, pair.snr); // V accumulated transposed
                rotate_row_pair(u, m, p, q, pair.csl, pair.snl); // U accumulated transposed
            }
            continue;
        }
        // `dbdsqr`'s relative convergence tests inside the block.
        if let Some(i) = deflation.forward_split(d, e, p, q) {
            e[i] = T::ZERO;
            continue;
        }

        iter += 1;
        if iter > MAX_ITER {
            return Err(leto::LetoError::StorageError {
                reason: "bidiagonal SVD QR failed to converge".to_string(),
            });
        }

        // A negligible diagonal inside the block is invisible to a shifted step:
        // with `d[i] = 0` the implicit `BᵀB` is singular, the Wilkinson shift
        // takes the nonzero eigenvalue, and the sweep drives the *other*
        // diagonal to zero as well while `|e|` is preserved — a fixed point at
        // `d = 0, e ≠ 0` that never satisfies the deflation test. Chase the row
        // (or the trailing column) out instead; both split the block, so each
        // fires at most once per index and the iteration always makes progress.
        if let Some(i) = (p..=q).find(|&i| diagonal_is_negligible(d, e, i, p, q)) {
            if i < q {
                chase_negligible_diagonal_row::<T, VEC>(d, e, i, q, u, m, windows.rotation);
            } else {
                chase_negligible_diagonal_column::<T, VEC>(d, e, p, q, v, n, windows.rotation);
            }
            continue;
        }

        if deflation.shift_ruins_accuracy(d, e, p, q, k) {
            zero_shift_sweep::<T, VEC>(d, e, p, q, u, m, v, n, windows.rotation);
        } else {
            qr_step::<T, VEC>(d, e, p, q, u, m, v, n, windows);
        }
    }
}

/// One implicit-shift Golub–Kahan SVD step on the block `d[p..=q]`, `e[p..q]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn qr_step<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    u: &mut [T],
    m: usize,
    v: &mut [T],
    n: usize,
    windows: SweepWindows<T>,
) {
    // First column of (BᵀB − μI), formed scale-safely (LAPACK `dbdsqr`'s
    // shift, with `dlas2`'s care for the squares): with `m` the largest of
    // the six entries it reads, `t11, t22 ≤ 2m²`, `|t12| ≤ m²`, `|δ| ≤ m²`,
    // so the shift's discriminant `δ² + t12² ≤ 2m⁴` (degree 4, bound 2¹) and
    // `|y| ≤ m² + 3m²`, `|z| ≤ m²` (degree 2). Unscaled while `m` keeps the
    // degree-4 term normal and finite (`safmin ≤ m⁴`, `2m⁴ ≤ Ω`): an
    // underflowed `t12²` degrades the Wilkinson shift to the Rayleigh shift
    // `t22`, which stagnates on a nearly equal trailing pair (probed: a
    // `Bf16` 2×2 at every exponent from `2⁻¹³⁰` to `2⁻³²`); otherwise the six entries are divided by the power of two bringing
    // `m` into `[1, 2)` — exact, and only `c, s` of the first rotation are
    // used (its `r` is discarded below), which are scale-invariant.
    let eq2 = if q >= p + 2 { e[q - 2] } else { T::ZERO };
    let local = [d[q], d[q - 1], e[q - 1], eq2, d[p], e[p]];
    let exponent = windows.shift.exponent(&local);
    let [dq, dq1, eq1, eq2, dp, ep] = local.map(|v| v.scale_binary(-exponent));
    let mu = wilkinson_shift(dq, dq1, eq1, eq2);
    // `dbdsqr`'s second zero-shift test, `(σ/|d_p|)² < ε`, in `BᵀB`'s units:
    // a shift negligible against `d_p²` only perturbs the step.
    if mu.abs() < thresholds::machine_epsilon::<T>().mul(dp.mul(dp)) {
        zero_shift_sweep::<T, VEC>(d, e, p, q, u, m, v, n, windows.rotation);
        return;
    }
    let mut y = dp.mul(dp).sub(mu);
    let mut z = dp.mul(ep);

    for k in p..q {
        // Right rotation (mixes columns k, k+1) annihilating z → accumulate V.
        let (c, s, r_right) = givens(y, z, windows.rotation);
        if VEC {
            rotate_row_pair(v, n, k, k + 1, c, s); // V accumulated transposed
        }
        if k > p {
            // c·y + s·z = √(y²+z²) = r_right (returned by `givens`, not recomputed).
            e[k - 1] = r_right;
        }
        let mut f = c.mul(d[k]).add(s.mul(e[k]));
        e[k] = c.mul(e[k]).sub(s.mul(d[k]));
        let bulge_col = s.mul(d[k + 1]);
        d[k + 1] = c.mul(d[k + 1]);
        d[k] = f;

        // Left rotation (mixes rows k, k+1) annihilating the bulge → accumulate U.
        let (c, s, r_left) = givens(d[k], bulge_col, windows.rotation);
        if VEC {
            rotate_row_pair(u, m, k, k + 1, c, s); // U accumulated transposed
        }
        // c·d[k] + s·bulge_col = √(d[k]²+bulge_col²) = r_left (not recomputed).
        d[k] = r_left;
        f = c.mul(e[k]).add(s.mul(d[k + 1]));
        d[k + 1] = c.mul(d[k + 1]).sub(s.mul(e[k]));
        e[k] = f;
        if k + 1 < q {
            let bulge_row = s.mul(e[k + 1]);
            e[k + 1] = c.mul(e[k + 1]);
            y = e[k];
            z = bulge_row;
        }
    }
}
