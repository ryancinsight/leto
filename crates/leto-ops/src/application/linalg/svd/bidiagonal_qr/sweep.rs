//! The shifted sweep: the Wilkinson shift, one implicit-shift Golub–Kahan step,
//! and the iteration driving the bidiagonal to diagonal form.

use super::super::triangular_pair::triangular_svd;
use super::chase::{Chase, Direction, Down, Oriented, Up};
use super::deflation::{
    chase_negligible_diagonal_column, chase_negligible_diagonal_row, diagonal_is_negligible,
    Deflation,
};
use super::rotation::{givens, TransposedFactors};
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
/// When `VEC`, the left/right Givens rotations are accumulated into
/// `factors`; otherwise those updates are DCE'd (`factors` may be empty) — a
/// zero-cost specialization for the values-only path.
///
/// Each block's bulge is chased from its larger end towards its smaller
/// (`dbdsqr`'s `IDIR`, chosen when the iteration moves to a block disjoint
/// from the previous one; Demmel & Kahan 1990, §5, "Chasing the bulge up or
/// down"), so a graded block converges its smallest singular value at the
/// small end and keeps it to high relative accuracy whichever end is small.
///
/// Returns the number of iterations run: sweeps (zero-shift or shifted)
/// and negligible-diagonal chases — an upper bound on the `K` of Demmel &
/// Kahan's accumulated relative error bound (Theorem 6: `69·K·n²·ε` to first
/// order for `K` zero-shift sweeps).
pub(super) fn qr_iterate<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    k: usize,
    factors: &mut TransposedFactors<'_, T>,
) -> Result<usize> {
    if k <= 1 {
        return Ok(0);
    }
    let windows = SweepWindows::new();
    let deflation = Deflation::new(d, e, k);
    let mut q = k - 1;
    let mut iter = 0usize;
    // The block the last sweep ran on and its chase direction (`dbdsqr`'s
    // `OLDLL`, `OLDM`, `IDIR`).
    let mut chased: Option<(usize, usize, Direction)> = None;
    loop {
        // Peel converged singular values off the bottom. Only the active
        // region near the bottom is touched — already-converged blocks above
        // are not re-scanned each iteration (LAPACK/leto `delimit_subproblem`).
        while q > 0 && e[q - 1] == T::ZERO {
            q -= 1;
        }
        if q == 0 {
            return Ok(iter);
        }
        // Top of the bottom-most unreduced block: scan up, splitting at the
        // first `|e| ≤ thresh` (`dbdsqr`'s scan). A split at the bottom itself
        // leaves `p = q`, a converged value the peel above takes next pass
        // (`dbdsqr`'s `M = M − 1`, LAPACK 3.12.0 `dbdsqr.f` lines 471–477).
        let mut p = q;
        while p > 0 {
            if e[p - 1].abs() <= deflation.thresh {
                e[p - 1] = T::ZERO;
                break;
            }
            p -= 1;
        }
        if p == q {
            continue;
        }
        // A 2×2 block is diagonalized directly (`dbdsqr` with `dlasv2`): shifted
        // steps on it cycle when its smaller singular value is subnormal.
        if q == p + 1 {
            let pair = triangular_svd(d[p], e[p], d[q]);
            d[p] = pair.ssmax;
            e[p] = T::ZERO;
            d[q] = pair.ssmin;
            if VEC {
                factors.rotate_right(p, q, pair.csr, pair.snr);
                factors.rotate_left(p, q, pair.csl, pair.snl);
            }
            continue;
        }
        // `dbdsqr` chooses the chase direction only on a block disjoint from
        // the previous one ("from larger end diagonal element towards
        // smaller", LAPACK 3.12.0 `dbdsqr.f` lines 507–522), so a block whose
        // ends reorder while it converges does not flip back and forth.
        let direction = match chased {
            Some((old_p, old_q, direction)) if p <= old_q && q >= old_p => direction,
            _ => Direction::of_block(d, p, q),
        };
        // `dbdsqr`'s relative convergence tests inside the block, run in the
        // chase direction (loop 100 down, lines 528–552; loop 110 up, lines
        // 556–580).
        let split = match direction {
            Direction::Down => deflation.split(&Oriented::<T, Down>::new(d, e, p, q), p, q),
            Direction::Up => deflation.split(&Oriented::<T, Up>::new(d, e, p, q), p, q),
        };
        if let Some(i) = split {
            e[i] = T::ZERO;
            continue;
        }
        chased = Some((p, q, direction));

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
                chase_negligible_diagonal_row::<T, VEC>(d, e, i, q, factors, windows.rotation);
            } else {
                chase_negligible_diagonal_column::<T, VEC>(d, e, p, q, factors, windows.rotation);
            }
            continue;
        }

        match direction {
            Direction::Down => sweep::<T, VEC, Down>(d, e, p, q, deflation, factors, windows),
            Direction::Up => sweep::<T, VEC, Up>(d, e, p, q, deflation, factors, windows),
        }
    }
}

/// One sweep on the block `[p, q]` in orientation `C`: the zero-shift sweep
/// when `dbdsqr`'s first zero-shift test says a shift would ruin the relative
/// accuracy of the block's smallest singular value, else the shifted step.
fn sweep<T: RealScalar, const VEC: bool, C: Chase>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    deflation: Deflation<T>,
    factors: &mut TransposedFactors<'_, T>,
    windows: SweepWindows<T>,
) {
    let ruins = deflation.shift_ruins_accuracy(&Oriented::<T, C>::new(d, e, p, q), p, q);
    if ruins {
        zero_shift_sweep::<T, VEC, C>(d, e, p, q, factors, windows.rotation);
    } else {
        qr_step::<T, VEC, C>(d, e, p, q, factors, windows);
    }
}

/// One implicit-shift Golub–Kahan SVD step on the block `d[p..=q]`, `e[p..q]`,
/// chased in orientation `C` (LAPACK 3.12.0 `dbdsqr.f` loop 140, lines
/// 703–745, and loop 150, lines 751–790):
/// the shift comes from the trailing 2×2 of the oriented block — for
/// [`Up`] its top, as `dbdsqr`'s `IDIR = 2` takes it.
pub(super) fn qr_step<T: RealScalar, const VEC: bool, C: Chase>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    factors: &mut TransposedFactors<'_, T>,
    windows: SweepWindows<T>,
) {
    let mut block = Oriented::<T, C>::new(d, e, p, q);
    // First column of (BᵀB − μI), formed scale-safely (LAPACK `dbdsqr`'s
    // shift, with `dlas2`'s care for the squares): with `m` the largest of
    // the six entries it reads, `t11, t22 ≤ 2m²`, `|t12| ≤ m²`, `|δ| ≤ m²`,
    // so the shift's discriminant `δ² + t12² ≤ 2m⁴` (degree 4, bound 2¹) and
    // `|y| ≤ m² + 3m²`, `|z| ≤ m²` (degree 2). Unscaled while `m` keeps the
    // degree-4 term normal and finite (`safmin ≤ m⁴`, `2m⁴ ≤ Ω`): an
    // underflowed `t12²` degrades the Wilkinson shift to the Rayleigh shift
    // `t22`, which stagnates on a nearly equal trailing pair (probed: a
    // `Bf16` 2×2 at every exponent from `2⁻¹³⁰` to `2⁻³²`); otherwise the
    // six entries are divided by the power of two bringing `m` into
    // `[1, 2)` — exact, and only `c, s` of the first rotation are used (its
    // `r` is discarded below), which are scale-invariant.
    let eq2 = if q >= p + 2 { block.e(q - 2) } else { T::ZERO };
    let local = [
        block.d(q),
        block.d(q - 1),
        block.e(q - 1),
        eq2,
        block.d(p),
        block.e(p),
    ];
    let exponent = windows.shift.exponent(&local);
    let [dq, dq1, eq1, eq2, dp, ep] = local.map(|v| v.scale_binary(-exponent));
    let mu = wilkinson_shift(dq, dq1, eq1, eq2);
    // `dbdsqr`'s second zero-shift test, `(σ/|d_p|)² < ε`, in `BᵀB`'s units:
    // a shift negligible against `d_p²` only perturbs the step.
    if mu.abs() < thresholds::machine_epsilon::<T>().mul(dp.mul(dp)) {
        zero_shift_sweep::<T, VEC, C>(d, e, p, q, factors, windows.rotation);
        return;
    }
    let mut y = dp.mul(dp).sub(mu);
    let mut z = dp.mul(ep);

    for k in p..q {
        // Column rotation (mixes columns k, k+1) annihilating z.
        let (c, s, r_right) = givens(y, z, windows.rotation);
        if VEC {
            block.rotate_columns(factors, k, k + 1, (c, s));
        }
        if k > p {
            // c·y + s·z = √(y²+z²) = r_right (returned by `givens`, not recomputed).
            block.set_e(k - 1, r_right);
        }
        let (dk, ek, dk1) = (block.d(k), block.e(k), block.d(k + 1));
        let f = c.mul(dk).add(s.mul(ek));
        let ek = c.mul(ek).sub(s.mul(dk));
        let bulge_col = s.mul(dk1);
        let dk1 = c.mul(dk1);

        // Row rotation (mixes rows k, k+1) annihilating the bulge.
        let (c, s, r_left) = givens(f, bulge_col, windows.rotation);
        if VEC {
            block.rotate_rows(factors, k, k + 1, (c, s));
        }
        // c·f + s·bulge_col = √(f²+bulge_col²) = r_left (not recomputed).
        block.set_d(k, r_left);
        block.set_e(k, c.mul(ek).add(s.mul(dk1)));
        block.set_d(k + 1, c.mul(dk1).sub(s.mul(ek)));
        if k + 1 < q {
            let ek1 = block.e(k + 1);
            y = block.e(k);
            z = s.mul(ek1);
            block.set_e(k + 1, c.mul(ek1));
        }
    }
}
