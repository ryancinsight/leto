//! Deflation and splitting: `dbdsqr`'s relative convergence tests and the
//! chases that rotate a negligible diagonal out of the block.

use super::rotation::{givens, TransposedFactors};
use crate::application::linalg::scaling::KernelWindow;
use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;

/// The absolute deflation floor, `safmin`
/// ([`thresholds::deflation_count_log2`] derives it), in place of LAPACK
/// `dbdsqr`'s `MAXITR·(N·(N·UNFL))`, which in `F16` is `≈ 0.21` at `k = 24`
/// and would split superdiagonals of unit-scale matrices. An `eᵢ` driven into
/// the subnormals, where no relative test can be met short of an exact zero,
/// still deflates.
fn deflation_floor<T: RealScalar>() -> T {
    thresholds::safe_min::<T>()
}

/// LAPACK `dbdsqr`'s relative-accuracy deflation (`TOL ≥ 0`): `tol =
/// tolmul·ε`, `tolmul = max(10, min(100, ε^(−1/8)))`, and the split threshold
/// `thresh = max(tol·σ̃_min, floor)`, `σ̃_min` its lower estimate of the
/// smallest singular value over `√k` (the `SMINOA` recurrence).
#[derive(Clone, Copy)]
pub(super) struct Deflation<T> {
    tol: T,
    pub(super) thresh: T,
}

impl<T: RealScalar> Deflation<T> {
    pub(super) fn new(d: &[T], e: &[T], k: usize) -> Self {
        let eps = thresholds::machine_epsilon::<T>();
        let eighth_root = T::ONE.div(eps).sqrt().sqrt().sqrt();
        let (ten, hundred) = (T::from_usize(10), T::from_usize(100));
        let capped = if eighth_root < hundred {
            eighth_root
        } else {
            hundred
        };
        let tolmul = if capped > ten { capped } else { ten };
        let tol = tolmul.mul(eps);
        let mut sminoa = d[0].abs();
        let mut mu = sminoa;
        for i in 1..k {
            if sminoa == T::ZERO {
                break;
            }
            mu = d[i].abs().mul(mu.div(mu.add(e[i - 1].abs())));
            if mu < sminoa {
                sminoa = mu;
            }
        }
        let sminoa = sminoa.div(T::from_usize(k).sqrt());
        let relative = tol.mul(sminoa);
        let floor = deflation_floor::<T>();
        Self {
            tol,
            thresh: if relative > floor { relative } else { floor },
        }
    }

    /// `dbdsqr`'s first zero-shift test: `n·tol·(σ̃_min/σ_max) ≤ max(ε, tol/100)`
    /// — shifting would ruin the relative accuracy of the block's smallest
    /// singular value — with `σ̃_min` the forward recurrence's minimum over
    /// the block `[p, q]` and `σ_max` the largest `|dᵢ|, |eᵢ|` of the
    /// bidiagonal of order `k`.
    pub(super) fn shift_ruins_accuracy(
        self,
        d: &[T],
        e: &[T],
        p: usize,
        q: usize,
        k: usize,
    ) -> bool {
        let largest = d[..k].iter().chain(&e[..k - 1]).fold(T::ZERO, |acc, &x| {
            if x.abs() > acc {
                x.abs()
            } else {
                acc
            }
        });
        if largest == T::ZERO {
            return false;
        }
        let mut mu = d[p].abs();
        let mut smallest = mu;
        for i in p..q {
            mu = d[i + 1].abs().mul(mu.div(mu.add(e[i].abs())));
            if mu < smallest {
                smallest = mu;
            }
        }
        let eps = thresholds::machine_epsilon::<T>();
        let hundredth = self.tol.div(T::from_usize(100));
        let bound = if eps > hundredth { eps } else { hundredth };
        T::from_usize(k).mul(self.tol).mul(smallest.div(largest)) <= bound
    }

    /// `dbdsqr`'s forward convergence test on the block `[p, q]`: the bottom
    /// `|e_{q−1}| ≤ tol·|d_q|`, then the recurrence `μ ← |d_{i+1}|·μ/(μ + |eᵢ|)`
    /// from `μ = |d_p|`, splitting at the first `|eᵢ| ≤ tol·μ`. Returns the
    /// index of the superdiagonal it zeroes, if any.
    pub(super) fn forward_split(self, d: &[T], e: &[T], p: usize, q: usize) -> Option<usize> {
        if e[q - 1].abs() <= self.tol.mul(d[q].abs()) {
            return Some(q - 1);
        }
        let mut mu = d[p].abs();
        for i in p..q {
            if e[i].abs() <= self.tol.mul(mu) {
                return Some(i);
            }
            mu = d[i + 1].abs().mul(mu.div(mu.add(e[i].abs())));
        }
        None
    }
}

/// Is `d[i]` negligible against the superdiagonals adjoining it inside the block
/// `[p..=q]`? Precision-exact, in the same form as the `e` deflation test: the
/// entry is negligible exactly when adding it to its neighbours' magnitude does
/// not change that magnitude, i.e. `|d[i]| ≲ ulp(‖B‖ₗₒcₐₗ)`.
///
/// The scale is a *local* lower bound on `‖B‖` (one or two neighbours, never the
/// block norm), so this fires strictly less often than the standard
/// `|d| ≤ ε‖B‖` criterion, and only on values already below the rounding every
/// rotation in the sweep commits. It is never zero for an in-block index: the
/// deflation scan guarantees `e[q-1] ≠ 0`, and the splitting scan guarantees
/// `e[i] ≠ 0` for `p ≤ i < q`.
#[inline]
pub(super) fn diagonal_is_negligible<T: RealScalar>(
    d: &[T],
    e: &[T],
    i: usize,
    p: usize,
    q: usize,
) -> bool {
    let below = if i > p { e[i - 1].abs() } else { T::ZERO };
    let above = if i < q { e[i].abs() } else { T::ZERO };
    let scale = below.add(above);
    scale.add(d[i].abs()) == scale
}

/// Zero the superdiagonal `e[i]` sitting beside a negligible diagonal `d[i]`
/// (`p ≤ i < q`), splitting the block at `i`.
///
/// # Theorem (a zero diagonal row is removable by left rotations)
/// With `B[i,i] = 0`, row `i` of the block holds the single entry `e[i]` at
/// column `i+1`. Rotating rows `(j, i)` for `j = i+1 … q` — each chosen to
/// annihilate row `i`'s entry in column `j` against `d[j]` — leaves row `i`
/// entirely zero: each rotation kills the current entry and deposits the fill
/// `−s·e[j]` one column right, and the last one (`j = q`) has no column to its
/// right inside the block, so the fill exits. Row `i` zero means `d[i] = e[i] = 0`,
/// so `B` splits at `i` with `σ = 0` already isolated — no shift required, which
/// is exactly what a shifted step cannot achieve here. ∎
///
/// Left rotations touch `U` only (`B ← G B` ⟹ `Uᵀ ← G Uᵀ`), so `V` is untouched
/// and both factors stay orthogonal: the chase is a product of plane rotations.
pub(super) fn chase_negligible_diagonal_row<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    i: usize,
    q: usize,
    factors: &mut TransposedFactors<'_, T>,
    rotation: KernelWindow<T>,
) {
    d[i] = T::ZERO;
    let mut fill = e[i];
    e[i] = T::ZERO;
    for j in (i + 1)..=q {
        // Annihilate row `i`'s column-`j` entry against `d[j]`, keeping `d[j]`
        // as the surviving lead: the pair is ordered `(j, i)`.
        let (c, s, r) = givens(d[j], fill, rotation);
        if VEC {
            factors.rotate_left(j, i, c, s);
        }
        d[j] = r;
        if j < q {
            fill = s.mul(e[j]).neg();
            e[j] = c.mul(e[j]);
        }
    }
}

/// Zero the superdiagonal `e[q-1]` above a negligible **trailing** diagonal
/// `d[q]`, deflating `σ = 0` off the bottom of the block.
///
/// The transpose of `chase_negligible_diagonal_row`: with `B[q,q] = 0` the
/// last column of the block holds only `e[q-1]`, so column rotations `(j, q)`
/// for `j = q-1 … p` annihilate it against `d[j]` and chase the fill `−s·e[j-1]`
/// one row up per step, out of the top of the block. Column `q` ends zero, so
/// `d[q] = e[q-1] = 0` and the trailing zero deflates on the next pass.
///
/// Right rotations touch `V` only (`B ← B Gᵀ` ⟹ `Vᵀ ← G Vᵀ`); `U` is untouched.
pub(super) fn chase_negligible_diagonal_column<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    factors: &mut TransposedFactors<'_, T>,
    rotation: KernelWindow<T>,
) {
    d[q] = T::ZERO;
    let mut fill = e[q - 1];
    e[q - 1] = T::ZERO;
    let mut j = q - 1;
    loop {
        let (c, s, r) = givens(d[j], fill, rotation);
        if VEC {
            factors.rotate_right(j, q, c, s);
        }
        d[j] = r;
        if j == p {
            return;
        }
        fill = s.mul(e[j - 1]).neg();
        e[j - 1] = c.mul(e[j - 1]);
        j -= 1;
    }
}
