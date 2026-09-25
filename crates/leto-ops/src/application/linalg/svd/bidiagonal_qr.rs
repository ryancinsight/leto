//! Singular values via the implicit-shift bidiagonal QR algorithm
//! (Golub–Kahan / Golub–Reinsch), the accuracy-preserving alternative to the
//! Gram-matrix path.
//!
//! # Theorem (singular values of `A` = of its bidiagonal factor)
//! Golub–Kahan bidiagonalization gives orthogonal `U, V` with `A = U B Vᵀ` and
//! `B` upper bidiagonal. Orthogonal factors preserve singular values
//! (`σ(A) = σ(B)`), so it suffices to compute `σ(B)`. The implicit-shift QR
//! iteration applies, per sweep, a sequence of Givens rotations equivalent to one
//! shifted QR step of `BᵀB` **without ever forming `BᵀB`** (the "Golub–Kahan SVD
//! step"); the off-diagonal of the implicit `BᵀB` is driven to zero, so a
//! superdiagonal of `B` deflates and `B → diag(σ)`. Avoiding `BᵀB` keeps the
//! conditioning at `κ(A)` rather than `κ(A)² = κ(AᵀA)`, so small singular values
//! retain accuracy the Gram path loses. ∎
//!
//! The shifted step above assumes a nonsingular block. Exact rank deficiency
//! breaks that assumption — it puts an exact zero on the diagonal of `B`, where
//! the implicit `BᵀB` is singular and the sweep reaches a fixed point at
//! `d = 0`, `e ≠ 0` instead of deflating. That case is handled separately, by
//! rotating the offending row (or trailing column) out of the block; see
//! `chase_negligible_diagonal_row` and `chase_negligible_diagonal_column`.
//!
//! This module provides both the singular **values** (no `U`/`V` accumulation —
//! a zero-cost const-generic specialization) and the full thin SVD with `U`/`V`
//! (the rotations are accumulated into the bidiagonalization's orthogonal
//! factors). It is the sole SVD implementation, rank-deficient input included.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::triangular_pair::triangular_svd;
use super::{validate_input, SvdDecomposition};
use crate::application::linalg::scaling::{self, GateBound, KernelWindow};
use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, Result, Storage};

/// Iteration cap before declaring non-convergence (a safety bound; shifted QR
/// converges in `O(n)` sweeps).
const MAX_ITER: usize = 4000;

/// The matrix-tier gate's power-of-two bound `2^f` for the SVD family
/// (degree 1): with the kernels' products scale-safe (`givens`,
/// `qr_step`'s shift, the bidiagonal reflectors), every remaining
/// intermediate is a sum of entry-scale terms. The largest is a reflector
/// application's dot product `vᵀ·col` (`bidiagonal/colmajor.rs`,
/// `householder::apply_left`): `|vᵀ·col| ≤ ‖v‖₂·‖A‖_F`, with `‖v‖₂ ≤ 4√M`
/// (`M = max(rows, cols)`; the shared reflector's `v` has entries below 4
/// after its `[1, 2)` normalization, `householder::reflect_in_place`; the
/// column-major one has `|vᵢ| ≤ 1`) and `‖A‖_F ≤ 2^r·‖A‖_max`
/// ([`scaling::norm_ratio_log2`]). The rotation updates and deflation sums
/// stay below `2‖B‖ ≤ 2‖A‖_F` (`|c|, |s| ≤ 1`), inside the same bound.
///
/// The range the gate applies is degree 2, LAPACK `dgesvd`'s
/// `[√safmin/ε, ε/√safmin]` form (`scaling::balanced(matrix, 2, …)`): the
/// bound needs only degree 1, and `(Ω·2^−f)^½ ≤ Ω·2^−f` keeps it, but the
/// sweep needs headroom below `‖A‖_max` for the small singular values it
/// converges on. At the degree-1 lower end `smlnum` they sat within a few
/// binades of `safmin` in the narrow formats, where the shifted sweep lost
/// the relative precision it converges on and cycled (Bf16 skew-symmetric
/// tridiagonals at `2⁻¹¹⁴`).
///
/// Deflation floor: [`qr_iterate`] also deflates at or below
/// [`deflation_floor`]`(k)`, `k = min(rows, cols)`, so the gate's lower end is
/// raised to keep that floor below `ε·‖A‖_max`.
fn svd_bound<T: RealScalar>(rows: usize, cols: usize) -> impl FnOnce(&[T], T) -> GateBound {
    move |values, largest| {
        let half_log2_m = (thresholds::ceil_log2_count(rows.max(cols)) + 1) / 2;
        GateBound {
            factor_log2: 2 + half_log2_m + scaling::norm_ratio_log2(values, largest),
            floor_log2: deflation_floor_log2(rows.min(cols)),
        }
    }
}

/// `⌈log₂ k⌉`, the exponent of [`deflation_floor`] over `safmin`.
fn deflation_floor_log2(k: usize) -> i32 {
    thresholds::ceil_log2_count(k)
}

/// The absolute deflation floor of an order-`k` bidiagonal,
/// `2^⌈log₂ k⌉·safmin ≥ k·safmin`, in place of LAPACK `dbdsqr`'s
/// `maxitr·n²·unfl` (`maxitr = 6`). That constant assumes `n²·unfl ≪ ulp`,
/// which fails in `F16` (`6·24²·2⁻¹⁴ ≈ 0.2` at `k = 24`: it would split
/// superdiagonals of unit-scale matrices). The floor keeps its purpose — an
/// `eᵢ` driven into the subnormals, where no relative test can be met short
/// of an exact zero, still deflates — and the gate keeps it below
/// `ε·‖A‖_max`.
fn deflation_floor<T: RealScalar>(k: usize) -> T {
    thresholds::safe_min::<T>().scale_binary(deflation_floor_log2(k))
}

/// LAPACK `dbdsqr`'s relative-accuracy deflation (`TOL ≥ 0`): `tol =
/// tolmul·ε`, `tolmul = max(10, min(100, ε^(−1/8)))`, and the split threshold
/// `thresh = max(tol·σ̃_min, floor)`, `σ̃_min` its lower estimate of the
/// smallest singular value over `√k` (the `SMINOA` recurrence).
#[derive(Clone, Copy)]
struct Deflation<T> {
    tol: T,
    thresh: T,
}

impl<T: RealScalar> Deflation<T> {
    fn new(d: &[T], e: &[T], k: usize) -> Self {
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
        let floor = deflation_floor::<T>(k);
        Self {
            tol,
            thresh: if relative > floor { relative } else { floor },
        }
    }

    /// `dbdsqr`'s forward convergence test on the block `[p, q]`: the bottom
    /// `|e_{q−1}| ≤ tol·|d_q|`, then the recurrence `μ ← |d_{i+1}|·μ/(μ + |eᵢ|)`
    /// from `μ = |d_p|`, splitting at the first `|eᵢ| ≤ tol·μ`. Returns the
    /// index of the superdiagonal it zeroes, if any.
    fn forward_split(self, d: &[T], e: &[T], p: usize, q: usize) -> Option<usize> {
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

/// Singular values of a finite matrix, sorted descending, via bidiagonal QR.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite input, or QR
/// non-convergence.
pub fn singular_values<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    validate_input(matrix)?;
    let [rows, cols] = matrix.shape();
    if let Some((scaled, exponent)) = scaling::balanced(matrix, 2, svd_bound(rows, cols))? {
        let mut sigmas = singular_values_of_balanced(&scaled.view())?;
        scaling::restore(
            &mut sigmas,
            exponent,
            "singular value exceeds the scalar range",
        )?;
        return Ok(sigmas);
    }
    singular_values_of_balanced(matrix)
}

/// [`singular_values`] of a validated matrix whose entries are in range.
fn singular_values_of_balanced<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    let [rows, cols] = matrix.shape();

    // Bidiagonalization requires `m >= n`; σ(A) = σ(Aᵀ), so transpose wide input.
    // Uses the column-major working buffer (left reflector contiguous; local to
    // this path, no global layout change) and returns `(d, e)` directly.
    let (mut d, mut e) = if rows >= cols {
        crate::application::linalg::bidiagonal::bidiagonal_diag_colmajor(matrix)?
    } else {
        let transposed = transpose_to_owned(matrix)?;
        crate::application::linalg::bidiagonal::bidiagonal_diag_colmajor(&transposed.view())?
    };
    let k = rows.min(cols);

    let mut no_u: [T; 0] = [];
    let mut no_v: [T; 0] = [];
    qr_iterate::<T, false>(&mut d, &mut e, k, &mut no_u, 0, &mut no_v, 0)?;

    let mut sigmas: Vec<T> = d.into_iter().map(|x| x.abs()).collect();
    sigmas.sort_by(|a, b| {
        b.partial_cmp(a)
            .expect("singular values are finite (not NaN)")
    });
    Ok(sigmas)
}

/// Materialize `Aᵀ` (wide → tall) so the `m ≥ n` bidiagonalization applies.
fn transpose_to_owned<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    let mut values = vec![T::ZERO; rows * cols];
    if let Some(slice) = matrix.as_slice() {
        for i in 0..rows {
            let row = &slice[i * cols..i * cols + cols];
            for (j, &val) in row.iter().enumerate() {
                values[j * rows + i] = val;
            }
        }
    } else {
        for i in 0..rows {
            for j in 0..cols {
                values[j * rows + i] = *matrix.get([i, j])?;
            }
        }
    }
    Array2::from_shape_vec([cols, rows], values)
}

/// Givens rotation `(c, s, r)` with `c·a + s·b = r`, `−s·a + c·b = 0`.
///
/// Returns the rotated lead component `r = c·a + s·b` (the 2-norm `√(a²+b²)`
/// when `b ≠ 0`, else `a`) so the caller uses it directly instead of recomputing
/// `c·a + s·b` — `givens` already formed `√(a²+b²)` to normalize, so re-deriving
/// it at the call site is pure redundant arithmetic (the leto `cancel_y`
/// pattern, which returns the norm alongside the rotation).
///
/// Scale-safe as LAPACK `dlartg` (3.10, Anderson 2017): `a² + b² ≤ 2m²`,
/// `m = max(|a|, |b|)`, is formed unscaled while `m` lies in
/// `[√safmin, √(Ω/2)]` — `dlartg`'s `rtmin`/`rtmax` — and otherwise from
/// `a, b` divided by the power of two bringing `m` into `[1, 2)`. `c`, `s`
/// are scale-invariant and `r` is multiplied back, all exactly, so inside the
/// window the result is bit-for-bit the unscaled formula. `window` is
/// [`SweepWindows::rotation`].
#[inline]
fn givens<T: RealScalar>(a: T, b: T, window: KernelWindow<T>) -> (T, T, T) {
    if b == T::ZERO {
        return (T::ONE, T::ZERO, a);
    }
    let exponent = window.exponent(&[a, b]);
    let (a, b) = (a.scale_binary(-exponent), b.scale_binary(-exponent));
    // One reciprocal + two mults rather than two divides by the same `r`
    // (division is several× a multiply; the sweep calls this O(n²) times). The
    // extra rounding of `1/r` is within the SVD's differential tolerance.
    let r = a.mul(a).add(b.mul(b)).sqrt();
    let inv_r = T::ONE.div(r);
    (a.mul(inv_r), b.mul(inv_r), r.scale_binary(exponent))
}

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

/// Transpose a row-major `n × n` matrix into a fresh buffer.
fn transpose_square<T: RealScalar>(src: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = src[i * n + j];
        }
    }
    out
}

/// Apply the Givens `first' = c·first + s·second`, `second' = c·second − s·first`
/// to the two rows `(first, second)` of length `len` in a row-major matrix.
///
/// The pair need not be adjacent and need not be ordered (`first > second` is a
/// rotation with the roles swapped, which the zero-diagonal chase below needs);
/// each row is a contiguous `len`-slice either way.
///
/// # Theorem (transposed accumulation is bitwise-identical, and contiguous)
/// Accumulating `U` (or `V`) as its transpose `Uᵀ` while rotating two **rows** of
/// `Uᵀ` produces exactly the same factor `U` as rotating two **columns** of `U`,
/// bit for bit, and turns the strided column update into a contiguous one.
///
/// *Proof.* A plane rotation applied to columns `(k, k+1)` of `U` is the
/// right-multiplication `U ← U G` with `G` the embedded `2×2` Givens. Transposing,
/// `(U G)ᵀ = Gᵀ Uᵀ`, i.e. `Uᵀ ← Gᵀ Uᵀ`, which is a left-multiplication mixing
/// **rows** `(k, k+1)` of `Uᵀ` with the same scalar coefficients `(c, s)`. The two
/// updated entries in each position are the identical floating-point expression
/// `c·a + s·b` / `c·b − s·a` of the identical operands `a, b`, evaluated in the
/// identical order — so no rounding differs: the stored `Uᵀ` is the exact transpose
/// of the column-accumulated `U`. In row-major storage a row of `Uᵀ` is a
/// contiguous length-`len` slice, so the rotation is two contiguous disjoint
/// slices (cache-friendly, auto-vectorizable) instead of a stride-`len` column walk
/// (a cache line per element). The factors are recovered by reading rows of
/// `Uᵀ`/`Vᵀ` as columns of `U`/`V` at the `O(n²)` thin-extraction step — no
/// separate transpose pass — negligible against the `O(n³)` sweep. ∎
///
/// `U`/`V` are accumulated as `Uᵀ`/`Vᵀ` so every rotation hits this path.
#[inline]
fn rotate_row_pair<T: RealScalar>(
    mat: &mut [T],
    len: usize,
    first: usize,
    second: usize,
    c: T,
    s: T,
) {
    debug_assert!(first != second, "a Givens rotation mixes two distinct rows");
    let split = first.max(second);
    let (head, tail) = mat.split_at_mut(split * len);
    let low = &mut head[first.min(second) * len..][..len];
    let high = &mut tail[..len];
    let (row_a, row_b) = if first < second {
        (low, high)
    } else {
        (high, low)
    };
    for (a, b) in row_a.iter_mut().zip(row_b.iter_mut()) {
        let (va, vb) = (*a, *b);
        *a = c.mul(va).add(s.mul(vb));
        *b = c.mul(vb).sub(s.mul(va));
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
fn diagonal_is_negligible<T: RealScalar>(d: &[T], e: &[T], i: usize, p: usize, q: usize) -> bool {
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
fn chase_negligible_diagonal_row<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    i: usize,
    q: usize,
    u: &mut [T],
    m: usize,
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
            rotate_row_pair(u, m, j, i, c, s); // U accumulated transposed
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
fn chase_negligible_diagonal_column<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    v: &mut [T],
    n: usize,
    rotation: KernelWindow<T>,
) {
    d[q] = T::ZERO;
    let mut fill = e[q - 1];
    e[q - 1] = T::ZERO;
    let mut j = q - 1;
    loop {
        let (c, s, r) = givens(d[j], fill, rotation);
        if VEC {
            rotate_row_pair(v, n, j, q, c, s); // V accumulated transposed
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

/// The kernel windows of one bidiagonal QR run, computed once per call.
#[derive(Clone, Copy)]
struct SweepWindows<T> {
    /// [`givens`]: `a² + b² ≤ 2m²` (degree 2, bound `2¹`).
    rotation: KernelWindow<T>,
    /// [`qr_step`]'s shift and first column: the degree-4 discriminant
    /// `≤ 2m⁴` kept normal and finite.
    shift: KernelWindow<T>,
}

impl<T: RealScalar> SweepWindows<T> {
    fn new() -> Self {
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
fn qr_iterate<T: RealScalar, const VEC: bool>(
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

        qr_step::<T, VEC>(d, e, p, q, u, m, v, n, windows);
    }
}

/// One implicit-shift Golub–Kahan SVD step on the block `d[p..=q]`, `e[p..q]`.
#[allow(clippy::too_many_arguments)]
fn qr_step<T: RealScalar, const VEC: bool>(
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

/// Thin SVD `A = U Σ Vᵀ` for a **tall-or-square** matrix (`m ≥ n`) via the
/// implicit-shift bidiagonal QR with `U`/`V` accumulation.
///
/// Returns `(U, σ, V)` with `U` (`m × n`) and `V` (`n × n`) having orthonormal
/// columns and `σ` sorted descending (length `n`). Singular values are
/// non-negative (negative pivots are absorbed by flipping the matching `U`
/// column). `m ≥ n` is a precondition (the caller transposes wide input).
fn svd_tall<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<(Array2<T>, Vec<T>, Array2<T>)> {
    let [m, n] = matrix.shape();
    debug_assert!(m >= n, "svd_tall requires m >= n");
    let bidiag = crate::bidiagonalize(matrix)?;

    // Bidiagonalization factors `A = U_b B V_bᵀ`; accumulate the QR rotations on
    // top of them. To make every rotation a contiguous two-row update we hold the
    // factors **transposed**: `ut = U_bᵀ` (`m × m`), `vt = V_bᵀ` (`n × n`), so
    // after the sweep `ut = Uᵀ`, `vt = Vᵀ` (see `rotate_rows`). Column `j` of
    // `U`/`V` is therefore row `j` of `ut`/`vt`.
    let mut ut = transpose_square(bidiag.u().storage().as_slice(), m);
    let mut vt = transpose_square(bidiag.v().storage().as_slice(), n);
    let b = bidiag.b();
    let b_slice = b.storage().as_slice();
    let cols = b.shape()[1];
    let mut d = vec![T::ZERO; n];
    let mut e = vec![T::ZERO; n];
    for i in 0..n {
        d[i] = b_slice[i * cols + i];
        if i + 1 < n {
            e[i] = b_slice[i * cols + i + 1];
        }
    }

    qr_iterate::<T, true>(&mut d, &mut e, n, &mut ut, m, &mut vt, n)?;

    // Force σ ≥ 0: a negative pivot flips the sign of its left singular vector
    // (column `i` of `U` = row `i` of `ut`, contiguous).
    for i in 0..n {
        if d[i] < T::ZERO {
            d[i] = d[i].neg();
            for slot in &mut ut[i * m..i * m + m] {
                *slot = slot.neg();
            }
        }
    }

    // Descending sort of the singular values, carrying the U/V columns. Column
    // `old` of `U`/`V` is row `old` of `ut`/`vt` (a contiguous slice).
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).expect("singular values are finite"));

    let mut sigma = vec![T::ZERO; n];
    let mut u_thin = vec![T::ZERO; m * n]; // m × n
    let mut v_thin = vec![T::ZERO; n * n]; // n × n
    for (new_col, &old) in perm.iter().enumerate() {
        sigma[new_col] = d[old];
        for r in 0..m {
            u_thin[r * n + new_col] = ut[old * m + r];
        }
        for r in 0..n {
            v_thin[r * n + new_col] = vt[old * n + r];
        }
    }

    Ok((
        Array2::from_shape_vec([m, n], u_thin).expect("U shape matches storage"),
        sigma,
        Array2::from_shape_vec([n, n], v_thin).expect("V shape matches storage"),
    ))
}

/// Thin SVD `A = U Σ Vᵀ` via implicit-shift bidiagonal QR (Golub–Reinsch).
///
/// Wide input (`m < n`) is handled by `σ(A) = σ(Aᵀ)` with `U(A) = V(Aᵀ)`,
/// `V(A) = U(Aᵀ)`: the SVD of the tall `Aᵀ` is computed and its factors swapped.
///
/// Rank-deficient input is accepted: rank deficiency is data, reported as
/// `σᵢ = 0` in the returned `Σ`, not an error. `U` and `V` keep orthonormal
/// columns at every rank because both are accumulated products of Householder
/// reflectors and Givens rotations, whose orthogonality does not depend on the
/// singular values being nonzero. Callers that require full rank test
/// `singular_values.last()` against their own threshold — the appropriate
/// threshold is the caller's noise floor, which this function cannot know.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite input or QR
/// non-convergence.
pub fn svd_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<SvdDecomposition<T>> {
    validate_input(matrix)?;
    let [rows, cols] = matrix.shape();
    if let Some((scaled, exponent)) = scaling::balanced(matrix, 2, svd_bound(rows, cols))? {
        let mut decomposition = svd_of_balanced(&scaled.view())?;
        scaling::restore(
            &mut decomposition.singular_values,
            exponent,
            "singular value exceeds the scalar range",
        )?;
        return Ok(decomposition);
    }
    svd_of_balanced(matrix)
}

/// [`svd_decompose`] of a validated matrix whose entries are in range.
fn svd_of_balanced<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<SvdDecomposition<T>> {
    let [m, n] = matrix.shape();
    if m >= n {
        let (u, sigma, v) = svd_tall(matrix)?;
        Ok(SvdDecomposition {
            singular_values: sigma,
            left_singular_vectors: u,
            right_singular_vectors: v,
        })
    } else {
        // Compute the SVD of the tall Aᵀ and swap U ↔ V.
        let transposed = transpose_to_owned(matrix)?;
        let (u_t, sigma, v_t) = svd_tall(&transposed.view())?;
        Ok(SvdDecomposition {
            singular_values: sigma,
            left_singular_vectors: v_t,
            right_singular_vectors: u_t,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{qr_iterate, RealScalar};
    use eunomia::RealField;

    /// Backward-error bound for one bidiagonal QR reduction of order `k`:
    /// `‖E‖₂ ≤ p·ε·‖B‖₂` and `‖UᵀU − I‖₂ ≤ p·ε`, with `p` modest in the order
    /// (Golub & Van Loan, *Matrix Computations* 4th ed., §8.6.3 — the sweep is a
    /// product of plane rotations, each contributing `O(ε)`, and `O(k)` of them
    /// touch any one entry). `p = 8k` is used throughout; Weyl's theorem carries
    /// the same bound to every `|σ̂ᵢ − σᵢ|`, and to a `σ` whose exact value is 0.
    ///
    /// Returns `(absolute, relative)`: the absolute bound scaled by `‖B‖₂ ≈ σ₁`
    /// for singular values and reconstruction, the bare relative one for
    /// orthonormality (`UᵀU − I` is already dimensionless).
    fn error_bounds<T: RealScalar + RealField>(k: usize, norm: f64) -> (f64, f64) {
        #[allow(clippy::cast_precision_loss)]
        let relative = 8.0 * k as f64 * <T as RealField>::EPSILON.to_f64();
        (relative * norm, relative)
    }

    /// Run the iteration on the bidiagonal `(diag, superdiag)` with `U`/`V`
    /// initialized to the identity, then assert the singular values against
    /// `expected` (analytic, descending, trailing zeros included), the
    /// reconstruction `B = U Σ Vᵀ` of the **input** bidiagonal, and orthonormality
    /// of both accumulated factors.
    ///
    /// Reconstructing the input is what makes this a test of the rotations rather
    /// than only of the deflation: a chase that zeroes the right entries but
    /// accumulates into the wrong plane, the wrong row pair, or with the wrong
    /// sign leaves the singular values correct and the reconstruction wrong.
    fn check_bidiagonal<T: RealScalar + RealField>(
        diag: &[f64],
        superdiag: &[f64],
        expected: &[f64],
    ) {
        let k = diag.len();
        let mut d: Vec<T> = diag.iter().map(|&x| T::from_f64(x)).collect();
        let mut e: Vec<T> = (0..k)
            .map(|i| T::from_f64(superdiag.get(i).copied().unwrap_or(0.0)))
            .collect();
        let mut ut = vec![T::ZERO; k * k];
        let mut vt = vec![T::ZERO; k * k];
        for i in 0..k {
            ut[i * k + i] = T::ONE;
            vt[i * k + i] = T::ONE;
        }

        qr_iterate::<T, true>(&mut d, &mut e, k, &mut ut, k, &mut vt, k)
            .expect("the bidiagonal iteration converges on a zero diagonal");

        let (absolute, relative) = error_bounds::<T>(k, expected[0]);

        let mut sigma: Vec<f64> = d.iter().map(|x| x.abs().to_f64()).collect();
        sigma.sort_by(|a, b| b.total_cmp(a));
        for (got, want) in sigma.iter().zip(expected) {
            assert!(
                (got - want).abs() <= absolute,
                "σ {got} vs {want} exceeds {absolute:e}"
            );
        }

        // Column `i` of `U`/`V` is row `i` of `ut`/`vt` (see `rotate_row_pair`).
        for row in 0..k {
            for col in 0..k {
                let value: f64 = (0..k)
                    .map(|i| ut[i * k + row].to_f64() * d[i].to_f64() * vt[i * k + col].to_f64())
                    .sum();
                let target = if col == row {
                    diag[row]
                } else if col == row + 1 {
                    superdiag[row]
                } else {
                    0.0
                };
                assert!(
                    (value - target).abs() <= absolute,
                    "B[{row}][{col}] reconstructs as {value} not {target}"
                );
            }
        }

        for (name, factor) in [("U", &ut), ("V", &vt)] {
            for a in 0..k {
                for b in 0..k {
                    let dot: f64 = (0..k)
                        .map(|r| factor[a * k + r].to_f64() * factor[b * k + r].to_f64())
                        .sum();
                    let target = f64::from(u8::from(a == b));
                    assert!(
                        (dot - target).abs() <= relative,
                        "{name}ᵀ{name}[{a}][{b}] = {dot} not {target}"
                    );
                }
            }
        }
    }

    /// A **trailing** zero diagonal: `B = [[3, 4], [0, 0]]`, exactly rank 1.
    ///
    /// This is the reduced form of the downstream reproducer — `[[1,2],[2,4],[3,6]]`
    /// bidiagonalizes at `f32` to `d = [−3.7416573, 0]`, `e = [7.4833145]`, an
    /// *exact* zero. Shifted QR alone cannot deflate it: the implicit `BᵀB` is
    /// singular, the Wilkinson shift takes its nonzero eigenvalue, and the sweep
    /// converges to the fixed point `d = (0, 0)` with `|e|` preserved, which the
    /// `scale + |e| == scale` test never accepts. Measured before the trailing
    /// column chase existed: `d[0]` decayed `2.4e-7 → 2.8e-14 → … → 6e-45` (the
    /// smallest subnormal) while `|e[0]|` stayed pinned at `8.3666`, spinning to
    /// the 4000-iteration cap and returning a non-convergence error.
    ///
    /// `σ = (‖(3, 4)‖, 0) = (5, 0)`: row 0 is the only nonzero row, so `σ₁` is its
    /// 2-norm and the second singular value is exactly zero.
    #[test]
    fn trailing_zero_diagonal_deflates_to_the_row_norm() {
        check_bidiagonal::<f32>(&[3.0, 0.0], &[4.0], &[5.0, 0.0]);
        check_bidiagonal::<f64>(&[3.0, 0.0], &[4.0], &[5.0, 0.0]);
    }

    /// An **interior** zero diagonal: `B = [[0,5,0],[0,3,6],[0,0,4]]` — `d[0] = 0`
    /// with `p = 0 < q = 2`, which takes the left-rotation row chase rather than
    /// the trailing-column one. Both branches are therefore covered.
    ///
    /// Oracle: column 0 is zero, so `σ₃ = 0` and the remaining singular values are
    /// those of `M = [[5,0],[3,6],[0,4]]`, i.e. `√λ` for the eigenvalues of the
    /// 2×2 `MᵀM = [[34, 18], [18, 52]]` — a closed form independent of this code.
    #[test]
    fn interior_zero_diagonal_chases_out_of_the_block() {
        let (trace, det) = (86.0f64, 1444.0f64);
        let discriminant = trace.mul_add(trace, -4.0 * det).sqrt();
        let expected = [
            ((trace + discriminant) / 2.0).sqrt(),
            ((trace - discriminant) / 2.0).sqrt(),
            0.0,
        ];
        check_bidiagonal::<f32>(&[0.0, 3.0, 4.0], &[5.0, 6.0], &expected);
        check_bidiagonal::<f64>(&[0.0, 3.0, 4.0], &[5.0, 6.0], &expected);
    }

    /// The chase writes `T::ZERO` into the deflated diagonal and nothing touches
    /// it afterwards, so a zero reached that way is bit-exact rather than a
    /// rounding residue. Pinning it distinguishes the structural fix from the
    /// prohibited alternative of widening `scale + |e| == scale`, which would
    /// leave the deficient direction at `O(ε‖B‖)` instead.
    #[test]
    fn chased_zero_singular_value_is_bit_exact() {
        for (diag, superdiag) in [
            (vec![3.0f64, 0.0], vec![4.0]),           // trailing-column chase
            (vec![0.0f64, 3.0, 4.0], vec![5.0, 6.0]), // interior row chase
        ] {
            let k = diag.len();
            let mut d: Vec<f32> = diag.iter().map(|&x| x as f32).collect();
            let mut e: Vec<f32> = (0..k)
                .map(|i| superdiag.get(i).copied().unwrap_or(0.0) as f32)
                .collect();
            let (mut no_u, mut no_v) = ([0.0f32; 0], [0.0f32; 0]);
            qr_iterate::<f32, false>(&mut d, &mut e, k, &mut no_u, 0, &mut no_v, 0).unwrap();
            assert!(
                d.contains(&0.0),
                "the chased direction must be exactly zero, got {d:?}"
            );
        }
    }

    /// `dbdsqr`'s forward test splits at `|eᵢ| ≤ tol·μ` with
    /// `tol = tolmul·ε`, `tolmul = min(100, ε^(−1/8)) ≈ 90.5` in `f64`: in
    /// `d = (1, 1, 1)`, `e = (50ε, ½)` the recurrence starts at
    /// `μ = |d₀| = 1`, so `e₀` splits before any rotation, and row 0 of both
    /// accumulated factors stays exactly `e₀ᵀ` (the `½` couples only the
    /// trailing pair). With `tol = ε` the block stays whole and the sweep's
    /// rotations mix row 0 by `O(e₀)`.
    #[test]
    fn forward_test_splits_at_tolmul_eps() {
        let mut d = vec![1.0f64, 1.0, 1.0];
        let mut e = vec![50.0 * f64::EPSILON, 0.5, 0.0];
        let identity = |n: usize| -> Vec<f64> {
            (0..n * n)
                .map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 })
                .collect()
        };
        let (mut u, mut v) = (identity(3), identity(3));
        qr_iterate::<f64, true>(&mut d, &mut e, 3, &mut u, 3, &mut v, 3).unwrap();
        assert_eq!(&u[..3], &[1.0, 0.0, 0.0], "U row 0: {u:?}");
        assert_eq!(&v[..3], &[1.0, 0.0, 0.0], "V row 0: {v:?}");
    }
}
