//! A-priori SVD value, vector, and factor bounds.

use eunomia::FloatElement;

/// `dbdsqr`'s `tol = tolmul·ε`, `tolmul = max(10, min(100, ε^(−1/8)))`.
fn dbdsqr_tol(eps: f64) -> f64 {
    let tolmul = eps.powf(-0.125).clamp(10.0, 100.0);
    tolmul * eps
}

/// `bidiagonal_qr`'s iteration cap (`MAX_ITER`).
const SVD_ITERATION_CAP: usize = 4000;

/// Inexact roundings on one entry's path through one Givens rotation
/// (`givens` + application): construction (`r`, `1/r`, `c`, `s`: `7`) and
/// the `c·a + s·b` update (`3`). Entries written directly from the returned
/// `r` see only its `4`.
const ROTATION_ROUNDINGS: f64 = 10.0;

/// Inexact roundings on one bidiagonal entry's path through one shifted
/// `dbdsqr` sweep: the Wilkinson shift (`≤ 20`) plus the two overlapping
/// `k`-iterations touching the entry (`2 × 26`: two Givens constructions
/// and the bulge combinations each). Zero-shift sweeps skip the shift;
/// `75` covers both with margin.
const DBDSQR_SWEEP_ROUNDINGS: f64 = 75.0;

/// Inexact roundings on one bidiagonal entry's path through one
/// negligible-diagonal chase: one Givens construction and two multiplies
/// per touched entry.
const DBDSQR_CHASE_ROUNDINGS: f64 = 12.0;

/// Each bidiagonal entry joins at most two 2×2 blocks over a run: firing
/// zeroes the shared superdiagonal permanently, so the pair can never
/// reform — an entry is the top of one pair and the bottom of one pair at
/// most.
const DLASV2_OCCURRENCES: f64 = 2.0;

/// Singular values of an `rows × cols` matrix (`singular_values`,
/// `svd_decompose`): each is within `η·‖A‖_F` of the exact one (Weyl).
///
/// With `M = max(rows, cols)`, `k = min(rows, cols)`:
/// - bidiagonalization: `2k − 1` one-sided reflectors of length `≤ M`;
/// - QR sweeps: at most `SVD_ITERATION_CAP` iterations, each at most
///   `2(k − 1)` rotations (`givens`: `r = √(a² + b²)`, `θ₃`; `1/r`, `θ₄`;
///   `ĉ = a·(1/r)`, `θ₅`), plus at most `k − 1` zero-diagonal chases of
///   `k − 1` rotations each;
/// - deflations: at most `k − 1` superdiagonals zeroed, each below
///   `dbdsqr`'s `tol` times a diagonal (or the floor, `≤ ε‖A‖_F`), and at
///   most `k` diagonals zeroed below `u` times their neighbours' sum;
/// - 2×2 direct diagonalizations (`triangular_svd`): at most `k − 1` firings
///   (each zeroes a superdiagonal permanently, `continue`ing past the
///   iteration counter), two values each with `≤ 40` roundings over terms
///   `≤ 2‖block‖₂.
pub fn svd(rows: usize, cols: usize, eps: f64) -> f64 {
    let (big, small) = (rows.max(cols), rows.min(cols));
    let k = f64::from_count(small);
    let rotations = f64::from_count(SVD_ITERATION_CAP) * 2.0 * (k - 1.0) + (k - 1.0) * (k - 1.0);
    super::compose(&[
        super::repeat(super::householder(big, eps), 2.0 * k - 1.0),
        super::repeat(super::rotation(5, eps), rotations),
        super::repeat(dbdsqr_tol(eps).max(eps), k - 1.0),
        super::repeat(eps, k),
        super::repeat(super::entries(2, 40, 2.0, eps), k - 1.0),
    ])
}

/// Per-row error of the accumulated singular vectors `Û` (or `V̂`) against the
/// exactly orthogonal factors of [`svd`]: the bidiagonalization's reflectors
/// (at most `k` per side, length `≤ M`), every QR rotation (`givens`,
/// `θ₅`), counted for both sides at [`svd`]'s totals, and the 2×2 pairs'
/// factor rotations (each row in at most two pairs — a fired pair's zeroed
/// superdiagonal never reforms — two one-sided rotations each with
/// `triangular_svd`'s `(ĉ, ŝ)` carrying `θ₃₅`).
fn svd_vectors(rows: usize, cols: usize, eps: f64) -> f64 {
    let (big, small) = (rows.max(cols), rows.min(cols));
    let k = f64::from_count(small);
    let rotations = f64::from_count(SVD_ITERATION_CAP) * 2.0 * (k - 1.0) + (k - 1.0) * (k - 1.0);
    super::compose(&[
        super::repeat(super::householder(big, eps), k),
        super::repeat(super::rotation(5, eps), rotations),
        super::repeat(2.0 * super::rotation(35, eps), 2.0),
    ])
}

/// The residual of the returned SVD factors, `‖Â − Û·Σ̂·V̂ᵀ‖_F ≤ η_R·‖Â‖_F`,
/// and the orthogonality of each, `‖Ûᵀ Û − I‖_F ≤ η_O` (likewise `V̂`), a
/// priori. As [`schur_factors`], with `‖D_U‖_F, ‖D_V‖_F ≤ d = √M·η_U`
/// ([`svd_vectors`]): `ÛΣ̂V̂ᵀ − ŨΣ̂Ṽᵀ = D_UΣ̂Ṽᵀ + ŨΣ̂D_Vᵀ + D_UΣ̂D_Vᵀ`, so
/// `η_R = η + (2d + d²)(1 + η)` and `η_O = 2d + d²`.
pub fn svd_factors(rows: usize, cols: usize, eps: f64) -> (f64, f64) {
    let eta = svd(rows, cols, eps);
    let d = (f64::from_count(rows.max(cols))).sqrt() * svd_vectors(rows, cols, eps);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}

// ---- Probabilistic bounds (Higham–Mary, at instrumented counts). ----
//
// The deterministic bounds above compose per-step `γ` at iteration caps,
// which is vacuous for the iterative routines in `F16`/`Bf16`. These bounds
// count the same roundings per entry over the run's *actual*
// transformations (recorded by `leto_ops::transformation_counts`) and apply
// [`gamma_tilde`] once per entry path: every `θ`-product in the deterministic
// derivation becomes one `γ̃_N(λ)` with `N` the entry's total rounding
// count, `λ = λ₇`. Deterministic data perturbations (deflation floors) are
// not roundings and compose as before, on the good event.
// Failure probability: union over the run's products, crudely capped by
// (entries × roundings) × [`theta_failure_probability`]; the call sites
// state it, and a flake-free suite run is the empirical leg.

/// Per-entry rounding count of the Golub–Kahan value path: the
/// bidiagonalization (`k` left reflectors of length `≤ M`, `k−1` right of
/// length `≤ k−1`; [`reflector_roundings`] covers both the `U/V` and the
/// values-only `larfg` (`3m+4`) constructions), the instrumented sweeps
/// and chases, and the capped 2×2 direct diagonalizations.
fn svd_roundings(rows: usize, cols: usize, sweeps: usize, chases: usize) -> f64 {
    let (big, k) = (rows.max(cols), rows.min(cols));
    let bidiagonal = f64::from_count(k) * super::reflector_roundings(big)
        + f64::from_count(k.saturating_sub(1)) * super::reflector_roundings(k.saturating_sub(1));
    let iteration = f64::from_count(sweeps) * DBDSQR_SWEEP_ROUNDINGS
        + f64::from_count(chases) * DBDSQR_CHASE_ROUNDINGS;
    let direct = DLASV2_OCCURRENCES * super::DLASV2_ROUNDINGS;
    bidiagonal + iteration + direct
}

/// Per-entry rounding count of the accumulated `U`/`V`: `U` meets the `k`
/// left reflectors and `V` the `k−1` right ones (the max covers both); per
/// sweep an entry meets two rotations plus the shift chain (`40`); per
/// chase the pivot row meets `≤ k` rotations, others one.
fn svd_vectors_roundings(rows: usize, cols: usize, sweeps: usize, chases: usize) -> f64 {
    let (big, k) = (rows.max(cols), rows.min(cols));
    let bidiagonal = f64::from_count(k) * super::reflector_roundings(big);
    let per_sweep = 2.0 * ROTATION_ROUNDINGS + 20.0;
    let per_chase = ROTATION_ROUNDINGS * f64::from_count(k);
    let iteration = f64::from_count(sweeps) * per_sweep + f64::from_count(chases) * per_chase;
    let direct = DLASV2_OCCURRENCES * super::DLASV2_ROUNDINGS;
    bidiagonal + iteration + direct
}

/// Probabilistic SVD value bound at `sweeps`/`chases` instrumented
/// iterations: one `γ̃` over the entry path, composed with the
/// deterministic deflation floors exactly as in [`svd`].
pub fn svd_tilde(rows: usize, cols: usize, eps: f64, sweeps: usize, chases: usize) -> f64 {
    let roundings = super::gamma_tilde(svd_roundings(rows, cols, sweeps, chases), eps);
    let k = rows.min(cols);
    super::compose(&[
        roundings,
        super::repeat(
            dbdsqr_tol(eps).max(eps),
            f64::from_count(k.saturating_sub(1)),
        ),
        super::repeat(eps, f64::from_count(k)),
    ])
}

/// Probabilistic `U`/`V` per-row bound at instrumented iterations.
fn svd_vectors_tilde(rows: usize, cols: usize, eps: f64, sweeps: usize, chases: usize) -> f64 {
    super::gamma_tilde(svd_vectors_roundings(rows, cols, sweeps, chases), eps)
}

/// Probabilistic `(residual, orthogonality)` pair, same algebra as
/// [`svd_factors`].
pub fn svd_factors_tilde(
    rows: usize,
    cols: usize,
    eps: f64,
    sweeps: usize,
    chases: usize,
) -> (f64, f64) {
    let eta = svd_tilde(rows, cols, eps, sweeps, chases);
    let d = (f64::from_count(rows.max(cols))).sqrt()
        * svd_vectors_tilde(rows, cols, eps, sweeps, chases);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}
