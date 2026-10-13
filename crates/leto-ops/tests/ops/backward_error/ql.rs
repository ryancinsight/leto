//! A-priori QL value and vector bounds.

use eunomia::FloatElement;

/// `symmetric_qr/ql.rs`'s sweeps per eigenvalue (`dsteqr`'s `30`).
const QL_SWEEPS_PER_EIGENVALUE: usize = 30;

/// Inexact roundings on one tridiagonal entry's path through one implicit-QL
/// sweep (`symmetric_qr/ql.rs`): the Wilkinson-shift formation (`≤ 15`),
/// the two overlapping rotations touching the entry (`2 × 24`, the
/// `θ₆` Givens plus bulge combinations each), the tail chain (`7`), and
/// the shift accumulation (`1`). `75` covers with margin.
const QL_SWEEP_ROUNDINGS: f64 = 75.0;

/// The per-sweep backward error of `symmetric_qr/ql.rs`'s implicit QL on an
/// active block of order `≤ n`, relative to `‖T‖₂`. Every value the sweep
/// holds is bounded by `‖T − σI‖₂ ≤ 2‖T‖₂`: the accumulated shift `σ` is an
/// eigenvalue of a 2×2 principal submatrix of `T`, so by interlacing
/// `|σ| ≤ ‖T‖₂`; and `|p|, |g|, |h| ≤ |d| + |e| ≤ 4‖T‖₂`.
/// - shifts `dⱼ ← dⱼ − h`: `n` entries, one rounding, terms `≤ 4‖T‖₂`;
/// - the head `dₗ = eₗ/(p + r)`, `d_{l+1} = eₗ(p + r)` (`r = hypot(p, 1)`,
///   `θ₅`): two entries, `≤ 7` roundings, terms `≤ 5‖T‖₂`;
/// - each inner rotation (`ĉ = p/r`, `ŝ = eᵢ/r`: `θ₆`): `d_{i+1} = h +
///   s(cg + s dᵢ)`, `p = c dᵢ − s g`, `e_{i+1} = s·r` — three values, `≤ 24`
///   roundings (`g = c eᵢ` and `h = c p` carry `θ₇`), terms `≤ 12‖T‖₂`;
/// - the tail `p = −s s₂ c₃ e_{l+1} eₗ / d_{l+1}`, `eₗ = s p`, `dₗ = c p`:
///   two entries, `≤ 32` roundings, terms `≤ 4‖T‖₂`;
/// - the shift accumulation `σ ← σ + h`: `≤ 3‖T‖₂·u`, carried into every
///   later eigenvalue.
fn ql_sweep(n: usize, eps: f64) -> f64 {
    let u = eps / 2.0;
    super::entries(n, 1, 4.0, eps)
        + super::entries(2, 7, 5.0, eps)
        + (f64::from_count(n) - 1.0) * super::entries(3, 24, 12.0, eps)
        + super::entries(2, 32, 4.0, eps)
        + 3.0 * u * (f64::from_count(n)).sqrt()
}

/// Eigenvalues of a symmetric `n × n` matrix through the tridiagonal QL
/// (`symmetric_eigen_qr`): `A + E` has exactly the computed eigenvalues,
/// `‖E‖_F ≤ η·‖A‖_F`, with `‖T‖₂ ≤ ‖A‖_F`.
///
/// - reduction: `n − 2` rank-2 updates of length `≤ n`;
/// - QL: at most `QL_SWEEPS_PER_EIGENVALUE·n` sweeps ([`ql_sweep`]);
/// - deflations: at most `n − 1` off-diagonals zeroed below `u·t`,
///   `t ≤ 2‖T‖₂`;
/// - the final `dₗ + σ`: one rounding over `|dₗ| + |σ| ≤ 3‖T‖₂`, `n` entries.
pub fn ql(n: usize, eps: f64) -> f64 {
    let nf = f64::from_count(n);
    super::compose(&[
        super::repeat(super::householder_rank2(n, eps), (nf - 2.0).max(0.0)),
        super::repeat(
            ql_sweep(n, eps),
            f64::from_count(QL_SWEEPS_PER_EIGENVALUE) * nf,
        ),
        super::repeat(eps, nf - 1.0),
        super::entries(n, 1, 3.0, eps),
    ])
}

/// Per-column error of the QL eigenvectors against the exactly orthogonal
/// `Q̃` of [`ql`]: `‖q̂ⱼ − q̃ⱼ‖₂ ≤ η_Q`. The reflectors are accumulated onto
/// rows of `I` (`n − 2` one-sided applications of length `≤ n`), then every
/// QL rotation is applied to a row pair (`ĉ, ŝ` with `θ₆`), at most
/// `(n − 1)` per sweep.
pub fn ql_vectors(n: usize, eps: f64) -> f64 {
    let nf = f64::from_count(n);
    super::compose(&[
        super::repeat(super::householder(n, eps), (nf - 2.0).max(0.0)),
        super::repeat(
            super::rotation(6, eps),
            f64::from_count(QL_SWEEPS_PER_EIGENVALUE) * nf * (nf - 1.0),
        ),
    ])
}

/// Per-entry rounding count of the tridiagonal-QL value path: the
/// reduction (`n − 2` rank-2 updates, each a sequential `θ` of length
/// `17n + 42` per [`householder_rank2`]'s derivation), `sweeps`
/// instrumented sweeps, and the final `dₗ + σ` rounding. The eigenvalues-only
/// and vector paths share the iteration; deflations stay deterministic.
fn ql_roundings(n: usize, sweeps: usize) -> f64 {
    let nf = f64::from_count(n);
    let reduction = (nf - 2.0).max(0.0) * (17.0 * nf + 42.0);
    let iteration = f64::from_count(sweeps) * QL_SWEEP_ROUNDINGS;
    reduction + iteration + 1.0
}

/// Per-row rounding count of the accumulated QL eigenvectors: the
/// reduction's reflectors one-sided plus two rotations (`θ₆`, `8`
/// roundings each with the application) per sweep.
fn ql_vectors_roundings(n: usize, sweeps: usize) -> f64 {
    let nf = f64::from_count(n);
    let reduction = (nf - 2.0).max(0.0) * super::reflector_roundings(n);
    let iteration = f64::from_count(sweeps) * 16.0;
    reduction + iteration
}

/// Probabilistic QL value bound at `sweeps` instrumented sweeps: one `γ̃`
/// over the entry path, composed with the deterministic deflation floors
/// exactly as in [`ql`].
pub fn ql_tilde(n: usize, eps: f64, sweeps: usize) -> f64 {
    let roundings = super::gamma_tilde(ql_roundings(n, sweeps), eps);
    super::compose(&[roundings, super::repeat(eps, f64::from_count(n) - 1.0)])
}

/// Probabilistic QL-vectors per-row bound at instrumented sweeps.
pub fn ql_vectors_tilde(n: usize, eps: f64, sweeps: usize) -> f64 {
    super::gamma_tilde(ql_vectors_roundings(n, sweeps), eps)
}
