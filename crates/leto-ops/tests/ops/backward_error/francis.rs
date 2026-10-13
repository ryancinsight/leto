//! A-priori Francis-QR value, vector, and Schur-factor bounds.

use eunomia::FloatElement;

/// Inexact roundings on one entry's path through a one-sided stack-reflector
/// application (`francis::bulge`): construction (`stack_reflector`, length
/// `≤ 3`: two reductions and `τ`, `12`) and application (`w`, `β`, update,
/// `9`). Length-2 chases cost less; `21` covers both.
const STACK_ROUNDINGS: f64 = 21.0;

/// Inexact roundings on one `H` entry's path through one Francis step: at
/// most three left and three right one-sided stack-reflector applications
/// (a length-3 reflector spans a 3-window, so entry `(i, j)` meets the
/// reflectors with `k ∈ [i−2, i]` from the left and `k ∈ [j−2, j]` from the
/// right), plus the shift/first-column chain (`≤ 50`) seeding the first
/// reflector and propagating through the chase.
const FRANCIS_STEP_ROUNDINGS: f64 = 6.0 * STACK_ROUNDINGS + 50.0;

/// Inexact roundings on one Schur-vector entry's path through one Francis
/// step: at most three right applications (the row window slides the same
/// way), plus the same shift chain through the applied reflectors.
const FRANCIS_VECTORS_STEP_ROUNDINGS: f64 = 3.0 * STACK_ROUNDINGS + 50.0;

/// Eigenvalues of an `n × n` matrix through Francis (`schur`,
/// `eigenvalues`): `A + E` has exactly the computed spectrum, `‖E‖_F ≤
/// η·‖A‖_F`.
///
/// - Hessenberg: `n − 2` two-sided reflectors of length `≤ n − 1`;
/// - Francis steps: at most `francis_iteration_cap(n)` per deflation, at most
///   `n` deflations, each step at most `n − 1` two-sided reflectors of
///   length `≤ 3` (the eigenvalues-only window applies the same reflectors to
///   a subset of columns; the skipped entries never reach a diagonal block);
/// - deflations: at most `n − 1` subdiagonals zeroed, each `≤ ulp·tst ≤
///   2ε‖H‖₂` by `dlahqr`'s pre-check, `≤ 4ε‖H‖₂` by the neighbourhood test
///   of a stalled block (four entries of `H`), or the floor;
/// - `dlanv2` per 2×2 block (at most `n/2`): its rotation (`ĉs, ŝn` carry at
///   most `θ₃₀` through the equal-diagonal and triangularizing rotations)
///   applied two-sided outside the block, and the four block entries written
///   from its formulas — the exact rotated block up to `≤ 40` roundings over
///   terms summing to `≤ 5‖block‖` (the zeroed entry is the residual of the
///   computed shift in its quadratic, bounded the same way) — and the
///   eigenvalue `a′ ± i√|b′|√|c′|` read with `θ₃` on `|b′c′|`.
pub fn francis(n: usize, eps: f64) -> f64 {
    let nf = f64::from_count(n);
    let steps = f64::from_count(super::francis_iteration_cap(n)) * nf;
    let blocks = (nf / 2.0).floor();
    super::compose(&[
        super::repeat(
            super::householder(n.saturating_sub(1), eps),
            2.0 * (nf - 2.0).max(0.0),
        ),
        super::repeat(super::householder(3, eps), steps * 2.0 * (nf - 1.0)),
        super::repeat(4.0 * eps, nf - 1.0),
        super::repeat(
            2.0 * super::rotation(30, eps)
                + super::entries(4, 40, 5.0, eps)
                + super::gamma(3.0, eps),
            blocks,
        ),
    ])
}

/// Per-row error of the accumulated Schur vectors against the exactly
/// orthogonal `Q̃` of [`francis`]: the Hessenberg reflectors applied to the
/// rows of `I`, every Francis reflector applied from the right, and every
/// `dlanv2` rotation, at the same counts.
pub fn francis_vectors(n: usize, eps: f64) -> f64 {
    let nf = f64::from_count(n);
    super::compose(&[
        super::repeat(super::householder(n, eps), (nf - 2.0).max(0.0)),
        super::repeat(
            super::householder(3, eps),
            f64::from_count(super::francis_iteration_cap(n)) * nf * (nf - 1.0),
        ),
        super::repeat(super::rotation(30, eps), (nf / 2.0).floor()),
    ])
}

/// The residual of the returned Schur factors, `‖Â − Q̂·T̂·Q̂ᵀ‖_F ≤
/// η_R·‖Â‖_F`, and their orthogonality, `‖Q̂ᵀQ̂ − I‖_F ≤ η_O`, a priori.
///
/// *Derivation.* `Â + E = Q̃·T̂·Q̃ᵀ` with `‖E‖_F ≤ η‖Â‖_F` ([`francis`]) and
/// `Q̂ = Q̃ + D`, `‖D‖_F ≤ d = √n·η_Q` ([`francis_vectors`], per row). Then
/// `Q̂ᵀQ̂ − I = Q̃ᵀD + DᵀQ̃ + DᵀD` gives `η_O = 2d + d²`, and
/// `Â − Q̂T̂Q̂ᵀ = −E − (DT̂Q̃ᵀ + Q̃T̂Dᵀ + DT̂Dᵀ)` with
/// `‖T̂‖_F ≤ (1 + η)‖Â‖_F` gives `η_R = η + (2d + d²)(1 + η)`.
pub fn schur_factors(n: usize, eps: f64) -> (f64, f64) {
    let eta = francis(n, eps);
    let d = (f64::from_count(n)).sqrt() * francis_vectors(n, eps);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}

/// Per-entry rounding count of the Francis value path: the Hessenberg
/// reduction (`≤ 2(n−2)` one-sided applications of length `≤ n−1`, both
/// paths), `steps` Francis steps, the `⌊n/2⌋` `dlanv2` blocks (rotation
/// `θ_30` applied two-sided, `36` per entry; block entries `40`), and the
/// `θ_3` read-off. The eigenvalues-only windows only exclude entries, so
/// the Schur-path count covers both.
fn francis_roundings(n: usize, steps: usize) -> f64 {
    let hessenberg = 2.0
        * f64::from_count(n.saturating_sub(2))
        * super::reflector_roundings(n.saturating_sub(1));
    let chase = f64::from_count(steps) * FRANCIS_STEP_ROUNDINGS;
    let blocks = f64::from_count(n / 2) * 36.0 + super::DLASV2_ROUNDINGS + 3.0;
    hessenberg + chase + blocks
}

/// Per-entry rounding count of the accumulated Schur vectors: the
/// Hessenberg reflectors from the right, every Francis step's right
/// applications, and the `dlanv2` rotations (`36` covers the one-sided
/// application).
fn francis_vectors_roundings(n: usize, steps: usize) -> f64 {
    let hessenberg =
        f64::from_count(n.saturating_sub(2)) * super::reflector_roundings(n.saturating_sub(1));
    let chase = f64::from_count(steps) * FRANCIS_VECTORS_STEP_ROUNDINGS;
    let blocks = f64::from_count(n / 2) * 36.0;
    hessenberg + chase + blocks
}

/// Probabilistic Francis value bound at `steps` instrumented steps: one
/// `γ̃` over the entry path, composed with the deterministic deflation
/// floors exactly as in [`francis`].
pub fn francis_tilde(n: usize, eps: f64, steps: usize) -> f64 {
    let roundings = super::gamma_tilde(francis_roundings(n, steps), eps);
    super::compose(&[
        roundings,
        super::repeat(4.0 * eps, f64::from_count(n.saturating_sub(1))),
    ])
}

/// Probabilistic Schur-vectors per-row bound at `steps` instrumented steps.
fn francis_vectors_tilde(n: usize, eps: f64, steps: usize) -> f64 {
    super::gamma_tilde(francis_vectors_roundings(n, steps), eps)
}

/// Probabilistic `(residual, orthogonality)` pair, same algebra as
/// [`schur_factors`].
pub fn schur_factors_tilde(n: usize, eps: f64, steps: usize) -> (f64, f64) {
    let eta = francis_tilde(n, eps, steps);
    let d = (f64::from_count(n)).sqrt() * francis_vectors_tilde(n, eps, steps);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}
