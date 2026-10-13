//! A-priori symmetric-eigen certificate bound.

use eunomia::FloatElement;

/// An a-posteriori bound on the eigenvalue errors of a symmetric solver that
/// returns an eigenbasis — used for a *reference* solve (Jacobi), never for
/// the routine under test: every `λ̂ᵢ` (ascending) is within the returned
/// bound of `λᵢ(A)` (ascending).
///
/// *Derivation.* With `R = A·V̂ − V̂·Λ̂` and `δ = ‖V̂ᵀV̂ − I‖₂ < 1`, write the
/// polar decomposition `V̂ = U·P` (`U` orthogonal, `P` symmetric positive
/// definite): `‖P − I‖₂ ≤ ‖P² − I‖₂ = δ` and `‖P⁻¹‖₂ ≤ 1/(1 − δ)`. Then
/// `A·U − U·Λ̂ = R·P⁻¹ + U·((P − I)Λ̂P⁻¹ + Λ̂(P⁻¹ − I))`, so
/// `‖A − U·Λ̂·Uᵀ‖₂ ≤ (‖R‖₂ + 2δ·‖Λ̂‖₂)/(1 − δ)`, and Weyl's inequality matches
/// the sorted eigenvalues of `A` and `Λ̂` within that. `R` and `V̂ᵀV̂` are
/// evaluated in `f64` on the exact images; their own rounding is bounded by
/// `γ_{n+2}(|A||V̂| + |V̂||Λ̂|)` and `γ_n|V̂|ᵀ|V̂|` (Frobenius) and added.
/// `vectors` holds the eigenvectors as columns of a row-major `n × n`.
pub fn symmetric_certificate(a: &[f64], values: &[f64], vectors: &[f64], n: usize) -> f64 {
    let eps = f64::EPSILON;
    let (mut residual, mut residual_rounding) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            let av: f64 = (0..n).map(|k| a[i * n + k] * vectors[k * n + j]).sum();
            let magnitude: f64 = (0..n)
                .map(|k| (a[i * n + k] * vectors[k * n + j]).abs())
                .sum::<f64>()
                + (vectors[i * n + j] * values[j]).abs();
            residual += (av - vectors[i * n + j] * values[j]).powi(2);
            residual_rounding += (super::gamma(f64::from_count(n) + 2.0, eps) * magnitude).powi(2);
        }
    }
    let (mut gram, mut gram_rounding) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n)
                .map(|k| vectors[k * n + i] * vectors[k * n + j])
                .sum();
            let magnitude: f64 = (0..n)
                .map(|k| (vectors[k * n + i] * vectors[k * n + j]).abs())
                .sum();
            gram += (dot - if i == j { 1.0 } else { 0.0 }).powi(2);
            gram_rounding += (super::gamma(f64::from_count(n), eps) * magnitude).powi(2);
        }
    }
    let delta = gram.sqrt() + gram_rounding.sqrt();
    assert!(
        delta < 1.0,
        "reference eigenbasis is not near-orthonormal: {delta}"
    );
    let lambda_max = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    (residual.sqrt() + residual_rounding.sqrt() + 2.0 * delta * lambda_max) / (1.0 - delta)
}
