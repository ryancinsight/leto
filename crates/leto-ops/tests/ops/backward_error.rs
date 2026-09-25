//! Derived a-priori backward-error bounds for the dense factorizations,
//! composed from per-operation rounding bounds over the transformations each
//! routine applies, with each iteration count taken at the routine's own cap.
//!
//! # Model
//!
//! Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (2002):
//! every basic operation satisfies `fl(x op y) = (x op y)(1 + δ)`, `|δ| ≤ u`,
//! `u = ε/2` (§2.2, the standard model), and a quantity passing through `k`
//! such roundings carries `(1 + θ_k)`, `|θ_k| ≤ γ_k = k·u/(1 − k·u)`
//! (Lemma 3.1), with `γ_j + γ_k + γ_j·γ_k ≤ γ_{j+k}` (Lemma 3.3). An inner
//! product of length `m` in any order has `|fl(xᵀy) − xᵀy| ≤ γ_m·|x|ᵀ|y|`
//! (§3.1). Two consequences are used throughout:
//!
//! - **Entry rule.** A computed quantity that is a sum of terms, each passing
//!   through at most `r` roundings (its operands' own `θ` included), differs
//!   from the exact sum by at most `γ_r·Σ|terms|`.
//! - **Similarity rule.** A step that forms a transformation `G` from computed
//!   data and writes new entries has, for the exactly orthogonal `G̃` nearest
//!   the computed one, `T_{new} = G̃ᵀ(T + F)G̃` with `‖F‖_F` the size of the
//!   entry errors (plus `‖G − G̃‖` times the entries); over the whole run,
//!   `A + E = Q̃·T_final·Q̃ᵀ` with `‖E‖_F ≤ Σ‖F_j‖_F`, composed exactly as
//!   `Π(1 + ηⱼ) − 1` relative to `‖A‖_F` (orthogonal steps preserve it).
//!
//! The per-operation bounds below are derived in their own documentation;
//! the per-routine functions count operations from the code, at its caps.
//! The gates keep every relied-upon intermediate normal and finite, and
//! place each routine's absolute deflation floor at or below `ε·‖A‖_max`
//! (`linalg::thresholds::homogeneous_safe_range`); a floor deflation is
//! therefore counted as a relative `ε`. The bounds are first order in `u`
//! within each operation (`γ` absorbs the products by Lemma 3.3) and exact
//! in their composition.
//!
//! These are worst-case bounds at the iteration caps, so they are loose
//! (and, for 8- and 11-bit formats at the caps, can exceed `1`); the
//! measured errors are reported separately and never asserted.

/// Higham's `γ_k = k·u/(1 − k·u)`, `u = ε/2`; `+∞` once `k·u ≥ 1`.
pub fn gamma(k: f64, eps: f64) -> f64 {
    let ku = k * eps / 2.0;
    if ku >= 1.0 {
        f64::INFINITY
    } else {
        ku / (1.0 - ku)
    }
}

/// `(1 + η)^count − 1`: `count` applications of a relative bound `η`.
fn repeat(eta: f64, count: f64) -> f64 {
    (count * eta.ln_1p()).exp_m1()
}

/// `Π(1 + ηᵢ) − 1`.
fn compose(parts: &[f64]) -> f64 {
    parts.iter().map(|eta| eta.ln_1p()).sum::<f64>().exp_m1()
}

/// One-sided application of a computed Householder reflector of length `m`
/// to a vector `x`, against the exactly orthogonal reflector `P̃` of the
/// computed vector `v̂`: `‖ŷ − P̃x‖₂ ≤ γ_{8m+29}·‖x‖₂`.
///
/// *Derivation.* Every reflector in the crate forms `v̂` and a scalar `ĉ`
/// meant to be `2/(v̂ᵀv̂)`, with `ĉ·v̂ᵀv̂ = 2(1 + θ_q)`: `q = m + 1` where
/// `ĉ = 2/fl(v̂ᵀv̂)` (`householder::reflect_in_place`, `francis`'s stack
/// reflector: `m` nonnegative squares, one division); `q = 3m + 12` for
/// `dlarfg`'s `τ̂` (`bidiagonal/colmajor.rs`: `β̂` carries `θ_{m+1}` from the
/// norm, `α − β̂` adds magnitudes, `θ_{m+2}`; `v̂ᵢ = xᵢ/(α − β̂)`, `θ_{m+4}`;
/// `τ̂ = (β̂ − α)/β̂`, `θ_{2m+4}`; `v̂ᵀv̂` of the `θ_{m+4}` entries,
/// `θ_{2m+8}`). So `‖P̂ − P̃‖₂ = |ĉ − 2/v̂ᵀv̂|·‖v̂‖² ≤ 2γ_q`. The application
/// `ŷᵢ = fl(xᵢ − v̂ᵢ·fl(ĉ·fl(v̂ᵀx)))` has
/// `|ŷᵢ − (P̂x)ᵢ| ≤ γ_{m+2}·ĉ|v̂ᵢ||v̂|ᵀ|x| + u|ŷᵢ|`, so by Cauchy–Schwarz
/// `‖ŷ − P̂x‖₂ ≤ 2γ_{m+2}‖x‖₂ + u‖x‖₂` to first order. Adding `2γ_q` and
/// taking `q ≤ 3m + 12`: `2γ_{3m+12} + 2γ_{m+2} + γ_1 ≤ γ_{8m+29}`.
pub fn householder(m: usize, eps: f64) -> f64 {
    gamma(8.0 * m as f64 + 29.0, eps)
}

/// The symmetric rank-2 update `B − v̂ŵᵀ − ŵv̂ᵀ` of `symmetric_qr/reduce.rs`
/// (a two-sided reflector of length `m`), against `P̃BP̃`: `‖F‖_F ≤
/// γ_{17m+42}·‖B‖_F`.
///
/// *Derivation.* With `ĉ = β̂` from `reflect_in_place` (`q = m + 1`, `P̂` vs
/// `P̃` costing `4γ_{m+1}` two-sided), `p̂ = β̂·fl(Bv̂)` errs by
/// `γ_{m+1}β̂‖B‖_F‖v̂‖`; the correction `½β̂·fl(p̂ᵀv̂)` by
/// `4γ_{m+1}‖B‖_F/‖v̂‖` after multiplying by `‖v̂‖` (using `β̂‖v̂‖² ≈ 2`), so
/// `‖ŵ − w‖ ≤ (6γ_{m+1} + 4u)‖B‖_F/‖v̂‖` and `‖w‖ ≤ 4‖B‖₂/‖v̂‖`. The update
/// `2‖v̂‖‖ŵ − w‖ + γ₂·2‖v̂‖‖w‖ + u‖B‖` gives `12γ_{m+1} + 8u + 16u + u`;
/// with the reflector's `4γ_{m+1}` and the off-diagonal `α̂` written from the
/// norm (`γ_{m+1}`): `17γ_{m+1} + 25u ≤ γ_{17m+42}`.
pub fn householder_rank2(m: usize, eps: f64) -> f64 {
    gamma(17.0 * m as f64 + 42.0, eps)
}

/// One-sided application of a plane rotation whose computed `ĉ, ŝ` carry
/// `θ_k` against the exact rotation of their inputs: each of the two updated
/// components `ĉx₁ + ŝx₂` errs by `γ_{k+2}·(|c||x₁| + |s||x₂|) ≤
/// γ_{k+2}‖x‖₂` (Higham Lemmas 19.7–19.8 with `k = 4`), so `√2·γ_{k+2}`.
pub fn rotation(k: usize, eps: f64) -> f64 {
    std::f64::consts::SQRT_2 * gamma(k as f64 + 2.0, eps)
}

/// The entry rule for `count` written entries, each with at most `r`
/// roundings over terms summing to at most `size·‖A‖`: Frobenius
/// `√count·γ_r·size`.
pub fn entries(count: usize, r: usize, size: f64, eps: f64) -> f64 {
    (count as f64).sqrt() * gamma(r as f64, eps) * size
}

/// `bidiagonal_qr`'s iteration cap (`MAX_ITER`).
pub const SVD_ITERATION_CAP: usize = 4000;
/// `francis`'s per-deflation iteration cap (`MAX_ITER`).
pub const FRANCIS_ITERATION_CAP: usize = 2000;
/// `symmetric_qr/ql.rs`'s sweeps per eigenvalue (`dsteqr`'s `30`).
pub const QL_SWEEPS_PER_EIGENVALUE: usize = 30;

/// `dbdsqr`'s `tol = tolmul·ε`, `tolmul = max(10, min(100, ε^(−1/8)))`.
fn dbdsqr_tol(eps: f64) -> f64 {
    let tolmul = eps.powf(-0.125).clamp(10.0, 100.0);
    tolmul * eps
}

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
///   `dbdsqr`'s `tol` times a diagonal (or the floor, `≤ ε‖A‖_max`), and at
///   most `k` diagonals zeroed below `u` times their neighbours' sum.
pub fn svd(rows: usize, cols: usize, eps: f64) -> f64 {
    let (big, small) = (rows.max(cols), rows.min(cols));
    let k = small as f64;
    let rotations = (SVD_ITERATION_CAP as f64) * 2.0 * (k - 1.0) + (k - 1.0) * (k - 1.0);
    compose(&[
        repeat(householder(big, eps), 2.0 * k - 1.0),
        repeat(rotation(5, eps), rotations),
        repeat(dbdsqr_tol(eps).max(eps), k - 1.0),
        repeat(eps, k),
    ])
}

/// Eigenvalues of an `n × n` matrix through Francis (`schur`,
/// `eigenvalues`): `A + E` has exactly the computed spectrum, `‖E‖_F ≤
/// η·‖A‖_F`.
///
/// - Hessenberg: `n − 2` two-sided reflectors of length `≤ n − 1`;
/// - Francis steps: at most `FRANCIS_ITERATION_CAP` per deflation, at most
///   `n` deflations, each step at most `n − 1` two-sided reflectors of
///   length `≤ 3` (the eigenvalues-only window applies the same reflectors to
///   a subset of columns; the skipped entries never reach a diagonal block);
/// - deflations: at most `n − 1` subdiagonals zeroed, each `≤ ulp·tst ≤
///   2ε‖H‖₂` by `dlahqr`'s pre-check (or the floor);
/// - `dlanv2` per 2×2 block (at most `n/2`): its rotation (`ĉs, ŝn` carry at
///   most `θ₃₀` through the equal-diagonal and triangularizing rotations)
///   applied two-sided outside the block, and the four block entries written
///   from its formulas — the exact rotated block up to `≤ 40` roundings over
///   terms summing to `≤ 5‖block‖` (the zeroed entry is the residual of the
///   computed shift in its quadratic, bounded the same way) — and the
///   eigenvalue `a′ ± i√|b′|√|c′|` read with `θ₃` on `|b′c′|`.
pub fn francis(n: usize, eps: f64) -> f64 {
    let nf = n as f64;
    let steps = (FRANCIS_ITERATION_CAP as f64) * nf;
    let blocks = (nf / 2.0).floor();
    compose(&[
        repeat(
            householder(n.saturating_sub(1), eps),
            2.0 * (nf - 2.0).max(0.0),
        ),
        repeat(householder(3, eps), steps * 2.0 * (nf - 1.0)),
        repeat(2.0 * eps, nf - 1.0),
        repeat(
            2.0 * rotation(30, eps) + entries(4, 40, 5.0, eps) + gamma(3.0, eps),
            blocks,
        ),
    ])
}

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
    entries(n, 1, 4.0, eps)
        + entries(2, 7, 5.0, eps)
        + (n as f64 - 1.0) * entries(3, 24, 12.0, eps)
        + entries(2, 32, 4.0, eps)
        + 3.0 * u * (n as f64).sqrt()
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
    let nf = n as f64;
    compose(&[
        repeat(householder_rank2(n, eps), (nf - 2.0).max(0.0)),
        repeat(ql_sweep(n, eps), (QL_SWEEPS_PER_EIGENVALUE as f64) * nf),
        repeat(eps, nf - 1.0),
        entries(n, 1, 3.0, eps),
    ])
}

/// Per-column error of the QL eigenvectors against the exactly orthogonal
/// `Q̃` of [`ql`]: `‖q̂ⱼ − q̃ⱼ‖₂ ≤ η_Q`. The reflectors are accumulated onto
/// rows of `I` (`n − 2` one-sided applications of length `≤ n`), then every
/// QL rotation is applied to a row pair (`ĉ, ŝ` with `θ₆`), at most
/// `(n − 1)` per sweep.
pub fn ql_vectors(n: usize, eps: f64) -> f64 {
    let nf = n as f64;
    compose(&[
        repeat(householder(n, eps), (nf - 2.0).max(0.0)),
        repeat(
            rotation(6, eps),
            (QL_SWEEPS_PER_EIGENVALUE as f64) * nf * (nf - 1.0),
        ),
    ])
}

/// Column-pivoted QR of an `rows × cols` matrix: `A·P + E = Q̃·R̂` with
/// `‖E‖_F ≤ η·‖A‖_F` (`min(rows, cols)` one-sided reflectors of length
/// `≤ rows` on the columns of `R`), and each row of the accumulated `Q̂`
/// within the same `η` of `Q̃` (the same reflectors applied to the rows of
/// `I`), so `‖A·P − Q̂·R̂‖_F ≤ (η + √rows·η·(1 + η))·‖A‖_F`.
pub fn col_piv_qr(rows: usize, cols: usize, eps: f64) -> f64 {
    let eta = repeat(householder(rows, eps), rows.min(cols) as f64);
    eta + (rows as f64).sqrt() * eta * (1.0 + eta)
}

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
            residual_rounding += (gamma(n as f64 + 2.0, eps) * magnitude).powi(2);
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
            gram_rounding += (gamma(n as f64, eps) * magnitude).powi(2);
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

/// Per-row error of the accumulated Schur vectors against the exactly
/// orthogonal `Q̃` of [`francis`]: the Hessenberg reflectors applied to the
/// rows of `I`, every Francis reflector applied from the right, and every
/// `dlanv2` rotation, at the same counts.
pub fn francis_vectors(n: usize, eps: f64) -> f64 {
    let nf = n as f64;
    compose(&[
        repeat(householder(n, eps), (nf - 2.0).max(0.0)),
        repeat(
            householder(3, eps),
            (FRANCIS_ITERATION_CAP as f64) * nf * (nf - 1.0),
        ),
        repeat(rotation(30, eps), (nf / 2.0).floor()),
    ])
}
