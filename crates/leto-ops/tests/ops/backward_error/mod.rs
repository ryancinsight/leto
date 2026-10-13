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
//! place each routine's absolute deflation floor at or below `ε·‖A‖_F`
//! (`linalg::thresholds::homogeneous_safe_range`, through a lower bound
//! `2^l ≤ ‖A‖_F/‖A‖_max`); a floor deflation is therefore counted as a
//! relative `ε`. The bounds are first order in `u`
//! within each operation (`γ` absorbs the products by Lemma 3.3) and exact
//! in their composition.
//!
//! These are *a-priori* worst-case bounds at the iteration caps. A relative
//! backward error of `1` admits every output of the right norm, so a bound
//! at or above `1` says nothing: callers assert one only through
//! [`informative`]. At the caps it is vacuous for the iterative routines
//! (SVD, Francis, QL) in `F16` and `Bf16` — one length-3 reflector alone
//! costs `γ₅₃ ≈ 0.027` in `F16`, `0.26` in `Bf16` — for pivoted QR in
//! `Bf16`, and in `f32` for Francis at `n = 8` (`η ≈ 1.03`) and the Schur
//! residual ([`schur_factors`]) from `n = 5`. Those cases are covered by the *a-posteriori*
//! certificates of `a_posteriori.rs`, which bound the returned values by the
//! residuals of the returned factors, measured.

mod col_piv_qr;
mod francis;
mod ql;
mod svd;
mod symmetric;

pub use col_piv_qr::{col_piv_qr, col_piv_qr_pivot_slack, col_piv_qr_tilde};
pub use francis::{francis, francis_tilde, francis_vectors, schur_factors, schur_factors_tilde};
pub use ql::{ql, ql_tilde, ql_vectors, ql_vectors_tilde};
pub use svd::{svd, svd_factors, svd_factors_tilde, svd_tilde};
pub use symmetric::symmetric_certificate;

use eunomia::FloatElement;

/// Higham's `γ_k = k·u/(1 − k·u)`, `u = ε/2`; `+∞` once `k·u ≥ 1`.
pub fn gamma(k: f64, eps: f64) -> f64 {
    let ku = k * eps / 2.0;
    if ku >= 1.0 {
        f64::INFINITY
    } else {
        ku / (1.0 - ku)
    }
}

/// `Some(η)` when the a-priori bound `η` is informative (`η < 1`), `None`
/// when it is vacuous and must not be asserted.
pub fn informative(eta: f64) -> Option<f64> {
    (eta < 1.0).then_some(eta)
}

/// The probabilistic parameter: [`gamma_tilde`]'s `λ`, one value for the
/// whole oracle. `λ = 7` holds the per-product failure probability
/// ([`theta_failure_probability`]) at `≈ 7e-11` in every format while
/// keeping the `F16` iterative bounds informative at instrumented counts
/// and the `Bf16` SVD value bound informative at order `≤ 3`; larger `λ`
/// costs the `Bf16` coverage, smaller `λ` costs failure-probability margin.
pub const PROBABILISTIC_LAMBDA: f64 = 7.0;

/// Higham–Mary's `γ̃_n(λ) = exp(λ√n·u + n·u²/(1−u)) − 1`, `u = ε/2`,
/// `λ = λ₇` ([`PROBABILISTIC_LAMBDA`]): the probabilistic counterpart of
/// [`gamma`] for a relative error accumulated as a product `Π(1+δᵢ) − 1`
/// over `n` roundings.
///
/// *Model.* Higham & Mary, "A new approach to probabilistic rounding error
/// analysis", SIAM J. Sci. Comput. 41(5), A2815–A2835 (2019), Theorem 2.4,
/// eqs. 2.1, 2.3 (verified against the MIMS EPrint 2018.33 preprint; journal
/// volume and pages from citing literature): under Model 2.1 — the `δᵢ`
/// independent mean-zero with `|δᵢ| ≤ u`, an assumption about round-to-nearest
/// stated here rather than proved — `|Π(1+δᵢ) − 1| ≤ γ̃_n(λ)` with
/// probability at least `1 − 2exp(−λ²(1−u)²/2)` per product. A sum of terms
/// is covered the same way (their eq. 2.1): each term carries its own
/// product, and `γ̃` is increasing in `n`, so the term count bounds them
/// jointly. The suite union-bounds the failure probability over the run's
/// products (see the routine bounds); `γ̃` replaces `γ` term by term wherever
/// the deterministic derivation counts roundings, while data perturbations
/// (deflation floors) stay deterministic and compose as before.
///
/// *Magnitudes.* IEEE rounding perturbs each *result* by relative `u`, and
/// every rounded result here is `O(‖A‖)` (values) or `O(1)` with
/// `O(‖A‖)`-sensitivity (rotation/reflector parameters through smooth
/// formulas), propagated through sensitivity-`1` orthogonal maps — so each
/// rounding's forward influence is `∼u·‖A‖` and no extra size factor joins
/// `γ̃_N` (the deterministic derivations' small sizes corroborate the
/// kernels' stability). The empirical leg is the margin: firing bounds sit
/// `3–100×` above the measured errors; a systematic undercount would fail
/// the suite, not hide in it.
pub fn gamma_tilde(n: f64, eps: f64) -> f64 {
    let u = eps / 2.0;
    let lambda = PROBABILISTIC_LAMBDA;
    (lambda * n.sqrt() * u + n * u * u / (1.0 - u)).exp_m1()
}

/// The per-product failure probability of [`gamma_tilde`]:
/// `2exp(−λ²(1−u)²/2)`, `u = ε/2`. The suite asserts each bound over many
/// entries and cases; the union over a run's products is that many times
/// this, stated at the call sites, and a flake-free suite run is the
/// empirical leg (Model 2.1 is conservative for these kernels).
pub fn theta_failure_probability(eps: f64) -> f64 {
    let u = eps / 2.0;
    let lambda = PROBABILISTIC_LAMBDA;
    2.0 * (-lambda * lambda * (1.0 - u) * (1.0 - u) / 2.0).exp()
}

/// Inexact roundings on one entry's path through a one-sided Householder
/// application of length `m`, construction included
/// (`householder::reflect_in_place` + `apply_left`/`apply_right`): the
/// construction's two norm reductions (`4m`), `sqrt`, `v₀` and `β` (`3`),
/// and the application's reduction (`2m`), `β` scale and update (`3`).
/// Exact ops (`scale_binary`, negations, `×±1`) contribute no `δ` and are
/// not counted. SIMD reordering changes summation order, not the count.
fn reflector_roundings(m: usize) -> f64 {
    6.0 * f64::from_count(m) + 6.0
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
    gamma(8.0 * f64::from_count(m) + 29.0, eps)
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
    gamma(17.0 * f64::from_count(m) + 42.0, eps)
}

/// One-sided application of a plane rotation whose computed `ĉ, ŝ` carry
/// `θ_k` against the exact rotation of their inputs: each of the two updated
/// components `ĉx₁ + ŝx₂` errs by `γ_{k+2}·(|c||x₁| + |s||x₂|) ≤
/// γ_{k+2}‖x‖₂` (Higham Lemmas 19.7–19.8 with `k = 4`), so `√2·γ_{k+2}`.
pub fn rotation(k: usize, eps: f64) -> f64 {
    std::f64::consts::SQRT_2 * gamma(f64::from_count(k) + 2.0, eps)
}

/// The entry rule for `count` written entries, each with at most `r`
/// roundings over terms summing to at most `size·‖A‖`: Frobenius
/// `√count·γ_r·size`.
pub fn entries(count: usize, r: usize, size: f64, eps: f64) -> f64 {
    (f64::from_count(count)).sqrt() * gamma(f64::from_count(r), eps) * size
}

/// `francis`'s per-deflation iteration cap at order `n` (`iteration_cap`):
/// `dlahqr`'s `ITMAX = 30·max(10, n)`.
pub fn francis_iteration_cap(n: usize) -> usize {
    30 * n.max(10)
}

/// Inexact roundings on one entry's path through one 2×2 direct
/// diagonalization (`triangular_svd`): the value chains run `≤ 35`
/// roundings; `40` covers values and factor rotations alike.
const DLASV2_ROUNDINGS: f64 = 40.0;

/// The probabilistic oracle's failure budget: with `λ = λ₇` the per-product
/// failure probability is `≈ 7e-11` in every format, so the runs' union
/// (thousands of products each over hundreds of firing cases) stays
/// negligible. Pins the knob: lowering `λ` to chase vacuous bounds would
/// explode this.
#[test]
fn probabilistic_oracle_failure_budget_stays_negligible() {
    for eps in [
        f64::EPSILON,
        f64::from(f32::EPSILON),
        2.0f64.powi(-10),
        2.0f64.powi(-7),
    ] {
        let p = theta_failure_probability(eps);
        assert!(
            p < 1e-9,
            "λ₇ must hold the per-product failure probability below 1e-9 at ε = {eps:e}, got {p:e}"
        );
    }
}
