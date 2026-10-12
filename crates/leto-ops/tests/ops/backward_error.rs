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
    6.0 * m as f64 + 6.0
}

/// Inexact roundings on one entry's path through a one-sided stack-reflector
/// application (`francis::bulge`): construction (`stack_reflector`, length
/// `≤ 3`: two reductions and `τ`, `12`) and application (`w`, `β`, update,
/// `9`). Length-2 chases cost less; `21` covers both.
const STACK_ROUNDINGS: f64 = 21.0;

/// Inexact roundings on one entry's path through one Givens rotation
/// (`givens` + application): construction (`r`, `1/r`, `c`, `s`: `7`) and
/// the `c·a + s·b` update (`3`). Entries written directly from the returned
/// `r` see only its `4`.
const ROTATION_ROUNDINGS: f64 = 10.0;

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
/// `francis`'s per-deflation iteration cap at order `n` (`iteration_cap`):
/// `dlahqr`'s `ITMAX = 30·max(10, n)`.
pub fn francis_iteration_cap(n: usize) -> usize {
    30 * n.max(10)
}
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
///   `dbdsqr`'s `tol` times a diagonal (or the floor, `≤ ε‖A‖_F`), and at
///   most `k` diagonals zeroed below `u` times their neighbours' sum;
/// - 2×2 direct diagonalizations (`triangular_svd`): at most `k − 1` firings
///   (each zeroes a superdiagonal permanently, `continue`ing past the
///   iteration counter), two values each with `≤ 40` roundings over terms
///   `≤ 2‖block‖₂.
pub fn svd(rows: usize, cols: usize, eps: f64) -> f64 {
    let (big, small) = (rows.max(cols), rows.min(cols));
    let k = small as f64;
    let rotations = (SVD_ITERATION_CAP as f64) * 2.0 * (k - 1.0) + (k - 1.0) * (k - 1.0);
    compose(&[
        repeat(householder(big, eps), 2.0 * k - 1.0),
        repeat(rotation(5, eps), rotations),
        repeat(dbdsqr_tol(eps).max(eps), k - 1.0),
        repeat(eps, k),
        repeat(entries(2, 40, 2.0, eps), k - 1.0),
    ])
}

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
    let nf = n as f64;
    let steps = francis_iteration_cap(n) as f64 * nf;
    let blocks = (nf / 2.0).floor();
    compose(&[
        repeat(
            householder(n.saturating_sub(1), eps),
            2.0 * (nf - 2.0).max(0.0),
        ),
        repeat(householder(3, eps), steps * 2.0 * (nf - 1.0)),
        repeat(4.0 * eps, nf - 1.0),
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

/// Relative slack `τ` of the pivot order of column-pivoted QR with `DLAQP2`
/// norm downdating (ADR 0035): for every step `k < rank` and every later
/// column `j`, the exact tail `S_j = Σ_{i ≥ k} R_{ij}²` of the returned `R`
/// satisfies `S_j ≤ (1 + τ)·R_{kk}²`. `+∞` once the bound is vacuous.
///
/// *Derivation.* Let `u = ε/2`, `η` the one-reflector bound of [`householder`],
/// `χ = (1 + η)² − 1`, `m = rows`, `p = min(rows, cols)`. A reflector
/// application moves the squared norm of the rows `≥ k` of a column by at most
/// `χ` relatively, so over `≤ p` applications the exact tail sums `s` of the
/// stored entries stay within `Λ = (1 + η)^{2p}/(1 − γ_{m+1})` of the
/// reference `c_ref`, the cached key at its last exact computation, whose own
/// error is `γ_{m+1}·s`. One downdate `c' = fl(c − fl(a²))` adds at most
/// `χ·Λ + u(2 + χ)·Λ` of `c_ref` to the absolute error `E` (the drift of `s`,
/// the rounding of `a²`, and the rounding of the difference), so
/// `E_{t+1} = (E_t + (χ + u(2 + χ))Λ)/(1 − u)` from `E_0 = γ_{m+1}Λ`, in units
/// of `c_ref`, and `E_p` bounds every cached key. A key that survives has
/// `c > TOL3Z·c_ref` (the recompute test), so its relative error is at most
/// `ρ = E_p/TOL3Z` and `s/(1 + ρ) ≤ c ≤ s/(1 − ρ)`. The pivot has the largest
/// key, so every competitor satisfies `s_j ≤ s_pivot(1 + ρ)/(1 − ρ)`. Final
/// `R` differs from the state at step `k` by at most `p` further applications
/// (`(1 + η)^{2p}` on a tail sum), and `R_{kk}² ≥ s_pivot(1 − η)²`, giving
/// `1 + τ = ((1 + ρ)/(1 − ρ))·(1 + η)^{2p}/(1 − η)²`. `TOL3Z = √u` is the
/// library's threshold, taken one rounding low.
pub fn col_piv_qr_pivot_slack(rows: usize, cols: usize, eps: f64) -> f64 {
    let (m, steps) = (rows as f64, rows.min(cols) as f64);
    let u = eps / 2.0;
    let eta = householder(rows, eps);
    let key = gamma(m + 1.0, eps);
    if eta >= 1.0 || key >= 1.0 {
        return f64::INFINITY;
    }
    let drift = (1.0 + eta).powi(2) - 1.0;
    let growth = (2.0 * steps * eta.ln_1p()).exp();
    let lambda = growth / (1.0 - key);
    let mut error = key * lambda;
    for _ in 0..rows.min(cols) {
        error = (error + (drift + u * (2.0 + drift)) * lambda) / (1.0 - u);
    }
    let rho = error / (u.sqrt() * (1.0 - u));
    if rho >= 1.0 {
        return f64::INFINITY;
    }
    (1.0 + rho) / (1.0 - rho) * growth / (1.0 - eta).powi(2) - 1.0
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
            francis_iteration_cap(n) as f64 * nf * (nf - 1.0),
        ),
        repeat(rotation(30, eps), (nf / 2.0).floor()),
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
    let d = (n as f64).sqrt() * francis_vectors(n, eps);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}

/// Per-row error of the accumulated singular vectors `Û` (or `V̂`) against the
/// exactly orthogonal factors of [`svd`]: the bidiagonalization's reflectors
/// (at most `k` per side, length `≤ M`), every QR rotation (`givens`,
/// `θ₅`), counted for both sides at [`svd`]'s totals, and the 2×2 pairs'
/// factor rotations (each row in at most two pairs — a fired pair's zeroed
/// superdiagonal never reforms — two one-sided rotations each with
/// `triangular_svd`'s `(ĉ, ŝ)` carrying `θ₃₅`).
pub fn svd_vectors(rows: usize, cols: usize, eps: f64) -> f64 {
    let (big, small) = (rows.max(cols), rows.min(cols));
    let k = small as f64;
    let rotations = (SVD_ITERATION_CAP as f64) * 2.0 * (k - 1.0) + (k - 1.0) * (k - 1.0);
    compose(&[
        repeat(householder(big, eps), k),
        repeat(rotation(5, eps), rotations),
        repeat(2.0 * rotation(35, eps), 2.0),
    ])
}

/// The residual of the returned SVD factors, `‖Â − Û·Σ̂·V̂ᵀ‖_F ≤ η_R·‖Â‖_F`,
/// and the orthogonality of each, `‖Ûᵀ Û − I‖_F ≤ η_O` (likewise `V̂`), a
/// priori. As [`schur_factors`], with `‖D_U‖_F, ‖D_V‖_F ≤ d = √M·η_U`
/// ([`svd_vectors`]): `ÛΣ̂V̂ᵀ − ŨΣ̂Ṽᵀ = D_UΣ̂Ṽᵀ + ŨΣ̂D_Vᵀ + D_UΣ̂D_Vᵀ`, so
/// `η_R = η + (2d + d²)(1 + η)` and `η_O = 2d + d²`.
pub fn svd_factors(rows: usize, cols: usize, eps: f64) -> (f64, f64) {
    let eta = svd(rows, cols, eps);
    let d = (rows.max(cols) as f64).sqrt() * svd_vectors(rows, cols, eps);
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

/// Inexact roundings on one entry's path through one 2×2 direct
/// diagonalization (`triangular_svd`): the value chains run `≤ 35`
/// roundings; `40` covers values and factor rotations alike.
const DLASV2_ROUNDINGS: f64 = 40.0;

/// Each bidiagonal entry joins at most two 2×2 blocks over a run: firing
/// zeroes the shared superdiagonal permanently, so the pair can never
/// reform — an entry is the top of one pair and the bottom of one pair at
/// most.
const DLASV2_OCCURRENCES: f64 = 2.0;

/// Per-entry rounding count of the Francis value path: the Hessenberg
/// reduction (`≤ 2(n−2)` one-sided applications of length `≤ n−1`, both
/// paths), `steps` Francis steps, the `⌊n/2⌋` `dlanv2` blocks (rotation
/// `θ_30` applied two-sided, `36` per entry; block entries `40`), and the
/// `θ_3` read-off. The eigenvalues-only windows only exclude entries, so
/// the Schur-path count covers both.
pub fn francis_roundings(n: usize, steps: usize) -> f64 {
    let hessenberg = 2.0 * n.saturating_sub(2) as f64 * reflector_roundings(n.saturating_sub(1));
    let chase = steps as f64 * FRANCIS_STEP_ROUNDINGS;
    let blocks = (n / 2) as f64 * 36.0 + DLASV2_ROUNDINGS + 3.0;
    hessenberg + chase + blocks
}

/// Per-entry rounding count of the accumulated Schur vectors: the
/// Hessenberg reflectors from the right, every Francis step's right
/// applications, and the `dlanv2` rotations (`36` covers the one-sided
/// application).
pub fn francis_vectors_roundings(n: usize, steps: usize) -> f64 {
    let hessenberg = n.saturating_sub(2) as f64 * reflector_roundings(n.saturating_sub(1));
    let chase = steps as f64 * FRANCIS_VECTORS_STEP_ROUNDINGS;
    let blocks = (n / 2) as f64 * 36.0;
    hessenberg + chase + blocks
}

/// Per-entry rounding count of the Golub–Kahan value path: the
/// bidiagonalization (`k` left reflectors of length `≤ M`, `k−1` right of
/// length `≤ k−1`; [`reflector_roundings`] covers both the `U/V` and the
/// values-only `larfg` (`3m+4`) constructions), the instrumented sweeps
/// and chases, and the capped 2×2 direct diagonalizations.
pub fn svd_roundings(rows: usize, cols: usize, sweeps: usize, chases: usize) -> f64 {
    let (big, k) = (rows.max(cols), rows.min(cols));
    let bidiagonal = k as f64 * reflector_roundings(big)
        + k.saturating_sub(1) as f64 * reflector_roundings(k.saturating_sub(1));
    let iteration = sweeps as f64 * DBDSQR_SWEEP_ROUNDINGS + chases as f64 * DBDSQR_CHASE_ROUNDINGS;
    let direct = DLASV2_OCCURRENCES * DLASV2_ROUNDINGS;
    bidiagonal + iteration + direct
}

/// Per-entry rounding count of the accumulated `U`/`V`: `U` meets the `k`
/// left reflectors and `V` the `k−1` right ones (the max covers both); per
/// sweep an entry meets two rotations plus the shift chain (`40`); per
/// chase the pivot row meets `≤ k` rotations, others one.
pub fn svd_vectors_roundings(rows: usize, cols: usize, sweeps: usize, chases: usize) -> f64 {
    let (big, k) = (rows.max(cols), rows.min(cols));
    let bidiagonal = k as f64 * reflector_roundings(big);
    let per_sweep = 2.0 * ROTATION_ROUNDINGS + 20.0;
    let per_chase = ROTATION_ROUNDINGS * k as f64;
    let iteration = sweeps as f64 * per_sweep + chases as f64 * per_chase;
    let direct = DLASV2_OCCURRENCES * DLASV2_ROUNDINGS;
    bidiagonal + iteration + direct
}

/// Per-entry rounding count of pivoted QR: `min(rows, cols)` one-sided
/// reflectors of length `≤ rows` on `R` and on the `Q` rows alike. Exact
/// counts, no instrumentation: the loop trip count is structural.
pub fn col_piv_qr_roundings(rows: usize, cols: usize) -> f64 {
    rows.min(cols) as f64 * reflector_roundings(rows)
}

/// Probabilistic Francis value bound at `steps` instrumented steps: one
/// `γ̃` over the entry path, composed with the deterministic deflation
/// floors exactly as in [`francis`].
pub fn francis_tilde(n: usize, eps: f64, steps: usize) -> f64 {
    let roundings = gamma_tilde(francis_roundings(n, steps), eps);
    compose(&[roundings, repeat(4.0 * eps, n.saturating_sub(1) as f64)])
}

/// Probabilistic Schur-vectors per-row bound at `steps` instrumented steps.
pub fn francis_vectors_tilde(n: usize, eps: f64, steps: usize) -> f64 {
    gamma_tilde(francis_vectors_roundings(n, steps), eps)
}

/// Probabilistic `(residual, orthogonality)` pair, same algebra as
/// [`schur_factors`].
pub fn schur_factors_tilde(n: usize, eps: f64, steps: usize) -> (f64, f64) {
    let eta = francis_tilde(n, eps, steps);
    let d = (n as f64).sqrt() * francis_vectors_tilde(n, eps, steps);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}

/// Probabilistic SVD value bound at `sweeps`/`chases` instrumented
/// iterations: one `γ̃` over the entry path, composed with the
/// deterministic deflation floors exactly as in [`svd`].
pub fn svd_tilde(rows: usize, cols: usize, eps: f64, sweeps: usize, chases: usize) -> f64 {
    let roundings = gamma_tilde(svd_roundings(rows, cols, sweeps, chases), eps);
    let k = rows.min(cols);
    compose(&[
        roundings,
        repeat(dbdsqr_tol(eps).max(eps), k.saturating_sub(1) as f64),
        repeat(eps, k as f64),
    ])
}

/// Probabilistic `U`/`V` per-row bound at instrumented iterations.
pub fn svd_vectors_tilde(rows: usize, cols: usize, eps: f64, sweeps: usize, chases: usize) -> f64 {
    gamma_tilde(svd_vectors_roundings(rows, cols, sweeps, chases), eps)
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
    let d = (rows.max(cols) as f64).sqrt() * svd_vectors_tilde(rows, cols, eps, sweeps, chases);
    let orthogonality = 2.0 * d + d * d;
    (eta + orthogonality * (1.0 + eta), orthogonality)
}

/// Probabilistic pivoted-QR bound: one `γ̃` over the entry path with the
/// `Q`-row accumulation algebra of [`col_piv_qr`].
pub fn col_piv_qr_tilde(rows: usize, cols: usize, eps: f64) -> f64 {
    let eta = gamma_tilde(col_piv_qr_roundings(rows, cols), eps);
    eta + (rows as f64).sqrt() * eta * (1.0 + eta)
}

/// Inexact roundings on one tridiagonal entry's path through one implicit-QL
/// sweep (`symmetric_qr/ql.rs`): the Wilkinson-shift formation (`≤ 15`),
/// the two overlapping rotations touching the entry (`2 × 24`, the
/// `θ₆` Givens plus bulge combinations each), the tail chain (`7`), and
/// the shift accumulation (`1`). `75` covers with margin.
const QL_SWEEP_ROUNDINGS: f64 = 75.0;

/// Per-entry rounding count of the tridiagonal-QL value path: the
/// reduction (`n − 2` rank-2 updates, each a sequential `θ` of length
/// `17n + 42` per [`householder_rank2`]'s derivation), `sweeps`
/// instrumented sweeps, and the final `dₗ + σ` rounding. The eigenvalues-only
/// and vector paths share the iteration; deflations stay deterministic.
pub fn ql_roundings(n: usize, sweeps: usize) -> f64 {
    let nf = n as f64;
    let reduction = (nf - 2.0).max(0.0) * (17.0 * nf + 42.0);
    let iteration = sweeps as f64 * QL_SWEEP_ROUNDINGS;
    reduction + iteration + 1.0
}

/// Per-row rounding count of the accumulated QL eigenvectors: the
/// reduction's reflectors one-sided plus two rotations (`θ₆`, `8`
/// roundings each with the application) per sweep.
pub fn ql_vectors_roundings(n: usize, sweeps: usize) -> f64 {
    let nf = n as f64;
    let reduction = (nf - 2.0).max(0.0) * reflector_roundings(n);
    let iteration = sweeps as f64 * 16.0;
    reduction + iteration
}

/// Probabilistic QL value bound at `sweeps` instrumented sweeps: one `γ̃`
/// over the entry path, composed with the deterministic deflation floors
/// exactly as in [`ql`].
pub fn ql_tilde(n: usize, eps: f64, sweeps: usize) -> f64 {
    let roundings = gamma_tilde(ql_roundings(n, sweeps), eps);
    compose(&[roundings, repeat(eps, n as f64 - 1.0)])
}

/// Probabilistic QL-vectors per-row bound at instrumented sweeps.
pub fn ql_vectors_tilde(n: usize, eps: f64, sweeps: usize) -> f64 {
    gamma_tilde(ql_vectors_roundings(n, sweeps), eps)
}

/// The probabilistic oracle's failure budget: with `λ = λ₇` the per-product
/// failure probability is `≈ 7e-11` in every format, so the runs' union
/// (thousands of products each over hundreds of firing cases) stays
/// negligible. Pins the knob: lowering `λ` to chase vacuous bounds would
/// explode this.
#[test]
fn probabilistic_oracle_failure_budget_stays_negligible() {
    for eps in [
        f64::EPSILON,
        f32::EPSILON as f64,
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
