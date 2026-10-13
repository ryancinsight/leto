//! A-priori column-pivoted QR bounds.

use eunomia::FloatElement;

/// Column-pivoted QR of an `rows × cols` matrix: `A·P + E = Q̃·R̂` with
/// `‖E‖_F ≤ η·‖A‖_F` (`min(rows, cols)` one-sided reflectors of length
/// `≤ rows` on the columns of `R`), and each row of the accumulated `Q̂`
/// within the same `η` of `Q̃` (the same reflectors applied to the rows of
/// `I`), so `‖A·P − Q̂·R̂‖_F ≤ (η + √rows·η·(1 + η))·‖A‖_F`.
pub fn col_piv_qr(rows: usize, cols: usize, eps: f64) -> f64 {
    let eta = super::repeat(
        super::householder(rows, eps),
        f64::from_count(rows.min(cols)),
    );
    eta + (f64::from_count(rows)).sqrt() * eta * (1.0 + eta)
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
    let (m, steps) = (f64::from_count(rows), f64::from_count(rows.min(cols)));
    let u = eps / 2.0;
    let eta = super::householder(rows, eps);
    let key = super::gamma(m + 1.0, eps);
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

/// Per-entry rounding count of pivoted QR: `min(rows, cols)` one-sided
/// reflectors of length `≤ rows` on `R` and on the `Q` rows alike. Exact
/// counts, no instrumentation: the loop trip count is structural.
fn col_piv_qr_roundings(rows: usize, cols: usize) -> f64 {
    f64::from_count(rows.min(cols)) * super::reflector_roundings(rows)
}

/// Probabilistic pivoted-QR bound: one `γ̃` over the entry path with the
/// `Q`-row accumulation algebra of [`col_piv_qr`].
pub fn col_piv_qr_tilde(rows: usize, cols: usize, eps: f64) -> f64 {
    let eta = super::gamma_tilde(col_piv_qr_roundings(rows, cols), eps);
    eta + (f64::from_count(rows)).sqrt() * eta * (1.0 + eta)
}
