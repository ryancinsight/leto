//! Householder QR factorization `A = Q R` (compact reflector storage).
//!
//! # Theorem (existence and the Householder construction)
//! Every `A ∈ ℝ^{m×n}` with `m ≥ n` admits `A = Q R` with `Q ∈ ℝ^{m×m}`
//! orthogonal and `R ∈ ℝ^{m×n}` upper-triangular.
//!
//! *Proof (constructive — exactly the reflectors [`qr_decompose`] applies).* For
//! pivot column `k`, let `x = A[k.., k]`. The Householder reflector
//! `H_k = I − β_k v_k v_kᵀ`, with `v_k = x − α e₁`, `α = −sign(x₀)‖x‖₂`, and
//! `β_k = 2 / (v_kᵀ v_k)`, is symmetric and orthogonal (`H_kᵀ H_k = I − 2β v vᵀ +
//! β² v (vᵀv) vᵀ = I` since `β vᵀv = 2`) and satisfies `H_k x = α e₁`, zeroing
//! entries `k+1..m` of column `k` while fixing rows/columns `0..k`. Hence
//! `H_{n-1} ⋯ H_0 A = R` is upper-triangular, and with
//! `Q = H_0 ⋯ H_{n-1}` (a product of orthogonal matrices, therefore orthogonal)
//! we obtain `A = Q R`. Choosing `α = −sign(x₀)‖x‖₂` makes `v₀ = x₀ − α` a sum of
//! like-signed magnitudes (no cancellation), giving a backward-stable reflector. ∎
//!
//! # Corollary (least-squares normal-equation-free solve)
//! For full-column-rank `A` (`m ≥ n`), `x̂ = argminₓ ‖A x − b‖₂` is the unique
//! solution of `R₁ x̂ = (Qᵀ b)[0..n]`, where `R₁` is the top `n × n` block of `R`.
//!
//! *Proof.* `Qᵀ` is orthogonal, so `‖A x − b‖₂ = ‖Qᵀ(A x − b)‖₂ = ‖R x − Qᵀb‖₂`.
//! Partitioning into the first `n` and last `m−n` rows, `R x` only reaches the
//! first `n` (lower rows of `R` are zero), so the last `m−n` rows contribute the
//! fixed residual `‖(Qᵀb)[n..m]‖₂` independent of `x`; the total is minimized by
//! annihilating the first `n` rows, i.e. `R₁ x̂ = (Qᵀb)[0..n]`, solvable by
//! back-substitution since full rank ⇒ `R₁` nonsingular.
//! [`QrDecomposition::solve_least_squares`] realizes this: it forms `Qᵀb` by
//! applying the stored reflectors and
//! back-substitutes — `Q` is never materialized. ∎
//!
//! Vertical structure (SoC, mirroring `bunch_kaufman/{decompose, solve}`): the
//! `decompose` leaf owns the factorization kernel ([`qr_decompose`]); the `solve`
//! leaf owns the least-squares solve ([`solve_least_squares`]); the
//! `decomposition` leaf owns the [`QrDecomposition`] type and its `Q`/`R`
//! materialization accessors. Generic over [`crate::RealScalar`], native precision.

pub mod decompose;
mod decomposition;
pub mod solve;

pub use decompose::qr_decompose;
pub use decomposition::QrDecomposition;
pub use solve::solve_least_squares;
