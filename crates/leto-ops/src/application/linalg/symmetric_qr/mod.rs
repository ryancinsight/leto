//! Real symmetric eigensolver: Householder tridiagonalization followed by the
//! implicit-shift QL iteration on the tridiagonal (EISPACK `tred2` + `tql2`).
//!
//! # Algorithm
//!
//! 1. **Tridiagonalization** (Golub & Van Loan, *Matrix Computations*, 4th ed.,
//!    Algorithm 8.3.1). Householder reflectors `H₀ … H_{n−3}` reduce
//!    `A = Aᵀ` to `T = Qᵀ A Q`, `Q = H₀ ⋯ H_{n−3}`, with `T` symmetric
//!    tridiagonal. Each step is a symmetric rank-2 update of the trailing block.
//! 2. **Implicit QL** (Bowdler, Martin, Reinsch & Wilkinson 1968, "The QR and
//!    QL algorithms for symmetric matrices", *Numer. Math.* 11, 293–306,
//!    procedure `tql2`). Wilkinson-shifted QL sweeps deflate `T` to
//!    `Λ = Sᵀ T S`; the sweep rotations accumulate into `Qᵀ`, so the
//!    eigenvectors of `A` are the rows of `(Q S)ᵀ`.
//!
//! Work is about `2n³` for the reduction, `4n³/3` to form `Qᵀ`, and `O(n³)` for
//! the rotation accumulation (under two sweeps per eigenvalue), against the
//! classical Jacobi method's `O(n²)` pivot search per rotation.
//!
//! # Accuracy
//!
//! Both stages apply orthogonal transformations, so the computed eigenvalues
//! are the exact eigenvalues of `A + E` with `‖E‖₂ ≤ p(n)·ε·‖A‖₂`, `p` a modest
//! polynomial (Golub & Van Loan §8.3.6 and §8.3.7; Wilkinson, *The Algebraic
//! Eigenvalue Problem*, §5.28 for the reduction). By Weyl's inequality each
//! eigenvalue lies within `‖E‖₂` of the exact one, and by the Davis–Kahan
//! `sin Θ` theorem an invariant subspace separated from the rest of the
//! spectrum by a gap `δ` moves by at most `‖E‖₂/δ`. Deflation uses the
//! precision-exact test `t + |eᵢ| == t` against the norm estimate
//! `t = maxᵢ(|dᵢ| + |eᵢ|)` of the whole tridiagonal (`tql2` grows it with the
//! sweep; fixing it keeps the shift ratio in range for `F16`), so no tolerance
//! literal enters: a sub-diagonal is dropped exactly when it is below the
//! rounding of `‖T‖`.
//!
//! # Range
//!
//! The reduction's reflectors (`householder::reflect_in_place`) normalize
//! their vector to `[1, 2)` before forming `Σxᵢ²` and `vᵀv`, so the reduction
//! and the QL rotations (through the scaled `hypot`) stay degree 1 in the
//! entries. One product does not: the chase's closing correction multiplies
//! two off-diagonals, `e_{l+1}·eₗ ≤ ‖A‖₂²` (`ql.rs`), which overflowed at
//! `f64` `2⁵³⁷` (`0·∞ = NaN`) and underflows symmetrically. The matrix-tier
//! gate (`linalg::scaling`) therefore bounds `‖A‖_max` to the degree-2 range
//! with bound `2^(2r)`, `2^r ≥ ‖A‖_F/‖A‖_max` (derived in `workspace.rs`); an
//! input inside it is factored completely unscaled, and one outside it is
//! moved by the minimal power of two into it, eigenvalues multiplied back
//! (the LAPACK `dsyev` norm scaling, with the overflow threshold rather than
//! `ε/safmin` as the upper end). Eigenvectors are unchanged by either path.
//! Scaling down is exact only while every entry stays representable — an
//! entry far below the largest can underflow (`linalg::scaling`'s
//! exactness note) — so the result is within the backward-error bound, not
//! exact entrywise. The QL shift ratio `p = (d_{l+1} − dₗ)/(2eₗ)` is bounded
//! by `4/ε` through the fixed norm estimate and consumed only through the
//! scaled `hypot`.
//!
//! Every operation runs in the precision of `T`.

mod ql;
mod reduce;
mod workspace;

pub use workspace::{symmetric_eigen_qr, SymmetricEigenWorkspace};
