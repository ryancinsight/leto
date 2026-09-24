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
//! precision-exact test `t + |eᵢ| == t` against the running norm estimate
//! `t = maxᵢ(|dᵢ| + |eᵢ|)` (the `tql2` form), so no tolerance literal enters: a
//! sub-diagonal is dropped exactly when it is below the rounding of `‖T‖`.
//!
//! Every operation runs in the precision of `T`.

mod ql;
mod reduce;
mod workspace;

pub use workspace::{symmetric_eigen_qr, SymmetricEigenWorkspace};
