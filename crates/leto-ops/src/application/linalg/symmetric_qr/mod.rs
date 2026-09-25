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
//! The unscaled reduction forms `Σxᵢ²` and `vᵀv`; for entries near the square
//! root of the largest (or smallest) finite value these overflow (underflow)
//! and the result is garbage, not an error — `F16` overflows `3·200²` at
//! `‖A‖ = 200`. So the matrix norm is first compared against the LAPACK
//! `dsyev`-style safe range `[rmin, rmax]` (`linalg::scaling`,
//! `linalg::thresholds::safe_range`); only when it falls outside is the
//! matrix balanced, by the even power of two bringing its largest entry into
//! `[1, 4)` (always inside `[rmin, rmax]`). A norm already in range is
//! factored completely unscaled, bit-for-bit identical to a hypothetical
//! unconditionally-scaled path only when its largest entry already lay in
//! `[1, 4)` — otherwise the two differ, which is the point: scaling by a
//! power of two is exact only while every scaled entry stays representable,
//! and an entry far below the largest one can underflow under a scale chosen
//! for the largest (see `linalg::scaling`'s module documentation), so the
//! unscaled path is preferred whenever it is safe. Eigenvectors are unchanged
//! by either path and the eigenvalues are multiplied back by `2ᵏ` when
//! scaling was applied (the EISPACK `tred2` row scaling and the LAPACK
//! `dsyev` norm scaling serve the same purpose). After scaling every quantity the algorithm
//! forms is bounded by a small multiple of `n`: `|aᵢⱼ| < 4`, `Σxᵢ² < 16n`,
//! `vᵀv ≤ 4‖x‖² < 64n`, and every entry of every reduced matrix and every
//! `dᵢ, eᵢ` is at most `‖A‖₂ < 4n` (orthogonal similarity preserves the 2-norm)
//! — far inside the range of every supported format for any `n` it can index
//! (`F16`: `64n < 65504` up to `n ≈ 1000`). The
//! QL shift ratio `p = (d_{l+1} − dₗ)/(2eₗ)` is unbounded as `eₗ → 0`, and is
//! consumed only through the scaled `hypot`. Squares that underflow come from
//! entries below `√(min positive)` relative to a largest entry of 1, which is
//! below `ε` for every supported format (`√` of the smallest subnormal: `F16`
//! 2.4e-4 against `ε = 9.8e-4`), so dropping them stays inside the
//! backward-error bound.
//!
//! Every operation runs in the precision of `T`.

mod ql;
mod reduce;
mod workspace;

pub use workspace::{symmetric_eigen_qr, SymmetricEigenWorkspace};
