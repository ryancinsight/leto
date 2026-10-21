//! Compact-WY block Householder reflectors: aggregate `r` reflectors into
//! `Q = I − V T Vᵀ` and apply the block as matrix–matrix products (BLAS-3).
//!
//! A single Householder reflector of width `w` applied to a trailing matrix is a
//! rank-1 update with `O(1)` arithmetic intensity — bandwidth-bound, no
//! contiguous SIMD span when `w` is small (the eig/SVD residual, ADR 0010). The
//! compact-WY representation fuses `r` consecutive reflectors so their combined
//! action on a trailing block is three GEMMs through `Scalar::tiled_gemm`, raising
//! arithmetic intensity to `O(r)` and routing the work onto the tuned SIMD
//! micro-kernel.
//!
//! # Theorem (compact-WY representation, Schreiber–Van Loan 1989)
//! For reflectors `Hⱼ = I − βⱼ vⱼ vⱼᵀ` (`j = 1…r`) with the `vⱼ` as the columns of
//! `V ∈ ℝ^{m×r}`, there is an upper-triangular `T ∈ ℝ^{r×r}` with
//! `H₁ H₂ … H_r = I − V T Vᵀ`. *Proof:* see [`accumulate::build_t`] — induction on
//! `r`, with `T` built columnwise by `T_{0:j,j} = −βⱼ T_{0:j,0:j}(V_{:,0:j}ᵀ vⱼ)`.
//!
//! # Corollary (BLAS-3 application)
//! `Qᵀ C = C − V (Tᵀ (Vᵀ C))` and `C Q = C − ((C V) T) Vᵀ`. Each parenthesised
//! product is a GEMM; the rank-`r` correction is a third GEMM accumulated in
//! place. Applying `Q`/`Qᵀ` is an orthogonal transform, so it preserves the
//! 2-norm/spectrum/singular values up to the standard blocked backward error
//! `O(r·ε‖C‖)` (the summation reorder vs sequential application is bounded, not
//! bitwise — admissible under the differential-tolerance contracts).
//!
//! Leaf modules: [`accumulate`] builds `T` (SSOT), [`apply`] applies the block
//! to a trailing matrix, and [`transpose`] supplies the row-major transpose
//! both need. The block apply depends only on the
//! [`crate::Scalar::tiled_gemm`] backend seam (DIP). Generic over
//! [`crate::RealScalar`]; native precision throughout.

mod accumulate;
mod apply;
mod transpose;

pub(in crate::application::linalg) use apply::apply_block_left;
