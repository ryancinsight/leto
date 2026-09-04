//! Generic 3-D finite-difference operators.
//!
//! SSOT extension of the 1-D [`FiniteDifference`](super::FiniteDifference) and
//! the 2-D [`crate::laplacian_2d_into`] operators to three spatial dimensions.
//! Replaces the per-consumer FD kernels previously duplicated in
//! `kwavers-math`, `cfd-math`, and `helios-imaging`.
//!
//! The provider covers the families the FDTD / acoustic / CFD / RT kernels
//! actually call:
//!
//! | Scheme | Order | Stencil | dst shape on diff axis |
//! |--------|-------|---------|------------------------|
//! | [`FiniteDifference3DScheme::CentralSecondOrder`] | O(Δx²) 3-point | symmetric interior | matches `field` |
//! | [`FiniteDifference3DScheme::CentralFourthOrder`] | O(Δx⁴) 5-point + 2nd/1st fall-back | matches `field` |
//! | [`FiniteDifference3DScheme::CentralSixthOrder`] | O(Δx⁶) 7-point + 4th/2nd/1st fall-back | matches `field` |
//! | [`FiniteDifference3DScheme::StaggeredForward`] | O(Δx) Yee face | one cell smaller |
//! | [`FiniteDifference3DScheme::StaggeredBackward`] | O(Δx) cell-on-integer-grid | matches `field` |
//!
//! All stencils are explicit, allocation-free, and operate on caller-supplied
//! `ArrayView3<T>` slices writing into pre-allocated `&mut Array3<T>` buffers.
//!
//! ```rust,ignore
//! use leto_ops::{FiniteDifference3D, FiniteDifference3DScheme};
//!
//! let op = FiniteDifference3D::central_fourth_order(0.001, 0.001, 0.001)?;
//! let grad_x = op.apply_x(field.view());
//! ```

#![expect(
    clippy::unwrap_used,
    reason = "ratchet LETO-UNWRAP-1: pre-existing debt"
)]

use eunomia::FloatElement;

mod central;
mod operator;
mod staggered;
#[cfg(test)]
mod tests;

#[inline]
pub(super) fn f<T: FloatElement>(v: f64) -> T {
    T::from_f64(v)
}

/// Stencil family + kernel ordering for [`FiniteDifference3D`].
///
/// Variant naming follows the FDTD / CFD cell-sweep vocabulary. Note that
/// [`Self::StaggeredBackward`] is the kwavers-side convention: dst shape
/// matches `field.shape` (the integer-cell arrangement rather than the
/// half-cell staggered arrangement), and `i=0` falls back to a forward
/// difference. This preserves the kwavers-side Yee-coupling solver contract
/// bit-equivalent to the previous `StaggeredGridOperator::apply_backward_*_into`
/// kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDifference3DScheme {
    /// Second-order central difference `dst = (f[i+1] − f[i−1]) / (2Δ)`.
    CentralSecondOrder,
    /// Fourth-order central `dst = (−f[i+2] + 8f[i+1] − 8f[i−1] + f[i−2]) / (12Δ)`.
    CentralFourthOrder,
    /// Sixth-order central:
    /// `dst = (−f[i+3] + 9f[i+2] − 45f[i+1] + 45f[i−1] − 9f[i−2] + f[i−3]) / (60Δ)`.
    CentralSixthOrder,
    /// Yee staggered forward face derivative:
    /// `dst[i,j,k] = (f[i+1,j,k] − f[i,j,k]) / Δ`. `dst` has one fewer cell on
    /// the differentiated axis.
    StaggeredForward,
    /// Yee coupling-field backward sweep (kwavers-side convention):
    /// `dst[0,j,k] = (f[1,j,k] − f[0,j,k]) / Δ` (forward fall-back at `i=0`),
    /// `dst[i>0,j,k] = (f[i,j,k] − f[i−1,j,k]) / Δ`. `dst` shape matches `field`.
    StaggeredBackward,
}

pub use operator::FiniteDifference3D;
