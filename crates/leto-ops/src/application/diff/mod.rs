//! Finite-difference differentiation schemes for 1-D arrays.
//!
//! SSOT for generic scalar finite-difference operators in the Atlas stack.
//! All operators are generic over `T: RealField + FloatElement + Copy`.
//!
//! ## Schemes
//!
//! | Scheme | Order | Notes |
//! |--------|-------|-------|
//! | `Forward` | 1st | One-sided, uses forward stencil |
//! | `Backward` | 1st | One-sided, uses backward stencil |
//! | `Central` | 2nd | Symmetric; preferred for interior points |
//! | `ForwardSecondOrder` | 2nd | Three-point one-sided forward |
//! | `BackwardSecondOrder` | 2nd | Three-point one-sided backward |
//!
//! ## Boundary handling
//!
//! At the boundary points where the primary stencil would go out-of-range,
//! all operators fall back to the nearest-equivalent one-sided scheme of the
//! same (or lower) accuracy.

pub mod finite_difference;
mod schemes;
pub mod three_dimensional;

pub use finite_difference::FiniteDifference;
pub use schemes::FiniteDifferenceScheme;
pub use three_dimensional::{FiniteDifference3D, FiniteDifference3DScheme};
