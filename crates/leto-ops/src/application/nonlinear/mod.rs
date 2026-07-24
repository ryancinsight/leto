//! Non-linear solver algorithms — SSOT for the Atlas simulation stack.
//!
//! ## Algorithms
//!
//! - [`anderson`]: Anderson Acceleration (Type-II MGS-QR variant, VecDeque history)
//!   for fixed-point iterations. Achieves superlinear convergence without an
//!   explicit Jacobian (Anderson 1965; Walker & Ni 2011).
//!
//! ## Design
//!
//! All algorithms are generic over `T: eunomia::RealField + eunomia::FloatElement`
//! and operate on `leto::{Array1, Array2}`. No dependency on domain crates.

pub mod anderson;
pub(crate) mod linalg;

pub use anderson::{AndersonAccelerator, AndersonConfig, AndersonMethod};
