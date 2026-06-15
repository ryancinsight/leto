//! Scalar matrix properties (queries that reduce a matrix to a scalar or count).
//!
//! Each property is a focused leaf module (SRP). Trace is generic over
//! [`Scalar`](crate::Scalar); rank is bounded on [`RealScalar`](crate::RealScalar)
//! because it consumes the singular-value spectrum.

/// Numerical rank via the singular-value spectrum.
pub mod rank;
/// Main-diagonal sum (trace).
pub mod trace;

pub use rank::{matrix_rank, matrix_rank_with_tolerance};
pub use trace::trace;
