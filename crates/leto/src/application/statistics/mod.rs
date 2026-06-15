//! Multivariate summary statistics over a `v × n` observation matrix.
//!
//! These operations consume a rank-2 array under the ndarray-stats / numpy
//! `rowvar = true` convention — **each row is a variable, each column an
//! observation** — and produce a `v × v` summary matrix.
//!
//! # Submodules
//! - [`covariance`](mod@crate::application::statistics::covariance) — [`covariance()`]
//! - [`correlation`](mod@crate::application::statistics::correlation) — [`pearson_correlation()`]

pub mod correlation;
pub mod covariance;

pub use correlation::pearson_correlation;
pub use covariance::covariance;
