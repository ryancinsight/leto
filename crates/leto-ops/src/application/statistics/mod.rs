//! Statistical quality metrics and distribution summaries.
//!
//! Pure functions with no domain dependencies. Both solver and clinical layers
//! import from here; neither depends on the other.
//!
//! These are host-side quality metrics over gathered results, so the module
//! has no GPU trait by design. Live device paths compose the same quantities
//! from elementwise and reduction operators directly.
//!
//! # Submodules
//!
//! | Module | Contract |
//! | --- | --- |
//! | `correlation` | Pearson product-moment correlation. |
//! | `error` | Root-mean-square error, and its two normalizations. |
//! | `snr` | Peak signal-to-noise ratio and its relative-RMSE form. |
//! | `phase` | Phase-shift correlation and its inverse in degrees. |
//! | `percentile` | Inter-percentile range of a value distribution. |
//!
//! Each leaf owns one concern, so a metric added to one family does not grow
//! the others, and the module tree names the concern a reader is looking for
//! rather than a single undifferentiated file.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod correlation;
mod error;
mod percentile;
mod phase;
mod snr;

/// Elementwise tolerance for an `n`-element accumulation in `T`:
/// `c(n)·ε` with a small constant, per the standard summation bound.
/// Shared by the leaf contract tests.
#[cfg(test)]
pub(crate) fn accumulation_tolerance<T: eunomia::RealField>(n: usize) -> T {
    T::from_count(n) * T::from_integer(4) * T::EPSILON
}

pub use correlation::pearson;
pub use error::{normalized_rmse, nrmse, rmse};
pub use percentile::percentile_range;
pub use phase::{phase_error_degrees_for_correlation, phase_shift_correlation_curve};
pub use snr::{psnr, validation_psnr_from_relative_rmse};
