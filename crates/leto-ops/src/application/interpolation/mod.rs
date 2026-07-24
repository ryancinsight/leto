//! 1-D interpolation trait and implementations.
//!
//! SSOT for univariate interpolation in the Atlas simulation stack.
//! All implementations are generic over `T: RealField + FloatElement + Copy`.
//!
//! ## Implementations
//!
//! | Type | Method | Points required |
//! |------|--------|----------------|
//! | [`LinearInterpolation`] | Piecewise-linear | ≥ 2 |
//! | [`CubicSplineInterpolation`] | Natural cubic spline | ≥ 3 |
//! | [`LagrangeInterpolation`] | Barycentric Lagrange | ≥ 2 |
//!
//! ## Usage
//!
//! ```rust
//! use leto_ops::application::interpolation::{LinearInterpolation, Interpolation1D};
//!
//! let interp = LinearInterpolation::<f64>::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0])
//!     .expect("valid data");
//! let y = interp.interpolate(0.5_f64).expect("in range");
//! assert!((y - 0.5_f64).abs() < 1e-12);
//! ```

pub mod cubic_spline;
pub mod lagrange;
pub mod linear;
mod utils;

pub use cubic_spline::CubicSplineInterpolation;
pub use lagrange::LagrangeInterpolation;
pub use linear::LinearInterpolation;

use eunomia::RealField;
use leto::{LetoError, Result};

/// Trait for 1-D interpolation methods.
///
/// All implementations must be `Send + Sync` so they can be used in parallel
/// solver loops.  Extrapolation (query outside `[x_min, x_max]`) always returns
/// [`LetoError::InvalidInput`].
pub trait Interpolation1D<T: RealField + Copy>: Send + Sync {
    /// Interpolate the function at a single query point `x`.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] — `x` is outside the data domain.
    fn interpolate(&self, x: T) -> Result<T>;

    /// Inclusive data domain `(x_min, x_max)`.
    fn bounds(&self) -> (T, T);

    /// Interpolate at every point in `xs`, short-circuiting on the first error.
    fn interpolate_many(&self, xs: &[T]) -> Result<Vec<T>> {
        xs.iter().map(|&x| self.interpolate(x)).collect()
    }
}

/// Validate and return `Err` when `x` is outside `(lo, hi)`.
pub(super) fn check_bounds<T: RealField + Copy>(x: T, lo: T, hi: T) -> Result<()> {
    if x < lo || x > hi {
        Err(LetoError::InvalidInput(
            "query point is outside the data range".into(),
        ))
    } else {
        Ok(())
    }
}

/// Validate that `x_data` is non-empty, long enough, and strictly increasing.
pub(super) fn validate_nodes<T: RealField + Copy>(x_data: &[T], min_len: usize) -> Result<()> {
    if x_data.len() < min_len {
        return Err(LetoError::InvalidInput(format!(
            "Need at least {min_len} points for interpolation"
        )));
    }
    if !x_data.windows(2).all(|w| w[0] < w[1]) {
        return Err(LetoError::InvalidInput(
            "x_data must be strictly increasing (no duplicate nodes)".into(),
        ));
    }
    Ok(())
}
