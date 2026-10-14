//! Interval search and node-validation helpers for interpolation.

use std::cmp::Ordering;

use eunomia::RealField;
use leto::{LetoError, Result};

/// Binary-search for the interval `[x_data[i], x_data[i+1])` containing `x`.
/// Returns `None` when `x` is outside the domain (extrapolation).
pub(super) fn find_interval<T: PartialOrd + Copy>(x_data: &[T], x: &T) -> Option<usize> {
    match x_data.binary_search_by(|p| {
        if p < x {
            Ordering::Less
        } else if p > x {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }) {
        Ok(idx) => Some(idx.min(x_data.len().saturating_sub(2))),
        Err(idx) => {
            if idx == 0 || idx >= x_data.len() {
                None
            } else {
                Some(idx - 1)
            }
        }
    }
}

/// Validate and return `Err` when `x` is outside `(lo, hi)`.
pub(super) fn check_bounds<T: RealField>(x: T, lo: T, hi: T) -> Result<()> {
    if x < lo || x > hi {
        Err(LetoError::InvalidInput(
            "query point is outside the data range".into(),
        ))
    } else {
        Ok(())
    }
}

/// Validate that `x_data` is non-empty, long enough, and strictly increasing.
pub(super) fn validate_nodes<T: RealField>(x_data: &[T], min_len: usize) -> Result<()> {
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
