//! Internal search helpers for interpolation.

use std::cmp::Ordering;

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
