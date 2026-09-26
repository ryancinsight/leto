//! The matrix-tier gate: the minimal power-of-two move bringing `‖A‖_max`
//! into a routine's safe range, and its restoration.

use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// The largest finite magnitude in `values`, or `None` when it is empty, all
/// zero, or holds a non-finite entry (which the caller's validation reports).
pub(crate) fn largest_magnitude<T: RealScalar>(values: impl IntoIterator<Item = T>) -> Option<T> {
    let mut largest = T::ZERO;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        let magnitude = value.abs();
        if magnitude > largest {
            largest = magnitude;
        }
    }
    if largest == T::ZERO {
        None
    } else {
        Some(largest)
    }
}

/// The minimal-magnitude integer `k` (either sign) with `largest·2⁻ᵏ`
/// inside `[rmin, rmax]`. `largest` is assumed already outside it (the
/// caller checks); `rmin < rmax`, both positive and finite.
fn minimal_exponent_to_range<T: RealScalar>(largest: T, rmin: T, rmax: T) -> i32 {
    if largest > rmax {
        // Scale down: the smallest k > 0 with `largest / 2^k <= rmax`.
        let mut k = largest.binary_exponent().unwrap_or(0) - rmax.binary_exponent().unwrap_or(0);
        if k < 1 {
            k = 1;
        }
        while largest.scale_binary(-k) > rmax {
            k += 1;
        }
        while k > 1 && largest.scale_binary(-(k - 1)) <= rmax {
            k -= 1;
        }
        k
    } else {
        // largest < rmin: scale up (k negative), the smallest |k| with
        // `largest / 2^k >= rmin`.
        let mut k = largest.binary_exponent().unwrap_or(0) - rmin.binary_exponent().unwrap_or(0);
        if k > -1 {
            k = -1;
        }
        while largest.scale_binary(-k) < rmin {
            k -= 1;
        }
        while k < -1 && largest.scale_binary(-(k + 1)) >= rmin {
            k += 1;
        }
        k
    }
}

/// The minimal-move exponent `k` bringing `largest` into `range`, or `None`
/// when it already lies inside (the caller factors its input unscaled).
pub(crate) fn balancing_exponent<T: RealScalar>(largest: T, range: (T, T)) -> Option<i32> {
    let (rmin, rmax) = range;
    if largest >= rmin && largest <= rmax {
        return None;
    }
    Some(minimal_exponent_to_range(largest, rmin, rmax))
}

/// Multiply every entry of `values` by `2^exponent` (exact while the results
/// stay normal).
pub(crate) fn scale_by_power_of_two<T: RealScalar>(values: &mut [T], exponent: i32) {
    for value in values {
        *value = value.scale_binary(exponent);
    }
}

/// A gate's power-of-two bounds (see
/// [`thresholds::homogeneous_safe_range`]): `2^factor_log2` bounds the
/// relied-upon intermediate from above; `2^floor_log2·safmin` is the
/// routine's absolute deflation threshold divided by `2^l ≤ ‖A‖_F/‖A‖_max`
/// ([`norm_ratio_floor_log2`](super::norm_ratio_floor_log2)), so that `floor ≤ ε·2^l·‖A‖_max ≤ ε·‖A‖_F`
/// is what the lower end secures (`0` when it has none beyond `safmin`).
#[derive(Clone, Copy)]
pub(crate) struct GateBound {
    pub(crate) factor_log2: i32,
    pub(crate) floor_log2: i32,
}

impl GateBound {
    /// A bound with no deflation floor.
    pub(crate) fn factor(factor_log2: i32) -> Self {
        Self {
            factor_log2,
            floor_log2: 0,
        }
    }
}

/// The matrix-tier gate over the entries `values`: `Ok(Some(k))`, the
/// minimal move, when `‖values‖_max` lies outside
/// [`thresholds::homogeneous_safe_range`] for `degree` and
/// `bound(values, ‖values‖_max)`; `Ok(None)` when it lies inside or `values`
/// is empty, all zero, or holds a non-finite entry.
///
/// # Errors
///
/// [`LetoError::Overflow`] when the range is empty
/// ([`thresholds::EmptyRange`]): the bound factor exceeds the format's
/// `Ω/smlnum`, so the routine's intermediates cannot be kept finite at this
/// order in `T` whatever the scaling, or no scaling keeps the routine's
/// deflation floor at or below `ε·‖A‖_F` (through its `2^l ≤ ‖A‖_F/‖A‖_max`).
pub(crate) fn gate_exponent<T: RealScalar>(
    values: &[T],
    degree: u32,
    bound: impl FnOnce(&[T], T) -> GateBound,
) -> Result<Option<i32>> {
    let Some(largest) = largest_magnitude(values.iter().copied()) else {
        return Ok(None);
    };
    let GateBound {
        factor_log2,
        floor_log2,
    } = bound(values, largest);
    let range = thresholds::homogeneous_safe_range::<T>(degree, factor_log2, floor_log2).map_err(
        |empty| LetoError::Overflow {
            reason: match empty {
                thresholds::EmptyRange::Intermediates => "the matrix order exceeds the scalar's exponent range: its bounded intermediates cannot be kept finite",
                thresholds::EmptyRange::DeflationFloor => "the matrix order exceeds the scalar's exponent range: no scaling keeps the absolute deflation floor 2^ceil(log2 n)*safmin at or below eps*||A||_F",
            },
        },
    )?;
    Ok(balancing_exponent(largest, range))
}

/// `matrix` multiplied by `2⁻ᵏ` as an owned contiguous copy, `k` from
/// [`gate_exponent`]; `Ok(None)` when no scaling applies, in which case the
/// caller factors `matrix` itself, unscaled.
///
/// # Errors
///
/// As [`gate_exponent`].
pub(crate) fn balanced<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    degree: u32,
    bound: impl FnOnce(&[T], T) -> GateBound,
) -> Result<Option<(Array2<T>, i32)>> {
    let mut values = match matrix.as_slice() {
        Some(slice) => slice.to_vec(),
        None => matrix.iter().copied().collect(),
    };
    let Some(exponent) = gate_exponent(&values, degree, bound)? else {
        return Ok(None);
    };
    scale_by_power_of_two(&mut values, -exponent);
    let array = Array2::from_shape_vec(matrix.shape(), values)
        .expect("invariant: a copy of a view keeps its shape and length");
    Ok(Some((array, exponent)))
}

/// Multiply scale-carrying results back by `2ᵏ`.
///
/// # Errors
///
/// [`LetoError::Overflow`] when a result exceeds the range of `T`: the true
/// value is not representable, so returning it as infinity would be a wrong
/// `Ok`.
pub(crate) fn restore<T: RealScalar>(
    values: &mut [T],
    exponent: i32,
    what: &'static str,
) -> Result<()> {
    scale_by_power_of_two(values, exponent);
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(LetoError::Overflow { reason: what })
    }
}
