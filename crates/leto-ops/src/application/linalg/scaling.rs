//! Exact power-of-two balancing of dense factorization inputs.
//!
//! The dense factorizations form sums of squares, Givens radii, Wilkinson
//! shifts and column norms. For entries near `√max` those overflow and for
//! entries near `√min` they underflow, so without scaling a factorization of
//! `s·A` fails to converge or returns a wrong result long before `s·A` itself
//! leaves the range of the scalar (f64 SVD failed below `2⁻²⁵⁸` and above
//! `2²⁵⁴`). Every routine therefore factors `2⁻ᵏ·A` instead, `k` chosen so the
//! largest entry lies in `[1, 4)`, and multiplies the scale-carrying results
//! (singular values, eigenvalues, the triangular factor) back by `2ᵏ`.
//!
//! **Exactness.** Multiplying by a power of two changes only the exponent, so
//! it is exact whenever neither operand nor result leaves the normal range;
//! orthogonal factors and permutations are scale-invariant and are returned
//! unchanged. `k` is **even**, so `√(2⁻ᵏ·x) = 2^(−k/2)·√x` exactly and every
//! square root the algorithms take commutes with the scaling bit for bit.
//! When the largest entry already lies in `[1, 4)`, `k = 0` and the input is
//! factored unchanged, so results at unit scale are bitwise those of the
//! unscaled algorithm.
//!
//! **Range after scaling.** Every entry is below `4`, so an `m × n` matrix
//! has `‖A‖_F < 4·√(mn)`: sums of squares stay below `16·mn`, far inside every
//! supported format (`F16` holds `65504`, i.e. up to `mn ≈ 4000`). Entries
//! whose squares underflow are below `√(min positive)` against a largest
//! entry of at least `1`, which is below `ε` for every supported format, so
//! dropping them stays inside the factorizations' backward error.

use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// The even exponent `k` with `max|xᵢ|·2⁻ᵏ ∈ [1, 4)`, or `None` when `values`
/// is empty, all zero, or holds a non-finite entry (which the caller's
/// validation reports).
pub(crate) fn balancing_exponent<T: RealScalar>(
    values: impl IntoIterator<Item = T>,
) -> Option<i32> {
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
    let exponent = largest.binary_exponent()?;
    // Round toward −∞ to an even exponent: e ∈ {2j, 2j+1} ↦ 2j.
    Some(exponent - exponent.rem_euclid(2))
}

/// Multiply every entry of `values` by `2^exponent` (exact while the results
/// stay normal).
pub(crate) fn scale_by_power_of_two<T: RealScalar>(values: &mut [T], exponent: i32) {
    for value in values {
        *value = value.scale_binary(exponent);
    }
}

/// `matrix` multiplied by `2⁻ᵏ` as an owned contiguous copy, with `k`; `None`
/// when no scaling applies (`k = 0`, a zero or empty matrix, or a non-finite
/// entry), in which case the caller factors `matrix` itself.
pub(crate) fn balanced<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Option<(Array2<T>, i32)> {
    let exponent = match matrix.as_slice() {
        Some(slice) => balancing_exponent(slice.iter().copied()),
        None => balancing_exponent(matrix.iter().copied()),
    }?;
    if exponent == 0 {
        return None;
    }
    let mut values = match matrix.as_slice() {
        Some(slice) => slice.to_vec(),
        None => matrix.iter().copied().collect(),
    };
    scale_by_power_of_two(&mut values, -exponent);
    let array = Array2::from_shape_vec(matrix.shape(), values)
        .expect("invariant: a copy of a view keeps its shape and length");
    Some((array, exponent))
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

#[cfg(test)]
mod tests {
    use super::{balancing_exponent, restore};
    use leto::LetoError;

    #[test]
    fn balancing_exponent_is_even_and_lands_in_one_to_four() {
        for (value, expected) in [
            (1.0_f64, 0),
            (3.99, 0),
            (4.0, 2),
            (0.5, -2),
            (0.25, -2),
            (0.2, -4),
        ] {
            let exponent = balancing_exponent([value, -value / 3.0]).expect("finite nonzero");
            assert_eq!(exponent, expected, "{value}");
            let scaled = value * 2.0_f64.powi(-exponent);
            assert!((1.0..4.0).contains(&scaled), "{value} -> {scaled}");
        }
        assert_eq!(balancing_exponent([0.0_f64, -0.0]), None);
        assert_eq!(balancing_exponent([1.0_f64, f64::INFINITY]), None);
        assert_eq!(balancing_exponent(Vec::<f64>::new()), None);
        // Subnormal maximum: 2⁻¹⁰⁷⁴ needs k = −1074, even.
        assert_eq!(balancing_exponent([f64::from_bits(1)]), Some(-1074));
    }

    #[test]
    fn restore_reports_an_unrepresentable_result() {
        let mut values = [1.5_f64, 3.0];
        assert!(matches!(
            restore(&mut values, 1023, "probe"),
            Err(LetoError::Overflow { reason: "probe" })
        ));
        let mut values = [1.5_f64, -3.0];
        restore(&mut values, 4, "probe").expect("in range");
        assert_eq!(values, [24.0, -48.0]);
    }
}
