//! Exact power-of-two balancing of dense factorization inputs, scaled only
//! when the input norm leaves the LAPACK safe range.
//!
//! The dense factorizations form sums of squares, Givens radii, Wilkinson
//! shifts and column norms. For entries near `√max` those overflow and for
//! entries near `√min` they underflow, so without scaling a factorization of
//! `s·A` fails to converge or returns a wrong result long before `s·A` itself
//! leaves the range of the scalar (f64 SVD failed below `2⁻²⁵⁸` and above
//! `2²⁵⁴`). Following LAPACK `xSYEV`/`xGEEV` (`dsyev.f`'s `ISCALE` block), each
//! routine first compares `‖A‖_max` against the safe range
//! `[rmin, rmax] = [√smlnum, √bignum]` (`smlnum = safmin/ε`,
//! `bignum = 1/smlnum`; [`thresholds::safe_range`](super::thresholds::safe_range)).
//! **When the norm is already inside that range, the matrix is factored
//! completely unscaled** — no power-of-two multiply touches it at all. Only
//! when the norm falls outside does the routine factor `2⁻ᵏ·A` instead, `k`
//! the even exponent bringing the largest entry into `[1, 4)` (always inside
//! `[rmin, rmax]`, see below), and multiply the scale-carrying results
//! (singular values, eigenvalues, the triangular factor) back by `2ᵏ`.
//!
//! **Exactness.** Multiplying by a power of two changes only the exponent, so
//! it is exact whenever neither operand nor result leaves the normal range;
//! orthogonal factors and permutations are scale-invariant and are returned
//! unchanged. `k` is **even**, so `√(2⁻ᵏ·x) = 2^(−k/2)·√x` exactly and every
//! square root the algorithms take commutes with the scaling bit for bit.
//! This is exact only while every scaled entry stays representable: an entry
//! already far below the largest one can underflow to zero under a scale
//! chosen for the largest entry (`diag(1e300, 1e-300)` in `f64`), exactly as
//! LAPACK's own scaling can lose small entries. That loss is bounded by the
//! factorizations' backward-error guarantee (`‖E‖ = O(p(n)·ε·‖A‖)`), never
//! promised as exact — the guarantee this module keeps is over the *whole*
//! computation, not entrywise.
//!
//! **In-range inputs are bitwise unchanged.** When `‖A‖_max` already lies in
//! `[rmin, rmax]`, no scaling is applied — not even the previous
//! bring-to-`[1, 4)` step — so the factorization runs on `A` itself and its
//! result is bit-for-bit that of the unscaled algorithm. This is what keeps
//! ordinary, already-well-scaled inputs (the overwhelming majority) immune to
//! any rounding the balancing step could otherwise introduce.
//!
//! **Range after scaling.** Every entry is below `4`, so an `m × n` matrix
//! has `‖A‖_F < 4·√(mn)`: sums of squares stay below `16·mn`, far inside every
//! supported format (`F16` holds `65504`, i.e. up to `mn ≈ 4000`). Entries
//! whose squares underflow are below `√(min positive)` against a largest
//! entry of at least `1`, which is below `ε` for every supported format, so
//! dropping them stays inside the factorizations' backward error.

use crate::application::linalg::thresholds::{in_product_safe_range, in_safe_range};
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// The largest finite magnitude in `values`, or `None` when it is empty, all
/// zero, or holds a non-finite entry (which the caller's validation reports).
fn largest_magnitude<T: RealScalar>(values: impl IntoIterator<Item = T>) -> Option<T> {
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

/// The even exponent `k` with `max|xᵢ|·2⁻ᵏ ∈ [1, 4)`, or `None` when `largest`
/// is already in range (`k = 0`; the caller factors the input unscaled).
fn even_exponent_to_one_four<T: RealScalar>(largest: T) -> Option<i32> {
    let exponent = largest.binary_exponent()?;
    // Round toward −∞ to an even exponent: e ∈ {2j, 2j+1} ↦ 2j.
    Some(exponent - exponent.rem_euclid(2))
}

/// The even exponent `k` with `max|xᵢ|·2⁻ᵏ ∈ [1, 4)`, or `None` when `values`
/// is empty, all zero, holds a non-finite entry (which the caller's
/// validation reports), or its norm already lies in the LAPACK safe range
/// (`k = 0`; the caller factors the input unscaled).
///
/// For the symmetric tridiagonal QL and Jacobi eigensolvers and the
/// column-pivoted QR, whose internal formulas square each entry **once**
/// (matching the LAPACK safe range's own derivation).
pub(crate) fn balancing_exponent<T: RealScalar>(
    values: impl IntoIterator<Item = T>,
) -> Option<i32> {
    let largest = largest_magnitude(values)?;
    if in_safe_range(largest) {
        return None;
    }
    even_exponent_to_one_four(largest)
}

/// As [`balancing_exponent`], but gated on
/// [`thresholds::in_product_safe_range`](super::thresholds::in_product_safe_range)
/// instead of the plain safe range.
///
/// For the Francis double-shift QR (`schur`, `eigenvalues`) and the
/// Golub–Kahan bidiagonal QR (the SVD family), whose shift/discriminant
/// formulas square an already-squared quantity — see
/// [`thresholds::product_safe_range`](super::thresholds::product_safe_range)
/// for the evidence and the filed defect this narrower gate works around.
pub(crate) fn balancing_exponent_for_products<T: RealScalar>(
    values: impl IntoIterator<Item = T>,
) -> Option<i32> {
    let largest = largest_magnitude(values)?;
    if in_product_safe_range(largest) {
        return None;
    }
    even_exponent_to_one_four(largest)
}

/// Multiply every entry of `values` by `2^exponent` (exact while the results
/// stay normal).
pub(crate) fn scale_by_power_of_two<T: RealScalar>(values: &mut [T], exponent: i32) {
    for value in values {
        *value = value.scale_binary(exponent);
    }
}

/// `matrix` multiplied by `2⁻ᵏ` as an owned contiguous copy, with `k`, using
/// `exponent_of` (either [`balancing_exponent`] or
/// [`balancing_exponent_for_products`]) to decide `k`; `None` when no scaling
/// applies (`k = 0`, a zero or empty matrix, a non-finite entry, or
/// `‖matrix‖_max` already in `exponent_of`'s safe range), in which case the
/// caller factors `matrix` itself, bit-for-bit unscaled.
/// Which safe range [`balanced_with`] gates scaling on.
#[derive(Clone, Copy)]
enum SafeRangeKind {
    /// The plain LAPACK range ([`balancing_exponent`]).
    Plain,
    /// The narrower range for degree-4 shift formulas
    /// ([`balancing_exponent_for_products`]).
    Products,
}

fn balanced_with<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    kind: SafeRangeKind,
) -> Option<(Array2<T>, i32)> {
    let exponent_of = |values: &[T]| match kind {
        SafeRangeKind::Plain => balancing_exponent(values.iter().copied()),
        SafeRangeKind::Products => balancing_exponent_for_products(values.iter().copied()),
    };
    let exponent = match matrix.as_slice() {
        Some(slice) => exponent_of(slice),
        None => {
            let owned: Vec<T> = matrix.iter().copied().collect();
            exponent_of(&owned)
        }
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

/// [`balanced_with`] gated on the plain LAPACK safe range
/// ([`balancing_exponent`]) — for the symmetric tridiagonal QL, Jacobi, and
/// column-pivoted QR.
pub(crate) fn balanced<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Option<(Array2<T>, i32)> {
    balanced_with(matrix, SafeRangeKind::Plain)
}

/// [`balanced_with`] gated on the narrower
/// [`thresholds::product_safe_range`](super::thresholds::product_safe_range)
/// ([`balancing_exponent_for_products`]) — for the Francis double-shift QR
/// (`schur`, `eigenvalues`) and the Golub–Kahan bidiagonal QR (the SVD
/// family).
pub(crate) fn balanced_for_products<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Option<(Array2<T>, i32)> {
    balanced_with(matrix, SafeRangeKind::Products)
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
    use crate::application::linalg::thresholds::safe_range;
    use leto::LetoError;

    #[test]
    fn in_safe_range_values_are_not_scaled() {
        // f64's safe range is about [1e-146, 1e146] (rmin·rmax = 1); every
        // one of these — including magnitudes far from `[1, 4)` — needs no
        // scaling at all, the core of the safe-range redesign.
        let (rmin, rmax) = safe_range::<f64>();
        assert!(rmin > 0.0 && rmin < 1e-100, "{rmin}");
        assert!(rmax > 1e100, "{rmax}");
        for value in [
            1.0_f64,
            3.99,
            4.0,
            0.5,
            0.25,
            0.2,
            1e100,
            1e-100,
            1e300 / 1e250,
        ] {
            assert_eq!(
                balancing_exponent([value, -value / 3.0]),
                None,
                "{value} should factor unscaled"
            );
        }
        assert_eq!(balancing_exponent([0.0_f64, -0.0]), None);
        assert_eq!(balancing_exponent([1.0_f64, f64::INFINITY]), None);
        assert_eq!(balancing_exponent(Vec::<f64>::new()), None);
    }

    #[test]
    fn out_of_range_values_scale_to_an_even_exponent_in_one_to_four() {
        for value in [
            1e300_f64,
            1e-300,
            f64::from_bits(1), /* subnormal min */
        ] {
            let exponent = balancing_exponent([value, -value / 3.0])
                .unwrap_or_else(|| panic!("{value} lies outside the safe range"));
            assert_eq!(exponent % 2, 0, "{value} -> odd exponent {exponent}");
            // `2.0.powi(-exponent)` overflows for the subnormal case
            // (`exponent = -1074`); use the production scaling primitive,
            // which scales the exponent bits directly instead of forming
            // `2^1074` as an intermediate f64.
            let mut scaled = [value];
            super::scale_by_power_of_two(&mut scaled, -exponent);
            assert!((1.0..4.0).contains(&scaled[0]), "{value} -> {}", scaled[0]);
        }
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
