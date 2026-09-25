//! Exact power-of-two balancing of dense factorization inputs, scaled by the
//! *minimal* move only when a routine's own intermediates would otherwise
//! leave the representable range.
//!
//! Each dense factorization relies on some largest intermediate quantity
//! that is homogeneous of some degree `d` in the input entries (a sum of
//! squares is degree 2; a discriminant built from an already-squared term is
//! degree 4; a rotation update that only recombines entries linearly is
//! degree 1) and bounded above by a derived dimension factor `c` times
//! `‖A‖_max^d`. Unscaled, that intermediate overflows or underflows long
//! before `A` itself leaves the range of the scalar (f64 SVD failed to
//! converge below `2⁻²⁵⁸` and above `2²⁵⁴`). Each call site here states its
//! own `(d, c)`, derived from its actual formulas — see
//! [`thresholds::homogeneous_safe_range`](super::thresholds::homogeneous_safe_range)
//! for the range this implies and each caller for its derivation.
//!
//! **When `‖A‖_max` already lies in that range, the matrix is factored
//! completely unscaled** — no power-of-two multiply touches it at all. Only
//! when it falls outside does the routine factor `2⁻ᵏ·A` instead, `k` the
//! *smallest* integer (in magnitude, either sign) bringing `‖A‖_max` back
//! inside the range — never a fixed target like `[1, 4)` — and the
//! scale-carrying results are multiplied back by `2ᵏ`. The minimal move
//! keeps as many low bits of every entry as scaling can: recentring to a
//! fixed target moves entries further than the correctness argument needs,
//! costing precision in exactly the small-entry-underflow way the
//! module-level exactness caveat below describes.
//!
//! **Exactness.** Multiplying by a power of two changes only the exponent, so
//! it is exact whenever neither operand nor result leaves the normal range.
//! This is exact only while every scaled entry stays representable: an entry
//! already far below the largest one can underflow to zero under a scale
//! chosen for the largest entry (`diag(1e300, 1e-300)` in `f64`), exactly as
//! LAPACK's own scaling can lose small entries. That loss is bounded by the
//! factorizations' backward-error guarantee, never promised as exact — the
//! guarantee this module keeps is over the *whole* computation, not
//! entrywise. The minimal-move policy above is precisely what minimizes this
//! loss: it never scales further than the correctness argument requires.
//!
//! **In-range inputs are bitwise unchanged.** When `‖A‖_max` already lies in
//! the applicable range, no scaling is applied, so the factorization runs on
//! `A` itself and its result is bit-for-bit that of the unscaled algorithm.

use crate::application::linalg::thresholds;
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

/// Extra bits of margin applied to the landing target, beyond the strict
/// minimum move. Probed at `1`: it did not fix the one remaining known
/// non-convergence (`schur` on `SIMILAR` at f32's smallest subnormal
/// exponent, `2⁻¹⁴⁹` — a genuine input-degeneracy case, not a boundary-margin
/// one: `0.5·2⁻¹⁴⁹` itself underflows to exact `0` when the test constructs
/// the scaled matrix, before any balancing runs, giving Francis a
/// structurally different, degenerate input) and *did* regress a separately
/// probed, working F16 case (`tests/ops/schur.rs`'s
/// `schur_resolves_the_f16_scale_regression_matrix`, which converges landing
/// exactly at the boundary and stops converging with one bit of margin).
/// Kept at `0` (no margin): the "boundary imprecision" hypothesis this was
/// meant to test is not supported by the evidence, and the two known
/// remaining non-convergences are each their own recorded, narrower cause
/// (`LETO-F16-FRANCIS-2026-09-24`; the f32-subnormal case documented at its
/// call site) rather than a general boundary-margin problem this constant
/// would fix.
const MARGIN_BITS: i32 = 0;

/// The near-minimal-magnitude integer `k` (either sign) with `largest·2⁻ᵏ`
/// inside `(rmin, rmax)`, landing [`MARGIN_BITS`] inside rather than exactly
/// at the boundary. `largest` is assumed already outside `(rmin, rmax)` (the
/// caller checks); `rmin < rmax`, both positive and finite.
fn minimal_exponent_to_range<T: RealScalar>(largest: T, rmin: T, rmax: T) -> i32 {
    if largest >= rmax {
        // Scale down: the smallest k > 0 with `largest / 2^k <= rmax`,
        // then `MARGIN_BITS` further down.
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
        k + MARGIN_BITS
    } else {
        // largest <= rmin: scale up (k negative), the smallest |k| with
        // `largest / 2^k >= rmin`, then `MARGIN_BITS` further up.
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
        k - MARGIN_BITS
    }
}

/// The minimal-move exponent `k` bringing `values`' largest magnitude into
/// [`thresholds::homogeneous_safe_range`]`(degree, dimension_factor)`, or
/// `None` when `values` is empty, all zero, holds a non-finite entry (the
/// caller's validation reports that), or the norm is already in range (the
/// caller factors the input unscaled).
pub(crate) fn balancing_exponent<T: RealScalar>(
    values: impl IntoIterator<Item = T>,
    degree: u32,
    dimension_factor: T,
) -> Option<i32> {
    let largest = largest_magnitude(values)?;
    let (rmin, rmax) = thresholds::homogeneous_safe_range::<T>(degree, dimension_factor);
    if largest > rmin && largest < rmax {
        return None;
    }
    Some(minimal_exponent_to_range(largest, rmin, rmax))
}

/// The even exponent `k` with `max|xᵢ|·2⁻ᵏ ∈ [1, 4)`, or `None` when `largest`
/// is already in `[1, 4)`.
fn even_exponent_to_one_four<T: RealScalar>(largest: T) -> Option<i32> {
    let exponent = largest.binary_exponent()?;
    // Round toward −∞ to an even exponent: e ∈ {2j, 2j+1} ↦ 2j.
    Some(exponent - exponent.rem_euclid(2))
}

/// As [`balancing_exponent`], but recentring to `[1, 4)` instead of a
/// minimal move once the norm is judged out of range.
///
/// **Empirical exception to the minimal-move policy**, for the Francis
/// double-shift QR (`schur`, `eigenvalues`) and Golub–Kahan bidiagonal QR
/// (the SVD family) specifically: probing `schur` on the `SIMILAR` matrix
/// (`tests/ops/scale_range.rs`) across every exponent found the *minimal*-move
/// landing — near the computed `(rmin, rmax)` boundary — non-convergent
/// across a wide band (Bf16: essentially every exponent from `2⁻¹³³` to
/// `2⁻³³`; f32: `2⁻¹⁴⁹`), while recentring the same out-of-range inputs to
/// `[1, 4)` converges throughout. The `(degree, dimension_factor)` derivation
/// is unaffected — it still decides correctly whether an input needs
/// scaling at all, so in-range inputs remain bitwise unscaled — but a value
/// that does need scaling is recentred rather than moved minimally, because
/// these two algorithms' shift/discriminant formulas are evidently more
/// numerically fragile near the derived boundary than the degree/dimension
/// analysis alone predicts (`LETO-FRANCIS-QUARTIC-SCALE-2026-09-24` tracks
/// closing that gap with scale-invariant formulas, which would let these
/// routines drop back to the minimal move).
pub(crate) fn balancing_exponent_recentered<T: RealScalar>(
    values: impl IntoIterator<Item = T>,
    degree: u32,
    dimension_factor: T,
) -> Option<i32> {
    let largest = largest_magnitude(values)?;
    let (rmin, rmax) = thresholds::homogeneous_safe_range::<T>(degree, dimension_factor);
    if largest > rmin && largest < rmax {
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

/// Which exponent policy [`balanced_with`] uses once a norm is judged out of
/// range: [`balancing_exponent`]'s minimal move, or
/// [`balancing_exponent_recentered`]'s recentre to `[1, 4)`.
#[derive(Clone, Copy)]
enum ExponentPolicy {
    Minimal,
    Recentered,
}

fn balanced_with<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    degree: u32,
    dimension_factor: T,
    policy: ExponentPolicy,
) -> Option<(Array2<T>, i32)> {
    let exponent_of = |values: &[T]| match policy {
        ExponentPolicy::Minimal => {
            balancing_exponent(values.iter().copied(), degree, dimension_factor)
        }
        ExponentPolicy::Recentered => {
            balancing_exponent_recentered(values.iter().copied(), degree, dimension_factor)
        }
    };
    let exponent = match matrix.as_slice() {
        Some(slice) => exponent_of(slice),
        None => {
            let owned: Vec<T> = matrix.iter().copied().collect();
            exponent_of(&owned)
        }
    }?;
    let mut values = match matrix.as_slice() {
        Some(slice) => slice.to_vec(),
        None => matrix.iter().copied().collect(),
    };
    scale_by_power_of_two(&mut values, -exponent);
    let array = Array2::from_shape_vec(matrix.shape(), values)
        .expect("invariant: a copy of a view keeps its shape and length");
    Some((array, exponent))
}

/// `matrix` multiplied by `2⁻ᵏ` as an owned contiguous copy, with `k`
/// ([`balancing_exponent`]'s minimal move for `(degree, dimension_factor)`);
/// `None` when no scaling applies (a zero or empty matrix, a non-finite
/// entry, or `‖matrix‖_max` already in range), in which case the caller
/// factors `matrix` itself, bit-for-bit unscaled.
pub(crate) fn balanced<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    degree: u32,
    dimension_factor: T,
) -> Option<(Array2<T>, i32)> {
    balanced_with(matrix, degree, dimension_factor, ExponentPolicy::Minimal)
}

/// As [`balanced`], but using [`balancing_exponent_recentered`] once out of
/// range — see its documentation for why the Francis/bidiagonal-QR family
/// uses this instead of the minimal move.
pub(crate) fn balanced_recentered<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    degree: u32,
    dimension_factor: T,
) -> Option<(Array2<T>, i32)> {
    balanced_with(matrix, degree, dimension_factor, ExponentPolicy::Recentered)
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
    use super::{balancing_exponent, balancing_exponent_recentered, restore};
    use crate::application::linalg::thresholds::homogeneous_safe_range;
    use leto::LetoError;

    #[test]
    fn in_range_values_are_not_scaled() {
        // Degree 2, dimension_factor 1: f64's range is about
        // [1e-146, 1e146] (rmin*rmax = 1); every one of these needs no
        // scaling at all.
        let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 1.0);
        assert!(rmin > 0.0 && rmin < 1e-100, "{rmin}");
        assert!(rmax > 1e100, "{rmax}");
        for value in [1.0_f64, 3.99, 4.0, 0.5, 0.25, 0.2, 1e100, 1e-100] {
            assert_eq!(
                balancing_exponent([value, -value / 3.0], 2, 1.0),
                None,
                "{value} should factor unscaled"
            );
        }
        assert_eq!(balancing_exponent([0.0_f64, -0.0], 2, 1.0), None);
        assert_eq!(balancing_exponent([1.0_f64, f64::INFINITY], 2, 1.0), None);
        assert_eq!(balancing_exponent(Vec::<f64>::new(), 2, 1.0), None);
    }

    #[test]
    fn recentered_variant_also_leaves_in_range_values_unscaled() {
        // Kills the "always scale Schur/SVD" mutant: `balancing_exponent_recentered`
        // (used by `balanced_recentered`, the Francis/bidiagonal-QR path)
        // must skip scaling exactly like the minimal-move variant when the
        // norm is already in range — the exponent policy differs only for
        // out-of-range norms.
        for value in [1.0_f64, 3.99, 0.5, 100.0] {
            assert_eq!(
                balancing_exponent_recentered([value, -value / 3.0], 2, 1.0),
                None,
                "{value} should factor unscaled"
            );
        }
        assert!(balancing_exponent_recentered([1e300_f64], 2, 1.0).is_some());
    }

    #[test]
    fn out_of_range_values_scale_by_the_minimal_move() {
        // 1e300 exceeds f64's degree-2 rmax (~1e146): the near-minimal move
        // brings it just inside (one `MARGIN_BITS` shy of the boundary), not
        // to a fixed [1,4) target.
        let (_, rmax) = homogeneous_safe_range::<f64>(2, 1.0);
        let k = balancing_exponent([1e300_f64], 2, 1.0).expect("1e300 is out of range");
        let mut scaled = [1e300_f64];
        super::scale_by_power_of_two(&mut scaled, -k);
        assert!(scaled[0] <= rmax, "{} > {rmax}", scaled[0]);
        // Near-minimality: giving up the margin bit (k - MARGIN_BITS, the
        // exponent the boundary search alone would pick) still lands at or
        // under rmax — it is the *next* step down, past the boundary
        // search's own minimum, that overshoots.
        let mut without_margin = [1e300_f64];
        super::scale_by_power_of_two(&mut without_margin, -(k - super::MARGIN_BITS));
        assert!(
            without_margin[0] <= rmax,
            "{k} added more than the {}-bit margin",
            super::MARGIN_BITS
        );
        let mut one_further_less = [1e300_f64];
        super::scale_by_power_of_two(&mut one_further_less, -(k - super::MARGIN_BITS - 1));
        assert!(one_further_less[0] > rmax, "{k} was not near-minimal");

        // Subnormal minimum in f64 scales up minimally too.
        let (rmin, _) = homogeneous_safe_range::<f64>(2, 1.0);
        let k = balancing_exponent([f64::from_bits(1)], 2, 1.0).expect("subnormal is out of range");
        let mut scaled = [f64::from_bits(1)];
        super::scale_by_power_of_two(&mut scaled, -k);
        assert!(scaled[0] >= rmin, "{} < {rmin}", scaled[0]);
        let mut without_margin = [f64::from_bits(1)];
        super::scale_by_power_of_two(&mut without_margin, -(k + super::MARGIN_BITS));
        assert!(
            without_margin[0] >= rmin,
            "{k} added more than the {}-bit margin",
            super::MARGIN_BITS
        );
        let mut one_further_less = [f64::from_bits(1)];
        super::scale_by_power_of_two(&mut one_further_less, -(k + super::MARGIN_BITS + 1));
        assert!(one_further_less[0] < rmin, "{k} was not near-minimal");
    }

    #[test]
    fn dimension_factor_narrows_the_range() {
        // A larger dimension_factor (more entries contributing to the
        // relied-upon intermediate) shrinks the safe interval symmetrically
        // in the sense that a value safe at factor 1 may need scaling at a
        // much larger factor.
        assert_eq!(balancing_exponent([100.0_f64], 2, 1.0), None);
        assert!(balancing_exponent([100.0_f64], 4, 1e290).is_some());
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
