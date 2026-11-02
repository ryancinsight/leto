//! Scaled sum-of-squares norm without rounding each physical partial norm.

use crate::domain::real::RealScalar;

/// Accumulate relative squares in T, applying the physical scale only once.
/// Repeated `hypot(partial, x)` can stagnate when both operands are subnormal,
/// even when the complete norm is representable (64 minimum values have
/// norm eight times the minimum). This is the scaled sum-of-squares pattern.
pub(crate) fn scaled_norm<T: RealScalar>(values: impl IntoIterator<Item = T>) -> T {
    let mut scale = T::ZERO;
    let mut squares = T::ONE;
    for value in values {
        let value = value.abs();
        if value == T::ZERO {
            continue;
        }
        if value > scale {
            let ratio = scale.div(value);
            squares = T::ONE.add(squares.mul(ratio).mul(ratio));
            scale = value;
        } else {
            let ratio = value.div(scale);
            squares = squares.add(ratio.mul(ratio));
        }
    }
    scale.mul(squares.sqrt())
}
