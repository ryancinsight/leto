//! Native exponent-scaled quaternion inverse recovery.

use super::Quaternion;
use eunomia::RealField;

// Exponent scaling avoids forming an unrepresentable squared norm. Keep this
// work outside the ordinary inverse and preserve legacy zero/nonfinite results.
#[cold]
#[inline(never)]
pub(super) fn recover_inverse<T: RealField>(
    q: Quaternion<T>,
    original_norm: T,
) -> Option<Quaternion<T>> {
    let legacy = || {
        if original_norm == T::ZERO {
            None
        } else {
            Some(q.conjugate() / original_norm)
        }
    };
    let components = [q.w, q.x, q.y, q.z];
    if components.iter().any(|component| !component.is_finite()) {
        return legacy();
    }
    let maximum = components.iter().fold(T::ZERO, |maximum, component| {
        maximum.max_scalar(component.abs())
    });
    let exponent = maximum.binary_exponent()?;
    let scaled = components.map(|component| component.scale_binary(-exponent));

    // The largest scaled magnitude is in [1,2), so the norm is in [1,16).
    // Preserve the rounding residuals before the final reciprocal scaling;
    // small mixed components must not disappear through an early division.
    let mut norm = T::ZERO;
    let mut tail = T::ZERO;
    for component in scaled {
        let square = component * component;
        let product_error = component.scalar_fmadd(component, -square);
        let sum = norm + square;
        let virtual_square = sum - norm;
        let sum_error = (norm - (sum - virtual_square)) + (square - virtual_square);
        tail += product_error + sum_error;
        norm = sum;
    }
    let inverse = scaled.map(|component| {
        if component == T::ZERO {
            return component;
        }
        let quotient = component / norm;
        let residual = (-quotient).scalar_fmadd(norm, component) - quotient * tail;
        (quotient + residual / norm).scale_binary(-exponent)
    });
    if inverse.iter().all(|component| component.is_finite()) {
        Some(Quaternion::new(
            inverse[0],
            -inverse[1],
            -inverse[2],
            -inverse[3],
        ))
    } else {
        // A nonrepresentable inverse keeps the former result/range policy.
        legacy()
    }
}
