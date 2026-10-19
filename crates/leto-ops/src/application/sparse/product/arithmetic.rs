//! Arithmetic policy is independent of CSR addressing and row scheduling.

use crate::domain::{
    scalar::Scalar,
    strategy::{ScalarStrategy, SimdStrategy},
};
use eunomia::NumericElement;

mod sealed {
    pub trait Arithmetic<T> {
        fn accumulate(value: T, rhs: &[T], output: &mut [T]);
    }
}

/// Sealed row-update policy for a CSR product.
///
/// [`ScalarStrategy`] uses native `T` multiplication followed by `AddAssign`,
/// without contraction, entry reordering, zero dropping or wider arithmetic.
/// [`SimdStrategy`] preserves [`Scalar::axpy_slice`], including its ISA-dependent
/// fused multiply/add behavior. The caller selects the numerical contract.
pub trait CsrArithmetic<T: NumericElement>: sealed::Arithmetic<T> + Send + Sync + 'static {}

impl<T: NumericElement> sealed::Arithmetic<T> for ScalarStrategy {
    #[inline]
    fn accumulate(value: T, rhs: &[T], output: &mut [T]) {
        for (slot, &element) in output.iter_mut().zip(rhs) {
            *slot += value * element;
        }
    }
}
impl<T: NumericElement> CsrArithmetic<T> for ScalarStrategy {}

impl<T: Scalar> sealed::Arithmetic<T> for SimdStrategy {
    #[inline]
    fn accumulate(value: T, rhs: &[T], output: &mut [T]) {
        T::axpy_slice(value, rhs, output);
    }
}
impl<T: Scalar> CsrArithmetic<T> for SimdStrategy {}

#[inline]
pub(super) fn accumulate<T: NumericElement, A: CsrArithmetic<T>>(
    value: T,
    rhs: &[T],
    output: &mut [T],
) {
    A::accumulate(value, rhs, output);
}
