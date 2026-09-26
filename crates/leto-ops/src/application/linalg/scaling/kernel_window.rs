//! The kernel tier: a kernel's representable window and the scale-safe
//! Euclidean norm of a pair.

use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;

/// A kernel's representable window for its local magnitude `m` (the
/// largest operand it reads): the kernel tier's gate, computed once per
/// routine call from [`thresholds::kernel_window`] and passed down to the
/// hot kernels.
#[derive(Clone, Copy)]
pub(crate) struct KernelWindow<T> {
    low: T,
    high: T,
}

impl<T: RealScalar> KernelWindow<T> {
    /// The window keeping the kernel's smallest relied-upon product (degree
    /// `lower_degree`) normal and its largest (degree `upper_degree`, bounded
    /// by `2^factor_log2·m^dᵤ`) finite.
    pub(crate) fn new(lower_degree: u32, upper_degree: u32, factor_log2: i32) -> Self {
        let (low, high) = thresholds::kernel_window::<T>(lower_degree, upper_degree, factor_log2);
        Self { low, high }
    }

    /// `0` when the largest magnitude among `operands` lies in the window (or
    /// is zero or non-finite — nothing to rescale), otherwise its binary
    /// exponent `k`, so the operands divided by `2ᵏ` have their largest
    /// magnitude in `[1, 2)`.
    pub(crate) fn exponent(self, operands: &[T]) -> i32 {
        let largest = operands
            .iter()
            .fold(T::ZERO, |acc, v| if v.abs() > acc { v.abs() } else { acc });
        if largest == T::ZERO || (largest >= self.low && largest <= self.high) {
            return 0;
        }
        largest.binary_exponent().unwrap_or(0)
    }
}

/// `√(x² + y²)` without the overflow or underflow of squaring the larger
/// operand — LAPACK `dlapy2`: the larger magnitude times
/// `√(1 + (smaller/larger)²)`.
#[inline]
pub(crate) fn hypot<T: RealScalar>(x: T, y: T) -> T {
    let (x, y) = (x.abs(), y.abs());
    let (large, small) = if x >= y { (x, y) } else { (y, x) };
    if large == T::ZERO {
        return T::ZERO;
    }
    let ratio = small.div(large);
    large.mul(T::ONE.add(ratio.mul(ratio)).sqrt())
}
