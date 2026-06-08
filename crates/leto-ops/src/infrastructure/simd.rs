use crate::domain::scalar::Scalar;

/// Vectorized slice addition using SIMD where available.
#[inline(always)]
pub fn simd_add<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    T::add_slice(a, b, out);
}

/// Vectorized slice subtraction using SIMD where available.
#[inline(always)]
pub fn simd_sub<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    T::sub_slice(a, b, out);
}

/// Vectorized slice multiplication using SIMD where available.
#[inline(always)]
pub fn simd_mul<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    T::mul_slice(a, b, out);
}

/// Vectorized slice division using SIMD where available.
#[inline(always)]
pub fn simd_div<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    T::div_slice(a, b, out);
}
