//! Arithmetic operators for [`Vector`].
//!
//! Element-wise operators build the result with [`core::array::from_fn`] — one
//! const-generic expression each, no hand-rolled loops — and monomorphize to
//! the same code as a fixed-width implementation.

use super::Vector;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T: Add<Output = T> + Copy, const N: usize> Add for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] + rhs.data[i]),
        }
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] - rhs.data[i]),
        }
    }
}

impl<T: Neg<Output = T> + Copy, const N: usize> Neg for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            data: core::array::from_fn(|i| -self.data[i]),
        }
    }
}

impl<T: Mul<Output = T> + Copy, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Self;
    /// Scalar multiplication.
    #[inline(always)]
    fn mul(self, scalar: T) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] * scalar),
        }
    }
}

impl<T: Div<Output = T> + Copy, const N: usize> Div<T> for Vector<T, N> {
    type Output = Self;
    /// Scalar division.
    #[inline(always)]
    fn div(self, scalar: T) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] / scalar),
        }
    }
}

impl<T: Add<Output = T> + Copy, const N: usize> AddAssign for Vector<T, N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> SubAssign for Vector<T, N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Mul<Output = T> + Copy, const N: usize> MulAssign<T> for Vector<T, N> {
    #[inline(always)]
    fn mul_assign(&mut self, scalar: T) {
        *self = *self * scalar;
    }
}

impl<T: Div<Output = T> + Copy, const N: usize> DivAssign<T> for Vector<T, N> {
    #[inline(always)]
    fn div_assign(&mut self, scalar: T) {
        *self = *self / scalar;
    }
}
