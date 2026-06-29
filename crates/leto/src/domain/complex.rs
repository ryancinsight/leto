//! Atlas-native complex number — the leto vocabulary replacement for
//! `num_complex::Complex`, removing the third-party dependency from leto's
//! production graph.
//!
//! Layout-compatible (`#[repr(C)]` `{ re, im }`) and `bytemuck::Pod`, so values
//! round-trip through GPU device buffers (hephaestus) and FFI boundaries
//! identically to `num_complex::Complex`.

use bytemuck::{Pod, Zeroable};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number `re + im·i`.
///
/// A minimal vocabulary type: a `#[repr(C)]` pair of real/imaginary components
/// with the arithmetic and trait surface leto's linear-algebra ops rely on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T> Complex<T> {
    /// Construct from real and imaginary parts.
    #[inline(always)]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

// SAFETY: `Complex<T>` is `#[repr(C)]` with two `T` fields, so it is zeroable
// and plain-old-data exactly when `T` is.
unsafe impl<T: Zeroable> Zeroable for Complex<T> {}
unsafe impl<T: Pod> Pod for Complex<T> {}

impl<T: Add<Output = T>> Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone> Mul for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re.clone() * other.re.clone() - self.im.clone() * other.im.clone(),
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Clone> Div
    for Complex<T>
{
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        let denom = other.re.clone() * other.re.clone() + other.im.clone() * other.im.clone();
        Self {
            re: (self.re.clone() * other.re.clone() + self.im.clone() * other.im.clone())
                / denom.clone(),
            im: (self.im * other.re.clone() - self.re * other.im) / denom,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: core::fmt::Display> core::fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}+{}i", self.re, self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::Complex;

    #[test]
    fn new_and_fields() {
        let c = Complex::new(2.0_f64, -3.0);
        assert_eq!(c.re, 2.0);
        assert_eq!(c.im, -3.0);
    }

    #[test]
    fn arithmetic_matches_definition() {
        let a = Complex::new(1.0_f64, 2.0);
        let b = Complex::new(3.0_f64, -1.0);
        assert_eq!(a + b, Complex::new(4.0, 1.0));
        assert_eq!(a - b, Complex::new(-2.0, 3.0));
        // (1+2i)(3-i) = 3 - i + 6i - 2i² = 3 + 5i + 2 = 5 + 5i
        assert_eq!(a * b, Complex::new(5.0, 5.0));
        // (1+2i)/(3-i) = (1+2i)(3+i)/((3)²+(1)²) = (3+i+6i-2)/10 = (1+7i)/10
        let q = a / b;
        assert!((q.re - 0.1).abs() < 1e-12 && (q.im - 0.7).abs() < 1e-12);
        assert_eq!(-a, Complex::new(-1.0, -2.0));
    }

    #[test]
    fn is_pod_layout() {
        // repr(C) { re, im } — same 2*size_of::<T> layout used for GPU buffers.
        assert_eq!(
            core::mem::size_of::<Complex<f32>>(),
            2 * core::mem::size_of::<f32>()
        );
    }
}
