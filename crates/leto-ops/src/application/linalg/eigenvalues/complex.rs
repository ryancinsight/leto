//! Minimal complex arithmetic over [`RealScalar`].
//!
//! `num_complex::Complex<T>`'s operators require `T: num_traits::Num`, which the
//! sealed [`RealScalar`] trait does not provide. This compute-local `Cplx<T>`
//! implements exactly the operations the QR eigenvalue iteration needs, in the
//! native precision of `T`. It is converted to the public
//! `num_complex::Complex<T>` only at the API boundary (a field copy, no ops).

use crate::domain::real::RealScalar;

/// A complex number `re + im·i` over a real scalar `T`.
#[derive(Clone, Copy)]
pub(super) struct Cplx<T> {
    pub(super) re: T,
    pub(super) im: T,
}

impl<T: RealScalar> Cplx<T> {
    #[inline]
    pub(super) fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Embed a real number.
    #[inline]
    pub(super) fn real(re: T) -> Self {
        Self { re, im: T::ZERO }
    }

    #[inline]
    pub(super) fn zero() -> Self {
        Self::real(T::ZERO)
    }

    #[inline]
    pub(super) fn add(self, o: Self) -> Self {
        Self::new(self.re.add(o.re), self.im.add(o.im))
    }

    #[inline]
    pub(super) fn sub(self, o: Self) -> Self {
        Self::new(self.re.sub(o.re), self.im.sub(o.im))
    }

    /// `(a+bi)(c+di) = (ac − bd) + (ad + bc)i`.
    #[inline]
    pub(super) fn mul(self, o: Self) -> Self {
        Self::new(
            self.re.mul(o.re).sub(self.im.mul(o.im)),
            self.re.mul(o.im).add(self.im.mul(o.re)),
        )
    }

    /// `z / w = z·conj(w) / |w|²`.
    #[inline]
    pub(super) fn div(self, o: Self) -> Self {
        let denom = o.abs_sq();
        let num = self.mul(o.conj());
        Self::new(num.re.div(denom), num.im.div(denom))
    }

    /// Multiply by a real scalar.
    #[inline]
    pub(super) fn scale(self, t: T) -> Self {
        Self::new(self.re.mul(t), self.im.mul(t))
    }

    #[inline]
    pub(super) fn conj(self) -> Self {
        Self::new(self.re, T::ZERO.sub(self.im))
    }

    /// `|z|² = re² + im²`.
    #[inline]
    pub(super) fn abs_sq(self) -> T {
        self.re.mul(self.re).add(self.im.mul(self.im))
    }

    /// `|z| = √(re² + im²)`.
    #[inline]
    pub(super) fn abs(self) -> T {
        self.abs_sq().sqrt()
    }

    /// Principal complex square root.
    ///
    /// For `z = re + im·i` with `r = |z|`,
    /// `√z = √((r+re)/2) + sign(im)·√((r−re)/2)·i`. *Check:* squaring gives real
    /// part `(r+re)/2 − (r−re)/2 = re` and imaginary part
    /// `2·√((r+re)/2)·√((r−re)/2)·sign(im) = √(r²−re²)·sign(im) = |im|·sign(im) = im`. ∎
    #[inline]
    pub(super) fn sqrt(self) -> Self {
        if self.abs_sq() <= T::ZERO {
            return Self::zero();
        }
        let r = self.abs();
        let two = T::ONE.add(T::ONE);
        let re_out = r.add(self.re).div(two).sqrt();
        let im_mag = r.sub(self.re).div(two).sqrt();
        let im_out = if self.im < T::ZERO {
            T::ZERO.sub(im_mag)
        } else {
            im_mag
        };
        Self::new(re_out, im_out)
    }
}
