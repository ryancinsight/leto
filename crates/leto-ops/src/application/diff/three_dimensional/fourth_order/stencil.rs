//! Stencil coefficients and their point/lane application, shared by the
//! [`dense`](super::dense) and [`strided`](super::strided) traversals.
//!
//! A coordinate `c` on an axis of `n` points takes one stencil:
//!
//! | Position | Stencil | Order |
//! |---|---|---|
//! | `2 ≤ c ≤ n − 3` | `(−f[c+2] + 8f[c+1] − 8f[c−1] + f[c−2]) / 12Δ` | 4 |
//! | `c ∈ {1, n − 2}` | `(f[c+1] − f[c−1]) / 2Δ` | 2 |
//! | `c = 0` | `(f[1] − f[0]) / Δ` | 1 |
//! | `c = n − 1` | `(f[n−1] − f[n−2]) / Δ` | 1 |
//! | `n = 1` | `0` | — |
//!
//! The first matching row wins, so an axis shorter than five points takes only
//! the lower-order closures (a two-point axis is one-sided at both ends), and a
//! singleton axis has no variation along it. Each stencil is written as one
//! arithmetic chain shared by both traversals, so they agree bit for bit.

use eunomia::{FloatElement, NumericElement, RealField};

use super::super::f;

/// Bytes one output element moves: itself and the four neighbours the
/// interior stencil reads.
pub(super) const ELEMENTS_PER_UNIT: usize = 5;

/// Scale factors of the three stencil orders.
#[derive(Clone, Copy)]
pub(super) struct Scales<T> {
    inv_12h: T,
    inv_2h: T,
    inv_h: T,
    eight: T,
}

impl<T: RealField + FloatElement + Copy> Scales<T> {
    pub(super) fn new(h: T) -> Self {
        Self {
            inv_12h: <T as NumericElement>::ONE / (f::<T>(12.0) * h),
            inv_2h: <T as NumericElement>::ONE / (f::<T>(2.0) * h),
            inv_h: <T as NumericElement>::ONE / h,
            eight: f::<T>(8.0),
        }
    }

    #[inline]
    pub(super) fn fourth(self, m2: T, m1: T, p1: T, p2: T) -> T {
        ((-self.eight * m1) + (self.eight * p1) + (-p2) + m2) * self.inv_12h
    }

    #[inline]
    pub(super) fn second(self, m1: T, p1: T) -> T {
        (p1 - m1) * self.inv_2h
    }

    #[inline]
    pub(super) fn first(self, lower: T, upper: T) -> T {
        (upper - lower) * self.inv_h
    }
}

/// The stencil coordinate `c` takes on an axis of `n` points.
#[derive(Clone, Copy)]
pub(super) enum Stencil {
    Flat,
    Forward,
    Backward,
    Second,
    Fourth,
}

impl Stencil {
    #[inline]
    pub(super) fn at(c: usize, n: usize) -> Self {
        if n == 1 {
            Self::Flat
        } else if c == 0 {
            Self::Forward
        } else if c == n - 1 {
            Self::Backward
        } else if c < 2 || c >= n - 2 {
            Self::Second
        } else {
            Self::Fourth
        }
    }

    /// The derivative at a point whose neighbour at signed offset `o` along
    /// the axis is `at(o)`; only the offsets the stencil reads are asked for.
    #[inline]
    pub(super) fn apply<T: RealField + FloatElement + Copy>(
        self,
        scales: Scales<T>,
        at: impl Fn(isize) -> T,
    ) -> T {
        match self {
            Self::Flat => <T as NumericElement>::ZERO,
            Self::Forward => scales.first(at(0), at(1)),
            Self::Backward => scales.first(at(-1), at(0)),
            Self::Second => scales.second(at(-1), at(1)),
            Self::Fourth => scales.fourth(at(-2), at(-1), at(1), at(2)),
        }
    }

    /// The derivative along a whole lane, added to what `out` already holds.
    #[inline]
    pub(super) fn add_lane<'a, T: RealField + FloatElement + Copy + 'a>(
        self,
        scales: Scales<T>,
        out: &mut [T],
        lane: impl Fn(isize) -> &'a [T],
    ) {
        match self {
            Self::Flat => (),
            Self::Forward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(0).iter().zip(lane(1))) {
                    *value += scales.first(lower, upper);
                }
            }
            Self::Backward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(-1).iter().zip(lane(0))) {
                    *value += scales.first(lower, upper);
                }
            }
            Self::Second => {
                for (value, (&m1, &p1)) in out.iter_mut().zip(lane(-1).iter().zip(lane(1))) {
                    *value += scales.second(m1, p1);
                }
            }
            Self::Fourth => {
                let neighbours = lane(-2).iter().zip(lane(-1)).zip(lane(1)).zip(lane(2));
                for (value, (((&m2, &m1), &p1), &p2)) in out.iter_mut().zip(neighbours) {
                    *value += scales.fourth(m2, m1, p1, p2);
                }
            }
        }
    }

    /// The derivative along a whole lane whose neighbour lane at signed
    /// offset `o` is `lane(o)`, each as long as `out`.
    #[inline]
    pub(super) fn apply_lane<'a, T: RealField + FloatElement + Copy + 'a>(
        self,
        scales: Scales<T>,
        out: &mut [T],
        lane: impl Fn(isize) -> &'a [T],
    ) {
        match self {
            Self::Flat => out.fill(<T as NumericElement>::ZERO),
            Self::Forward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(0).iter().zip(lane(1))) {
                    *value = scales.first(lower, upper);
                }
            }
            Self::Backward => {
                for (value, (&lower, &upper)) in out.iter_mut().zip(lane(-1).iter().zip(lane(0))) {
                    *value = scales.first(lower, upper);
                }
            }
            Self::Second => {
                for (value, (&m1, &p1)) in out.iter_mut().zip(lane(-1).iter().zip(lane(1))) {
                    *value = scales.second(m1, p1);
                }
            }
            Self::Fourth => {
                let neighbours = lane(-2).iter().zip(lane(-1)).zip(lane(1)).zip(lane(2));
                for (value, (((&m2, &m1), &p1), &p2)) in out.iter_mut().zip(neighbours) {
                    *value = scales.fourth(m2, m1, p1, p2);
                }
            }
        }
    }
}
