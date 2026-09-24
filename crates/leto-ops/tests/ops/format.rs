//! The binary formats the dense solvers support, with their exponent ranges
//! and machine epsilon, shared by the scale-range tests.

use eunomia::{Bf16, F16};
use leto_ops::RealScalar;

/// A supported scalar with the exponent range of its normal numbers.
pub trait Format: RealScalar {
    /// Binary exponent of the smallest positive normal value.
    const MIN_EXPONENT: i32;
    /// Binary exponent of the largest finite value.
    const MAX_EXPONENT: i32;
    /// Significand precision `p` in bits (the smallest subnormal is
    /// `2^(MIN_EXPONENT − p + 1)`).
    const PRECISION: i32;
}

impl Format for f64 {
    const MIN_EXPONENT: i32 = -1022;
    const MAX_EXPONENT: i32 = 1023;
    const PRECISION: i32 = 53;
}
impl Format for f32 {
    const MIN_EXPONENT: i32 = -126;
    const MAX_EXPONENT: i32 = 127;
    const PRECISION: i32 = 24;
}
impl Format for F16 {
    const MIN_EXPONENT: i32 = -14;
    const MAX_EXPONENT: i32 = 15;
    const PRECISION: i32 = 11;
}
impl Format for Bf16 {
    const MIN_EXPONENT: i32 = -126;
    const MAX_EXPONENT: i32 = 127;
    const PRECISION: i32 = 8;
}

/// Machine epsilon of `T`, found through `T`'s own addition.
pub fn epsilon<T: RealScalar>() -> f64 {
    let half = T::from_f64(0.5);
    let mut e = T::ONE;
    while T::ONE.add(e.mul(half)) > T::ONE {
        e = e.mul(half);
    }
    e.to_f64()
}
