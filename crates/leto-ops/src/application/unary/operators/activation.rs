//! The activation family with its gradient companions (hephaestus parity).

use super::contract::UnaryOp;
use crate::domain::RealScalar;
use eunomia::NumericElement;

// ── Activation markers ───────────────────────────────────────────────────
// Elementwise counterparts of hephaestus-core `activation_value`: same
// formulas, same branch structure, same crossover constants, so CPU and GPU
// lanes evaluate the same program. The measurement derivations for the
// crossovers live in hephaestus-core and are not repeated here.

/// Numerically stable logistic sigmoid (ADR 0061 Decision 6).
#[must_use]
fn stable_sigmoid<T: RealScalar>(x: T) -> T {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    if x >= zero {
        one / (one + (-x).exp())
    } else {
        let e = x.exp();
        e / (one + e)
    }
}
/// Numerically stable softplus `max(x, 0) + ln_1p(exp(-|x|))`
/// (ADR 0061 Decision 6).
#[must_use]
fn softplus_value<T: RealScalar>(x: T) -> T {
    x.max(<T as NumericElement>::ZERO) + (-<T as NumericElement>::abs(x)).exp().ln_1p()
}
/// Whether `x` is `+∞` or `-∞` (eunomia exposes no direct predicate).
///
/// Callers test an infinity's sign with `x > ZERO` rather than
/// `is_sign_positive`: exact on infinities, and it keeps these impls under
/// `RealScalar` (f32/f64/F16/BF16) instead of narrowing to `RealField`
/// (f32/f64 only).
#[must_use]
fn is_infinite<T: RealScalar>(x: T) -> bool {
    !x.is_finite() && !x.is_nan()
}
/// `clamp` over `RealScalar`, which lacks `RealField::clamp`: identical on
/// finite and infinite values. A NaN value falls through to itself, and both
/// call sites combine the result with a NaN-carrying factor, so NaN still
/// propagates exactly as under eunomia's NaN-ignoring form.
fn clamp_scalar<T: RealScalar>(v: T, lo: T, hi: T) -> T {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
/// `√(2/π)·(x + 0.044715·x³)`, shared by `GeluTanhOp` and `GeluTanhGradOp`.
fn gelu_tanh_arg<T: RealScalar>(x: T) -> T {
    let c0 = T::from_f64(0.797_884_560_802_865_4);
    let c1 = T::from_f64(0.044715);
    c0 * (x + c1 * x * x * x)
}
/// Largest `x` for which `SiluGradOp` forms `1 − sigmoid(x)` directly.
const SILU_GRAD_CROSSOVER: f64 = 2.63;
/// Largest `w = 2z` for which `GeluTanhGradOp` forms `1 − sigmoid(w)` directly.
const GELU_TANH_GRAD_CROSSOVER: f64 = 2.47;
/// Largest `sp` for which `MishGradOp` forms `1 − tanh(sp)²` directly.
const MISH_GRAD_CROSSOVER: f64 = 1.55;
/// Largest `|y|` for which `TanhGradOp` forms `1 − y²` directly.
const TANH_GRAD_CROSSOVER: f64 = 0.75;
/// ReLU operation marker: `max(x, 0)`, propagating NaN (eunomia's `max`
/// ignores a NaN operand, so the explicit check comes first).
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluOp;
impl<T: RealScalar> UnaryOp<T> for ReluOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x.is_nan() {
            x
        } else {
            x.max(<T as NumericElement>::ZERO)
        }
    }
}
/// ReLU gradient marker, taking the input: `1` for `x > 0`, else `0`,
/// propagating NaN.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluGradOp;
impl<T: RealScalar> UnaryOp<T> for ReluGradOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x.is_nan() {
            x
        } else if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}
/// ELU operation marker (`α = 1`): `x` for `x ≥ 0`, `expm1(x)` below.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluOp;
impl<T: RealScalar> UnaryOp<T> for EluOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            x.exp_m1()
        }
    }
}
/// ELU gradient marker, taking the input: `1` for `x ≥ 0`, `exp(x)` below.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluGradOp;
impl<T: RealScalar> UnaryOp<T> for EluGradOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            x.exp()
        }
    }
}
/// Logistic sigmoid operation marker, via the stable branched form.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidOp;
impl<T: RealScalar> UnaryOp<T> for SigmoidOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        stable_sigmoid(x)
    }
}
/// Sigmoid gradient marker, taking the forward output `y = sigmoid(x)`:
/// `y * (1 - y)` (ADR 0061 Decision 7).
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidGradOp;
impl<T: RealScalar> UnaryOp<T> for SigmoidGradOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, y: T) -> T {
        y * (<T as NumericElement>::ONE - y)
    }
}
/// Tanh gradient marker, taking the forward output `y = tanh(x)`: `1 - y²`
/// below the crossover, `(1 - y)(1 + y)` above it, where the direct form
/// loses the significand.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhGradOp;
impl<T: RealScalar> UnaryOp<T> for TanhGradOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, y: T) -> T {
        let one = <T as NumericElement>::ONE;
        if <T as NumericElement>::abs(y) <= T::from_f64(TANH_GRAD_CROSSOVER) {
            one - y * y
        } else {
            (one - y) * (one + y)
        }
    }
}
/// Softplus operation marker, via the stable `max + ln_1p` form.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusOp;
impl<T: RealScalar> UnaryOp<T> for SoftplusOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        softplus_value(x)
    }
}
/// Softplus gradient marker, taking the input: the logistic sigmoid.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusGradOp;
impl<T: RealScalar> UnaryOp<T> for SoftplusGradOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        stable_sigmoid(x)
    }
}
/// Softsign operation marker: `x / (1 + |x|)`, with the analytic `±1`
/// limit at infinities (the quotient would form `∞ / ∞`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftsignOp;
impl<T: RealScalar> UnaryOp<T> for SoftsignOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                <T as NumericElement>::ONE
            } else {
                -<T as NumericElement>::ONE
            };
        }
        x / (<T as NumericElement>::ONE + <T as NumericElement>::abs(x))
    }
}
/// Softsign gradient marker, taking the input: `1 / (1 + |x|)²`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftsignGradOp;
impl<T: RealScalar> UnaryOp<T> for SoftsignGradOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        let denom = <T as NumericElement>::ONE + <T as NumericElement>::abs(x);
        <T as NumericElement>::ONE / (denom * denom)
    }
}
/// Hard sigmoid operation marker: `(x / 6 + 0.5)` clamped to `[0, 1]`,
/// propagating NaN (`clamp` ignores a NaN operand, so the check comes first).
#[derive(Clone, Copy, Debug, Default)]
pub struct HardsigmoidOp;
impl<T: RealScalar> UnaryOp<T> for HardsigmoidOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;
        let six = T::from_f64(6.0);
        let half = T::from_f64(0.5);
        clamp_scalar(x / six + half, zero, one)
    }
}
/// Hard sigmoid gradient marker, taking the input: `1/6` on `(-3, 3)`,
/// else `0`, propagating NaN.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardsigmoidGradOp;
impl<T: RealScalar> UnaryOp<T> for HardsigmoidGradOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let three = T::from_f64(3.0);
        if x > -three && x < three {
            <T as NumericElement>::ONE / T::from_f64(6.0)
        } else {
            <T as NumericElement>::ZERO
        }
    }
}
/// Hard swish operation marker: `x * clamp(x + 3, 0, 6) / 6`, dividing
/// before multiplying so no intermediate overflows, and short-circuiting
/// `x ≤ -3` to `-0` (the product would form `-∞ · 0` at `-∞`).
#[derive(Clone, Copy, Debug, Default)]
pub struct HardswishOp;
impl<T: RealScalar> UnaryOp<T> for HardswishOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        let zero = <T as NumericElement>::ZERO;
        let three = T::from_f64(3.0);
        if x <= -three {
            return -zero;
        }
        let six = T::from_f64(6.0);
        x * (clamp_scalar(x + three, zero, six) / six)
    }
}
/// Hard swish gradient marker, taking the input: `1` for `x ≥ 3`,
/// `(2x + 3) / 6` on `(-3, 3)`, else `0`, propagating NaN.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardswishGradOp;
impl<T: RealScalar> UnaryOp<T> for HardswishGradOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let one = <T as NumericElement>::ONE;
        let two = one + one;
        let three = two + one;
        let six = T::from_f64(6.0);
        if x >= three {
            one
        } else if x > -three {
            (two * x + three) / six
        } else {
            <T as NumericElement>::ZERO
        }
    }
}
/// Exact GELU operation marker: `0.5 · x · erfc(-x / √2)`, accurate where
/// `1 + erf` would cancel, with the analytic limit at infinities.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluOp;
impl<T: RealScalar> UnaryOp<T> for GeluOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        let inv_sqrt_2 = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let half = T::from_f64(0.5);
        half * x * (-(x * inv_sqrt_2)).erfc()
    }
}
/// Exact GELU gradient marker, taking the input, with the analytic
/// `1`/`0` limits at infinities (the Gaussian term forms `±∞ · 0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluGradOp;
impl<T: RealScalar> UnaryOp<T> for GeluGradOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let inv_sqrt_2 = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
        let inv_sqrt_2pi = T::from_f64(0.398_942_280_401_432_7);
        let half = T::from_f64(0.5);
        let neg_half = T::from_f64(-0.5);
        half * (-(x * inv_sqrt_2)).erfc() + x * (neg_half * x * x).exp() * inv_sqrt_2pi
    }
}
/// Tanh-approximated GELU operation marker: `x · sigmoid(2z)`, with the
/// analytic limit at infinities.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhOp;
impl<T: RealScalar> UnaryOp<T> for GeluTanhOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        let z = gelu_tanh_arg(x);
        x * stable_sigmoid(z + z)
    }
}
/// Tanh-GELU gradient marker, taking the input: `s + 2·x·s·(1−s)·c0·(…)`,
/// `s = sigmoid(2z)`, with `1 − s` direct below the crossover and
/// `sigmoid(−w)` above, and the saturated-`s` short-circuit that avoids
/// forming `0 · ∞` where `x²` overflows.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhGradOp;
impl<T: RealScalar> UnaryOp<T> for GeluTanhGradOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let c0 = T::from_f64(0.797_884_560_802_865_4);
        let c1 = T::from_f64(0.044715);
        let z = gelu_tanh_arg(x);
        let w = z + z;
        let s = stable_sigmoid(w);
        let one = <T as NumericElement>::ONE;
        let two = one + one;
        let three = two + one;
        if w <= T::from_f64(GELU_TANH_GRAD_CROSSOVER) {
            let one_minus_s = one - s;
            if s * one_minus_s == <T as NumericElement>::ZERO {
                return s;
            }
            s + two * x * s * one_minus_s * c0 * (one + three * c1 * x * x)
        } else {
            let one_minus_s = stable_sigmoid(-w);
            let saturation = s * one_minus_s;
            if saturation == <T as NumericElement>::ZERO {
                return s;
            }
            s + two * x * saturation * c0 * (one + three * c1 * x * x)
        }
    }
}
/// SiLU operation marker: `x · sigmoid(x)`, with the analytic limit at
/// infinities.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluOp;
impl<T: RealScalar> UnaryOp<T> for SiluOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        x * stable_sigmoid(x)
    }
}
/// SiLU gradient marker, taking the input: `sig · (1 + x·(1 − sig))` with
/// `1 − sig` direct below the crossover and `sigmoid(−x)` above.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluGradOp;
impl<T: RealScalar> UnaryOp<T> for SiluGradOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let sig = stable_sigmoid(x);
        let one_minus_sig = if x <= T::from_f64(SILU_GRAD_CROSSOVER) {
            <T as NumericElement>::ONE - sig
        } else {
            stable_sigmoid(-x)
        };
        sig * (<T as NumericElement>::ONE + x * one_minus_sig)
    }
}
/// Mish operation marker: `x · tanh(softplus(x))`, with the analytic limit
/// at infinities.
#[derive(Clone, Copy, Debug, Default)]
pub struct MishOp;
impl<T: RealScalar> UnaryOp<T> for MishOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                x
            } else {
                -<T as NumericElement>::ZERO
            };
        }
        x * softplus_value(x).tanh()
    }
}
/// Mish gradient marker, taking the input: `t + x·(1−t²)·sigmoid(x)`,
/// `t = tanh(softplus(x))`, with `1 − t²` direct below the crossover and
/// `sech²` above.
#[derive(Clone, Copy, Debug, Default)]
pub struct MishGradOp;
impl<T: RealScalar> UnaryOp<T> for MishGradOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if is_infinite(x) {
            return if x > <T as NumericElement>::ZERO {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
        }
        let sp = softplus_value(x);
        let t = sp.tanh();
        let one = <T as NumericElement>::ONE;
        let one_minus_t_sq = if sp <= T::from_f64(MISH_GRAD_CROSSOVER) {
            one - t * t
        } else {
            let two = one + one;
            let four = two + two;
            let u = (-two * sp).exp();
            four * u / ((one + u) * (one + u))
        };
        t + x * one_minus_t_sq * stable_sigmoid(x)
    }
}
