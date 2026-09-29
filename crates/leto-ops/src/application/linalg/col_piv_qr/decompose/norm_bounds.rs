//! Conservative intervals for cached column pivot keys.
//!
//! For each shipped binary format, let `ε` be the gap above one and `σ` the
//! least positive subnormal. A native operation, including the two rounding
//! stages used by reduced formats, obeys `|fl(z) - z| ≤ ε|z| + σ`: composing
//! round-to-nearest `f32` and destination rounding gives relative error below
//! `ε` and additive error below `σ`. This assumes round-to-nearest, ties-to-even
//! and gradual underflow; altered rounding modes, flush-to-zero, and
//! denormals-are-zero are unsupported. Endpoint rounding uses a margin of
//! `4ρ|x| + 8τ`, where `ρ = ε` and `τ = σ`. For `ρ ≤ 1/64`, rounding the
//! margin product, its sum, and the final endpoint leaves outward slack of
//! `[4ρ(1−ρ)³−ρ]|x| + [5−13ρ+7ρ²]τ`, which exceeds the source operation's
//! error bound `(ρ|x| + τ)/(1−ρ)`. Any non-finite intermediate or reduction
//! too long for the bound invalidates the certificate and selects exact tail
//! recomputation.

use crate::application::linalg::householder::Reflector;
use crate::application::linalg::thresholds::{machine_epsilon, safe_min};
use crate::domain::real::RealScalar;

#[derive(Clone, Copy)]
pub(super) struct TailBounds<T> {
    pub(super) lower: T,
    pub(super) upper: T,
}

#[derive(Clone, Copy)]
struct Interval<T> {
    lower: T,
    upper: T,
}

#[derive(Clone, Copy)]
pub(super) struct Arithmetic<T> {
    rho: T,
    tau: T,
    endpoint_rho: T,
    endpoint_tau: T,
}

#[derive(Clone, Copy)]
pub(super) struct ReflectorError<T> {
    relative: T,
    absolute: T,
}

impl<T: RealScalar> Arithmetic<T> {
    pub(super) fn new() -> Option<Self> {
        let epsilon = machine_epsilon::<T>();
        let sigma = safe_min::<T>().mul(epsilon);
        let rho = epsilon;
        let tau = sigma;
        let endpoint_rho = rho.scale_binary(2);
        let endpoint_tau = tau.scale_binary(3);
        let max_rho = T::ONE.scale_binary(-6);
        if [sigma, rho, tau, endpoint_rho, endpoint_tau, max_rho]
            .into_iter()
            .any(|value| !value.is_finite() || value <= T::ZERO)
            || rho > max_rho
        {
            return None;
        }
        Some(Self {
            rho,
            tau,
            endpoint_rho,
            endpoint_tau,
        })
    }

    pub(super) fn enclose_squared_sum(self, rounded: T, length: usize) -> Option<TailBounds<T>> {
        if !rounded.is_finite() || rounded < T::ZERO {
            return None;
        }
        let (gamma, additive) = self.reduction_error(length)?;
        let rounded = Interval::point(rounded)?;
        let additive = Interval::point(additive)?;
        let gamma = Interval::point(gamma)?;
        let one = Interval::point(T::ONE)?;

        let lower_numerator = rounded.sub(additive, self)?.lower.max(T::ZERO);
        let lower_denominator = one.add(gamma, self)?.upper;
        let lower = Interval::point(lower_numerator)?
            .div(Interval::point(lower_denominator)?, self)?
            .lower
            .max(T::ZERO);

        let upper_numerator = rounded.add(additive, self)?.upper;
        let upper_denominator = one.sub(gamma, self)?.lower;
        if upper_denominator <= T::ZERO {
            return None;
        }
        let upper = Interval::point(upper_numerator)?
            .div(Interval::point(upper_denominator)?, self)?
            .upper;
        TailBounds::new(lower, upper)
    }

    pub(super) fn pivot_key(self, norm: TailBounds<T>, length: usize) -> Option<TailBounds<T>> {
        let (gamma, additive) = self.reduction_error(length)?;
        let one = Interval::point(T::ONE)?;
        let gamma = Interval::point(gamma)?;
        let norm = Interval::new(norm.lower, norm.upper)?;
        let lower_factor = one.sub(gamma, self)?;
        let lower = lower_factor
            .mul(Interval::point(norm.lower)?, self)?
            .sub(Interval::point(additive)?, self)?
            .lower
            .max(T::ZERO);
        let upper_factor = one.add(gamma, self)?;
        let upper = upper_factor
            .mul(Interval::point(norm.upper)?, self)?
            .add(Interval::point(additive)?, self)?
            .upper;
        TailBounds::new(lower, upper)
    }

    pub(super) fn after_reflector(
        self,
        norm: TailBounds<T>,
        effect: ReflectorError<T>,
        removed: T,
    ) -> Option<TailBounds<T>> {
        let transformed = effect.apply(self, norm)?;
        self.remove_entry(transformed, removed)
    }

    pub(super) fn remove_entry(self, norm: TailBounds<T>, removed: T) -> Option<TailBounds<T>> {
        let magnitude = Interval::point(removed.abs())?;
        let removed_squared = magnitude.mul(magnitude, self)?;
        let tail = Interval::new(norm.lower, norm.upper)?.sub(removed_squared, self)?;
        if tail.upper < T::ZERO {
            return None;
        }
        TailBounds::new(tail.lower.max(T::ZERO), tail.upper)
    }

    pub(super) fn reflector_error(self, reflector: &Reflector<T>) -> Option<ReflectorError<T>> {
        let length = reflector.v.len();
        let mut squared_norm = Interval::point(T::ZERO)?;
        for &value in &reflector.v {
            let value = Interval::point(value)?;
            squared_norm = squared_norm.add(value.mul(value, self)?, self)?;
        }

        let beta = reflector.beta.abs();
        let beta_norm = Interval::point(beta)?.mul(squared_norm, self)?;
        let twice = T::ONE.scale_binary(1);
        let delta = beta_norm
            .sub(Interval::point(twice)?, self)?
            .abs_upper(self)?;
        let vector_norm = squared_norm.sqrt(self)?.upper;
        let squared_norm_upper = squared_norm.upper;
        let (gamma, additive) = self.reduction_error(length)?;
        let one = T::ONE;
        let one_plus_rho = self.add_upper(one, self.rho)?;
        let one_plus_gamma = self.add_upper(one, gamma)?;
        let product_upper = self.mul_upper(one_plus_rho, one_plus_gamma)?;
        let g1 = Interval::point(product_upper)?
            .sub(Interval::point(one)?, self)?
            .upper;
        if g1 < T::ZERO || !g1.is_finite() {
            return None;
        }
        let dot_additive = self.add_upper(
            self.mul_upper(self.mul_upper(beta, one_plus_rho)?, additive)?,
            self.tau,
        )?;
        let two = T::ONE.scale_binary(1);
        let c2 = self.add_upper(
            self.mul_upper(two, self.rho)?,
            self.mul_upper(self.rho, self.rho)?,
        )?;
        let update_factor = self.add_upper(g1, self.mul_upper(c2, self.add_upper(one, g1)?)?)?;
        let relative = self.add_upper(
            self.rho,
            self.mul_upper(self.mul_upper(beta, squared_norm_upper)?, update_factor)?,
        )?;
        let vector_error = self.mul_upper(vector_norm, dot_additive)?;
        let count_upper = self.round_up(T::from_usize(length))?;
        let update_rounding = self.mul_upper(
            self.mul_upper(
                self.sqrt_upper(count_upper)?,
                self.add_upper(two, self.rho)?,
            )?,
            self.tau,
        )?;
        let absolute = self.add_upper(
            self.mul_upper(self.add_upper(one, c2)?, vector_error)?,
            update_rounding,
        )?;
        let relative = self.add_upper(delta, relative)?;
        Some(ReflectorError { relative, absolute })
    }

    fn reduction_error(self, length: usize) -> Option<(T, T)> {
        let operations = length.checked_mul(2)?;
        let count = self.round_up(T::from_usize(operations))?;
        if count < T::ZERO {
            return None;
        }
        let product = self.mul_upper(count, self.rho)?;
        if product >= T::ONE {
            return None;
        }
        let denominator = Interval::point(T::ONE)?
            .sub(Interval::point(product)?, self)?
            .lower;
        if denominator <= T::ZERO {
            return None;
        }
        let gamma = self.div_upper(product, denominator)?;
        let additive = self.div_upper(self.mul_upper(count, self.tau)?, denominator)?;
        (gamma < T::ONE).then_some((gamma, additive))
    }

    fn add_upper(self, lhs: T, rhs: T) -> Option<T> {
        Some(
            Interval::point(lhs)?
                .add(Interval::point(rhs)?, self)?
                .upper,
        )
    }

    fn mul_upper(self, lhs: T, rhs: T) -> Option<T> {
        Some(
            Interval::point(lhs)?
                .mul(Interval::point(rhs)?, self)?
                .upper,
        )
    }

    fn div_upper(self, lhs: T, rhs: T) -> Option<T> {
        Some(
            Interval::point(lhs)?
                .div(Interval::point(rhs)?, self)?
                .upper,
        )
    }

    fn sqrt_upper(self, value: T) -> Option<T> {
        Some(Interval::point(value)?.sqrt(self)?.upper)
    }

    fn round_down(self, rounded: T) -> Option<T> {
        self.round_outward(rounded, false)
    }

    fn round_up(self, rounded: T) -> Option<T> {
        self.round_outward(rounded, true)
    }

    fn round_outward(self, rounded: T, upward: bool) -> Option<T> {
        if !rounded.is_finite() {
            return None;
        }
        let margin = rounded.abs().mul(self.endpoint_rho).add(self.endpoint_tau);
        if !margin.is_finite() {
            return None;
        }
        let endpoint = if upward {
            rounded.add(margin)
        } else {
            rounded.sub(margin)
        };
        endpoint.is_finite().then_some(endpoint)
    }
}

impl<T: RealScalar> ReflectorError<T> {
    fn apply(self, arithmetic: Arithmetic<T>, norm: TailBounds<T>) -> Option<TailBounds<T>> {
        let x_norm = arithmetic.sqrt_upper(norm.upper)?;
        let error =
            arithmetic.add_upper(arithmetic.mul_upper(self.relative, x_norm)?, self.absolute)?;
        let squared_drift = arithmetic.add_upper(
            arithmetic.mul_upper(arithmetic.mul_upper(T::ONE.scale_binary(1), x_norm)?, error)?,
            arithmetic.mul_upper(error, error)?,
        )?;
        let lower = if norm.lower > squared_drift {
            arithmetic
                .round_down(norm.lower.sub(squared_drift))?
                .max(T::ZERO)
        } else {
            T::ZERO
        };
        let upper = arithmetic.add_upper(norm.upper, squared_drift)?;
        TailBounds::new(lower, upper)
    }
}

impl<T: RealScalar> Interval<T> {
    fn point(value: T) -> Option<Self> {
        value.is_finite().then_some(Self {
            lower: value,
            upper: value,
        })
    }

    fn new(lower: T, upper: T) -> Option<Self> {
        (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Self { lower, upper })
    }

    fn add(self, rhs: Self, arithmetic: Arithmetic<T>) -> Option<Self> {
        let lower = self.lower.add(rhs.lower);
        let upper = self.upper.add(rhs.upper);
        Self::new(arithmetic.round_down(lower)?, arithmetic.round_up(upper)?)
    }

    fn sub(self, rhs: Self, arithmetic: Arithmetic<T>) -> Option<Self> {
        let lower = self.lower.sub(rhs.upper);
        let upper = self.upper.sub(rhs.lower);
        Self::new(arithmetic.round_down(lower)?, arithmetic.round_up(upper)?)
    }

    fn mul(self, rhs: Self, arithmetic: Arithmetic<T>) -> Option<Self> {
        let products = [
            self.lower.mul(rhs.lower),
            self.lower.mul(rhs.upper),
            self.upper.mul(rhs.lower),
            self.upper.mul(rhs.upper),
        ];
        if products.into_iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut lower = products[0];
        let mut upper = products[0];
        for product in products.into_iter().skip(1) {
            if product < lower {
                lower = product;
            }
            if product > upper {
                upper = product;
            }
        }
        Self::new(arithmetic.round_down(lower)?, arithmetic.round_up(upper)?)
    }

    fn div(self, rhs: Self, arithmetic: Arithmetic<T>) -> Option<Self> {
        if rhs.lower <= T::ZERO && rhs.upper >= T::ZERO {
            return None;
        }
        let quotients = [
            self.lower.div(rhs.lower),
            self.lower.div(rhs.upper),
            self.upper.div(rhs.lower),
            self.upper.div(rhs.upper),
        ];
        if quotients.into_iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut lower = quotients[0];
        let mut upper = quotients[0];
        for quotient in quotients.into_iter().skip(1) {
            if quotient < lower {
                lower = quotient;
            }
            if quotient > upper {
                upper = quotient;
            }
        }
        Self::new(arithmetic.round_down(lower)?, arithmetic.round_up(upper)?)
    }

    fn sqrt(self, arithmetic: Arithmetic<T>) -> Option<Self> {
        if self.lower < T::ZERO {
            return None;
        }
        let lower = arithmetic.round_down(self.lower.sqrt())?.max(T::ZERO);
        let upper = arithmetic.round_up(self.upper.sqrt())?;
        if upper < T::ZERO {
            return None;
        }
        Self::new(lower, upper)
    }

    fn abs_upper(self, arithmetic: Arithmetic<T>) -> Option<T> {
        let lower = self.lower.abs();
        let upper = self.upper.abs();
        let maximum = if lower > upper { lower } else { upper };
        arithmetic.round_up(maximum)
    }
}

impl<T: RealScalar> TailBounds<T> {
    fn new(lower: T, upper: T) -> Option<Self> {
        if lower < T::ZERO {
            return None;
        }
        let interval = Interval::new(lower, upper)?;
        Some(Self {
            lower: interval.lower,
            upper: interval.upper,
        })
    }

    #[cfg(test)]
    pub(super) fn contains(self, value: T) -> bool {
        self.lower <= value && value <= self.upper
    }
}

#[cfg(test)]
mod tests {
    use super::Arithmetic;
    use crate::application::linalg::thresholds::{machine_epsilon, safe_min};
    use crate::domain::real::RealScalar;

    fn pivot_keys_remain_enclosed_at_exponent_boundaries<T: RealScalar>() {
        let epsilon = machine_epsilon::<T>();
        let sigma = safe_min::<T>().mul(epsilon);
        let arithmetic = Arithmetic::new().expect("shipped format meets the error bound");
        let values = [
            T::ZERO,
            sigma,
            sigma.scale_binary(1),
            safe_min::<T>(),
            safe_min::<T>().scale_binary(1),
            T::ONE.scale_binary(-1),
            T::ONE,
        ];

        for value in values {
            let Some(bounds) = arithmetic.enclose_squared_sum(value, 1) else {
                panic!("finite boundary value must admit a norm interval: {value:?}");
            };
            let Some(key) = arithmetic.pivot_key(bounds, 1) else {
                panic!("finite boundary value must admit a pivot interval: {value:?}");
            };
            assert!(
                key.contains(value),
                "native key {value:?} outside [{:?}, {:?}] for {}",
                key.lower,
                key.upper,
                core::any::type_name::<T>()
            );
        }
    }

    #[test]
    fn pivot_key_intervals_cover_normal_and_subnormal_boundaries() {
        use eunomia::{Bf16, F16};

        pivot_keys_remain_enclosed_at_exponent_boundaries::<f64>();
        pivot_keys_remain_enclosed_at_exponent_boundaries::<f32>();
        pivot_keys_remain_enclosed_at_exponent_boundaries::<F16>();
        pivot_keys_remain_enclosed_at_exponent_boundaries::<Bf16>();
    }
}
