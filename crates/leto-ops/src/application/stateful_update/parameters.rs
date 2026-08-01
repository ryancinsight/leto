use eunomia::{FloatElement, NumericElement};
use leto::{LetoError, Result};

use crate::RealScalar;

fn invalid(reason: &'static str) -> LetoError {
    LetoError::InvalidInput(reason.to_string())
}

fn positive<T: RealScalar>(value: T, reason: &'static str) -> Result<()> {
    if value.is_finite() && value > <T as NumericElement>::ZERO {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}

fn unit_interval<T: RealScalar>(value: T, reason: &'static str) -> Result<()> {
    if value.is_finite()
        && value >= <T as NumericElement>::ZERO
        && value < <T as NumericElement>::ONE
    {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}

/// Validated SGD learning rate and momentum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdParameters<T> {
    pub(super) learning_rate: T,
    pub(super) momentum: T,
}

impl<T: RealScalar> SgdParameters<T> {
    /// Validate and construct SGD parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] unless the learning rate is finite
    /// and positive and momentum is finite in `[0, 1)`.
    pub fn new(learning_rate: T, momentum: T) -> Result<Self> {
        positive(
            learning_rate,
            "SGD learning rate must be finite and positive",
        )?;
        unit_interval(momentum, "SGD momentum must be finite in [0, 1)")?;
        Ok(Self {
            learning_rate,
            momentum,
        })
    }
}

/// Validated Adam learning, moment, epsilon, and step parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamParameters<T> {
    pub(super) learning_rate: T,
    pub(super) beta_one: T,
    pub(super) beta_two: T,
    pub(super) epsilon: T,
    pub(super) bias_correction_one: T,
    pub(super) bias_correction_two: T,
}

impl<T: RealScalar> AdamParameters<T> {
    /// Validate and construct Adam parameters, computing bias corrections once.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] unless the learning rate and epsilon
    /// are finite and positive, both betas are finite in `[0, 1)`, and `step`
    /// is positive with representable positive bias corrections.
    pub fn new(learning_rate: T, beta_one: T, beta_two: T, epsilon: T, step: u64) -> Result<Self> {
        positive(
            learning_rate,
            "Adam learning rate must be finite and positive",
        )?;
        unit_interval(beta_one, "Adam beta one must be finite in [0, 1)")?;
        unit_interval(beta_two, "Adam beta two must be finite in [0, 1)")?;
        positive(epsilon, "Adam epsilon must be finite and positive")?;
        if step == 0 {
            return Err(invalid("Adam step must be positive"));
        }
        let one = <T as NumericElement>::ONE;
        let bias_correction_one = one - pow_unsigned(beta_one, step);
        let bias_correction_two = one - pow_unsigned(beta_two, step);
        positive(
            bias_correction_one,
            "Adam first bias correction must be positive",
        )?;
        positive(
            bias_correction_two,
            "Adam second bias correction must be positive",
        )?;
        Ok(Self {
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            bias_correction_one,
            bias_correction_two,
        })
    }
}

/// Validated AdamW parameters with decoupled weight decay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWParameters<T> {
    pub(super) adam: AdamParameters<T>,
    pub(super) weight_decay: T,
}

impl<T: RealScalar> AdamWParameters<T> {
    /// Validate and construct AdamW parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when the embedded Adam parameters are
    /// invalid or weight decay is not finite and non-negative.
    pub fn new(
        learning_rate: T,
        beta_one: T,
        beta_two: T,
        epsilon: T,
        weight_decay: T,
        step: u64,
    ) -> Result<Self> {
        if !weight_decay.is_finite() || weight_decay < <T as NumericElement>::ZERO {
            return Err(invalid(
                "AdamW weight decay must be finite and non-negative",
            ));
        }
        Ok(Self {
            adam: AdamParameters::new(learning_rate, beta_one, beta_two, epsilon, step)?,
            weight_decay,
        })
    }
}

/// Validated RMSProp learning rate, decay, and epsilon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsPropParameters<T> {
    pub(super) learning_rate: T,
    pub(super) alpha: T,
    pub(super) epsilon: T,
}

impl<T: RealScalar> RmsPropParameters<T> {
    /// Validate and construct RMSProp parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] unless learning rate and epsilon are
    /// finite and positive and alpha is finite in `[0, 1)`.
    pub fn new(learning_rate: T, alpha: T, epsilon: T) -> Result<Self> {
        positive(
            learning_rate,
            "RMSProp learning rate must be finite and positive",
        )?;
        unit_interval(alpha, "RMSProp alpha must be finite in [0, 1)")?;
        positive(epsilon, "RMSProp epsilon must be finite and positive")?;
        Ok(Self {
            learning_rate,
            alpha,
            epsilon,
        })
    }
}

/// Validated AdaGrad learning rate and epsilon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaGradParameters<T> {
    pub(super) learning_rate: T,
    pub(super) epsilon: T,
}

impl<T: RealScalar> AdaGradParameters<T> {
    /// Validate and construct AdaGrad parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] unless learning rate and epsilon are
    /// finite and positive.
    pub fn new(learning_rate: T, epsilon: T) -> Result<Self> {
        positive(
            learning_rate,
            "AdaGrad learning rate must be finite and positive",
        )?;
        positive(epsilon, "AdaGrad epsilon must be finite and positive")?;
        Ok(Self {
            learning_rate,
            epsilon,
        })
    }
}

fn pow_unsigned<T: FloatElement>(mut base: T, mut exponent: u64) -> T {
    let mut result = <T as NumericElement>::ONE;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result *= base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base *= base;
        }
    }
    result
}
