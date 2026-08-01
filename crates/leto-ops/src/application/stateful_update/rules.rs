use eunomia::NumericElement;
use leto::{ArrayView, ArrayViewMut, Result};

use super::parameters::{
    AdaGradParameters, AdamParameters, AdamWParameters, RmsPropParameters, SgdParameters,
};
use super::validation::{validate_one, validate_two};
use crate::{RealScalar, zip_mut_with};

mod sealed {
    pub trait Sealed {}
}

/// Closed compile-time rule contract for stateful updates.
pub trait StatefulUpdateRule<T: RealScalar, const N: usize>: sealed::Sealed {
    /// Validated rule parameters.
    type Parameters: Copy;
    /// Mutable state-view family required by this rule.
    type State<'a>
    where
        Self: 'a,
        T: 'a;

    /// Validate all views, then execute the monomorphized update.
    ///
    /// Finite and non-finite operand values follow the scalar type's IEEE
    /// arithmetic. Rule parameters are validated separately at construction.
    ///
    /// # Errors
    ///
    /// Returns a typed layout error when storage bounds, shapes, or mutable-view
    /// injectivity do not satisfy the update contract.
    fn apply<'a>(
        parameter: ArrayViewMut<'a, T, N>,
        gradient: ArrayView<'a, T, N>,
        state: Self::State<'a>,
        parameters: Self::Parameters,
    ) -> Result<()>;
}

/// Execute one scalar-preserving stateful update over borrowed views.
///
/// Finite and non-finite operand values follow the scalar type's IEEE
/// arithmetic. Rule parameters are validated separately at construction.
///
/// # Errors
///
/// Returns a typed layout error when storage bounds, shapes, or mutable-view
/// injectivity do not satisfy the update contract.
pub fn stateful_update<'a, T, Rule, const N: usize>(
    parameter: ArrayViewMut<'a, T, N>,
    gradient: ArrayView<'a, T, N>,
    state: Rule::State<'a>,
    parameters: Rule::Parameters,
) -> Result<()>
where
    T: RealScalar + 'a,
    Rule: StatefulUpdateRule<T, N> + 'a,
{
    Rule::apply(parameter, gradient, state, parameters)
}

/// Stochastic gradient descent with momentum.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sgd;
/// Adam adaptive-moment update.
#[derive(Clone, Copy, Debug, Default)]
pub struct Adam;
/// Adam with decoupled weight decay.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdamW;
/// RMSProp squared-gradient update.
#[derive(Clone, Copy, Debug, Default)]
pub struct RmsProp;
/// AdaGrad accumulated-squared-gradient update.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaGrad;

impl sealed::Sealed for Sgd {}
impl sealed::Sealed for Adam {}
impl sealed::Sealed for AdamW {}
impl sealed::Sealed for RmsProp {}
impl sealed::Sealed for AdaGrad {}

impl<T: RealScalar, const N: usize> StatefulUpdateRule<T, N> for Sgd {
    type Parameters = SgdParameters<T>;
    type State<'a>
        = ArrayViewMut<'a, T, N>
    where
        T: 'a;

    fn apply<'a>(
        parameter: ArrayViewMut<'a, T, N>,
        gradient: ArrayView<'a, T, N>,
        state: Self::State<'a>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        validate_one(&parameter, &gradient, &state)?;
        zip_mut_with(
            (parameter, state),
            &gradient,
            |(parameter, velocity), gradient| {
                *velocity = *velocity * parameters.momentum + *gradient;
                *parameter -= parameters.learning_rate * *velocity;
            },
        )
    }
}

fn adam_update<T: RealScalar>(
    parameter: &mut T,
    first: &mut T,
    second: &mut T,
    gradient: T,
    parameters: AdamParameters<T>,
) {
    let one = <T as NumericElement>::ONE;
    *first = *first * parameters.beta_one + (one - parameters.beta_one) * gradient;
    *second = *second * parameters.beta_two + (one - parameters.beta_two) * gradient * gradient;
    *parameter -= parameters.learning_rate * (*first / parameters.bias_correction_one)
        / (<T as NumericElement>::sqrt(*second / parameters.bias_correction_two)
            + parameters.epsilon);
}

impl<T: RealScalar, const N: usize> StatefulUpdateRule<T, N> for Adam {
    type Parameters = AdamParameters<T>;
    type State<'a>
        = (ArrayViewMut<'a, T, N>, ArrayViewMut<'a, T, N>)
    where
        T: 'a;

    fn apply<'a>(
        parameter: ArrayViewMut<'a, T, N>,
        gradient: ArrayView<'a, T, N>,
        state: Self::State<'a>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        let (first, second) = state;
        validate_two(&parameter, &gradient, &first, &second)?;
        zip_mut_with(
            (parameter, first, second),
            &gradient,
            |(parameter, first, second), gradient| {
                adam_update(parameter, first, second, *gradient, parameters);
            },
        )
    }
}

impl<T: RealScalar, const N: usize> StatefulUpdateRule<T, N> for AdamW {
    type Parameters = AdamWParameters<T>;
    type State<'a>
        = (ArrayViewMut<'a, T, N>, ArrayViewMut<'a, T, N>)
    where
        T: 'a;

    fn apply<'a>(
        parameter: ArrayViewMut<'a, T, N>,
        gradient: ArrayView<'a, T, N>,
        state: Self::State<'a>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        let (first, second) = state;
        validate_two(&parameter, &gradient, &first, &second)?;
        zip_mut_with(
            (parameter, first, second),
            &gradient,
            |(parameter, first, second), gradient| {
                let previous = *parameter;
                adam_update(parameter, first, second, *gradient, parameters.adam);
                *parameter -= parameters.adam.learning_rate * parameters.weight_decay * previous;
            },
        )
    }
}

impl<T: RealScalar, const N: usize> StatefulUpdateRule<T, N> for RmsProp {
    type Parameters = RmsPropParameters<T>;
    type State<'a>
        = ArrayViewMut<'a, T, N>
    where
        T: 'a;

    fn apply<'a>(
        parameter: ArrayViewMut<'a, T, N>,
        gradient: ArrayView<'a, T, N>,
        state: Self::State<'a>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        validate_one(&parameter, &gradient, &state)?;
        zip_mut_with(
            (parameter, state),
            &gradient,
            |(parameter, average), gradient| {
                let one = <T as NumericElement>::ONE;
                *average =
                    *average * parameters.alpha + (one - parameters.alpha) * *gradient * *gradient;
                *parameter -= parameters.learning_rate * *gradient
                    / (<T as NumericElement>::sqrt(*average) + parameters.epsilon);
            },
        )
    }
}

impl<T: RealScalar, const N: usize> StatefulUpdateRule<T, N> for AdaGrad {
    type Parameters = AdaGradParameters<T>;
    type State<'a>
        = ArrayViewMut<'a, T, N>
    where
        T: 'a;

    fn apply<'a>(
        parameter: ArrayViewMut<'a, T, N>,
        gradient: ArrayView<'a, T, N>,
        state: Self::State<'a>,
        parameters: Self::Parameters,
    ) -> Result<()> {
        validate_one(&parameter, &gradient, &state)?;
        zip_mut_with(
            (parameter, state),
            &gradient,
            |(parameter, sum), gradient| {
                *sum += *gradient * *gradient;
                *parameter -= parameters.learning_rate * *gradient
                    / (<T as NumericElement>::sqrt(*sum) + parameters.epsilon);
            },
        )
    }
}
