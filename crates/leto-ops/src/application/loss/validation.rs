use super::{CrossEntropyError, CrossEntropyOperand, CrossEntropyResult};
use crate::domain::real::RealScalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView, ArrayViewMut, LetoError};

#[derive(Debug, Clone, Copy)]
pub(super) struct CrossEntropyPlan<T> {
    pub(super) batch: usize,
    pub(super) classes: usize,
    pub(super) inverse_batch: T,
    probability_tolerance: T,
}

fn validate_view<T, const N: usize>(
    operand: CrossEntropyOperand,
    view: &ArrayView<'_, T, N>,
) -> CrossEntropyResult<()> {
    view.layout()
        .validate_storage_len(view.data().len())
        .map_err(|source| CrossEntropyError::Layout { operand, source })
}

fn validate_mut_view<T, const N: usize>(
    operand: CrossEntropyOperand,
    view: &ArrayViewMut<'_, T, N>,
) -> CrossEntropyResult<()> {
    view.layout()
        .validate_storage_len(view.data().len())
        .map_err(|source| CrossEntropyError::Layout { operand, source })?;
    if !view
        .layout()
        .is_injective()
        .map_err(|source| CrossEntropyError::Layout { operand, source })?
    {
        return Err(CrossEntropyError::Layout {
            operand,
            source: LetoError::StorageError {
                reason: "cross-entropy mutable view contains aliased logical offsets".to_string(),
            },
        });
    }
    Ok(())
}

fn validate_targets(targets: &[usize], batch: usize, classes: usize) -> CrossEntropyResult<()> {
    if targets.len() != batch {
        return Err(CrossEntropyError::TargetCount {
            expected: batch,
            actual: targets.len(),
        });
    }
    if let Some((batch, &target)) = targets
        .iter()
        .enumerate()
        .find(|(_, target)| **target >= classes)
    {
        return Err(CrossEntropyError::TargetOutOfRange {
            batch,
            target,
            classes,
        });
    }
    Ok(())
}

fn plan<T: RealScalar + RealField>(
    shape: [usize; 2],
    targets: &[usize],
) -> CrossEntropyResult<CrossEntropyPlan<T>> {
    let [batch, classes] = shape;
    if batch == 0 {
        return Err(CrossEntropyError::EmptyBatch);
    }
    if classes == 0 {
        return Err(CrossEntropyError::EmptyClasses);
    }
    validate_targets(targets, batch, classes)?;
    let batch_scalar = T::from_usize(batch);
    if !NumericElement::is_finite(batch_scalar) {
        return Err(CrossEntropyError::ScalarExtent {
            dimension: "batch",
            extent: batch,
        });
    }
    let class_scalar = T::from_usize(classes);
    if !NumericElement::is_finite(class_scalar) {
        return Err(CrossEntropyError::ScalarExtent {
            dimension: "class",
            extent: classes,
        });
    }
    // A normalized probability incurs one division rounding, then validation
    // sums `classes` stored values. Higham's gamma bound for the latter is
    // gamma(k) = k*epsilon/(1-k*epsilon), k = classes - 1. The product below
    // composes the division and summation relative-error bounds.
    let summation_steps = T::from_usize(classes.saturating_sub(1));
    let summation_error = <T as RealField>::EPSILON * summation_steps;
    if summation_error >= <T as NumericElement>::ONE {
        return Err(CrossEntropyError::ProbabilityResolution { classes });
    }
    let gamma = summation_error / (<T as NumericElement>::ONE - summation_error);
    let probability_tolerance =
        gamma + <T as RealField>::EPSILON * (<T as NumericElement>::ONE + gamma);
    if !NumericElement::is_finite(probability_tolerance)
        || probability_tolerance >= T::from_f64(0.5)
    {
        return Err(CrossEntropyError::ProbabilityResolution { classes });
    }
    Ok(CrossEntropyPlan {
        batch,
        classes,
        inverse_batch: <T as NumericElement>::ONE / batch_scalar,
        probability_tolerance,
    })
}

fn validate_forward_values<T: RealScalar>(
    logits: &ArrayView<'_, T, 2>,
    plan: CrossEntropyPlan<T>,
) -> CrossEntropyResult<()> {
    let class_log_bound = FloatElement::ln(T::from_usize(plan.classes));
    for batch in 0..plan.batch {
        let mut row_min = *logits
            .get([batch, 0])
            .expect("invariant: validated cross-entropy input index is in bounds");
        let mut row_max = row_min;
        if !NumericElement::is_finite(row_min) {
            return Err(CrossEntropyError::NonFinite {
                operand: CrossEntropyOperand::Logits,
            });
        }
        for class in 1..plan.classes {
            let value = *logits
                .get([batch, class])
                .expect("invariant: validated cross-entropy input index is in bounds");
            if !NumericElement::is_finite(value) {
                return Err(CrossEntropyError::NonFinite {
                    operand: CrossEntropyOperand::Logits,
                });
            }
            if value < row_min {
                row_min = value;
            }
            if value > row_max {
                row_max = value;
            }
        }
        let range_bound = row_max - row_min;
        if !NumericElement::is_finite(range_bound)
            || !NumericElement::is_finite(class_log_bound + range_bound)
        {
            return Err(CrossEntropyError::ArithmeticNonFinite { batch });
        }
    }
    Ok(())
}

fn validate_probabilities<T: RealScalar + RealField>(
    probabilities: &ArrayView<'_, T, 2>,
    plan: CrossEntropyPlan<T>,
) -> CrossEntropyResult<()> {
    for batch in 0..plan.batch {
        let mut sum = <T as NumericElement>::ZERO;
        for class in 0..plan.classes {
            let probability = *probabilities
                .get([batch, class])
                .expect("invariant: validated probability index is in bounds");
            if !NumericElement::is_finite(probability) {
                return Err(CrossEntropyError::NonFinite {
                    operand: CrossEntropyOperand::Probabilities,
                });
            }
            if probability < <T as NumericElement>::ZERO || probability > <T as NumericElement>::ONE
            {
                return Err(CrossEntropyError::InvalidProbabilities { batch });
            }
            sum += probability;
        }
        if !NumericElement::is_finite(sum)
            || NumericElement::abs(sum - <T as NumericElement>::ONE) > plan.probability_tolerance
        {
            return Err(CrossEntropyError::InvalidProbabilities { batch });
        }
    }
    Ok(())
}

fn expect_shape<const N: usize>(
    operand: CrossEntropyOperand,
    actual: [usize; N],
    expected: [usize; N],
) -> CrossEntropyResult<()> {
    if actual != expected {
        return Err(CrossEntropyError::Shape {
            operand,
            expected: expected.into(),
            actual: actual.into(),
        });
    }
    Ok(())
}

pub(super) fn validate_forward<T: RealScalar + RealField>(
    logits: &ArrayView<'_, T, 2>,
    targets: &[usize],
    loss: &ArrayViewMut<'_, T, 1>,
    probabilities: &ArrayViewMut<'_, T, 2>,
) -> CrossEntropyResult<CrossEntropyPlan<T>> {
    validate_view(CrossEntropyOperand::Logits, logits)?;
    validate_mut_view(CrossEntropyOperand::Loss, loss)?;
    validate_mut_view(CrossEntropyOperand::Probabilities, probabilities)?;
    expect_shape(CrossEntropyOperand::Loss, loss.shape(), [1])?;
    expect_shape(
        CrossEntropyOperand::Probabilities,
        probabilities.shape(),
        logits.shape(),
    )?;
    let plan = plan(logits.shape(), targets)?;
    validate_forward_values(logits, plan)?;
    Ok(plan)
}

pub(super) fn validate_backward<T: RealScalar + RealField>(
    output_gradient: &ArrayView<'_, T, 1>,
    probabilities: &ArrayView<'_, T, 2>,
    targets: &[usize],
    logit_gradient: &ArrayViewMut<'_, T, 2>,
) -> CrossEntropyResult<CrossEntropyPlan<T>> {
    validate_view(CrossEntropyOperand::OutputGradient, output_gradient)?;
    validate_view(CrossEntropyOperand::Probabilities, probabilities)?;
    validate_mut_view(CrossEntropyOperand::LogitGradient, logit_gradient)?;
    expect_shape(
        CrossEntropyOperand::LogitGradient,
        logit_gradient.shape(),
        probabilities.shape(),
    )?;
    expect_shape(
        CrossEntropyOperand::OutputGradient,
        output_gradient.shape(),
        [1],
    )?;
    let upstream = *output_gradient
        .get([0])
        .expect("invariant: validated output-gradient index is in bounds");
    if !NumericElement::is_finite(upstream) {
        return Err(CrossEntropyError::NonFinite {
            operand: CrossEntropyOperand::OutputGradient,
        });
    }
    let plan = plan(probabilities.shape(), targets)?;
    validate_probabilities(probabilities, plan)?;
    let scale = upstream * plan.inverse_batch;
    for (batch, &target) in targets.iter().enumerate() {
        for class in 0..plan.classes {
            let destination = *logit_gradient
                .get([batch, class])
                .expect("invariant: validated logit-gradient index is in bounds");
            if !NumericElement::is_finite(destination) {
                return Err(CrossEntropyError::NonFinite {
                    operand: CrossEntropyOperand::LogitGradient,
                });
            }
            let probability = *probabilities
                .get([batch, class])
                .expect("invariant: validated probability index is in bounds");
            let indicator = if class == target {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
            let result = destination + (probability - indicator) * scale;
            if !NumericElement::is_finite(result) {
                return Err(CrossEntropyError::ArithmeticNonFinite { batch });
            }
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::plan;
    use crate::application::loss::CrossEntropyError;

    #[test]
    fn probability_plan_rejects_uninformative_error_bound() {
        const CLASSES: usize = 1 << 22;
        let error = plan::<f32>([1, CLASSES], &[0])
            .expect_err("the probability error bound must remain informative");
        assert_eq!(
            error,
            CrossEntropyError::ProbabilityResolution { classes: CLASSES }
        );
    }
}
