use super::{AttentionError, AttentionGradients, AttentionMask, AttentionOperand, AttentionResult};
use crate::domain::real::RealScalar;
use eunomia::{NumericElement, RealField};
use leto::{ArrayView, ArrayViewMut, LetoError};

#[derive(Debug, Clone, Copy)]
pub(super) struct AttentionPlan {
    pub(super) batch: usize,
    pub(super) query_len: usize,
    pub(super) key_len: usize,
    pub(super) key_dim: usize,
    pub(super) value_dim: usize,
}

impl AttentionPlan {
    pub(super) fn score_elements(self) -> AttentionResult<usize> {
        self.batch
            .checked_mul(self.query_len)
            .and_then(|value| value.checked_mul(self.key_len))
            .ok_or(AttentionError::WorkspaceOverflow)
    }
}

fn validate_view<T, const N: usize>(
    operand: AttentionOperand,
    view: &ArrayView<'_, T, N>,
) -> AttentionResult<()> {
    view.layout()
        .validate_storage_len(view.data().len())
        .map_err(|source| AttentionError::Layout { operand, source })
}

fn validate_mut_view<T>(
    operand: AttentionOperand,
    view: &mut ArrayViewMut<'_, T, 3>,
) -> AttentionResult<()> {
    view.layout()
        .validate_storage_len(view.data().len())
        .map_err(|source| AttentionError::Layout { operand, source })?;
    if !view
        .layout()
        .is_injective()
        .map_err(|source| AttentionError::Layout { operand, source })?
    {
        return Err(AttentionError::Layout {
            operand,
            source: LetoError::StorageError {
                reason: "attention mutable view contains aliased logical offsets".to_string(),
            },
        });
    }
    Ok(())
}

fn validate_finite<T: RealScalar, const N: usize>(
    operand: AttentionOperand,
    view: &ArrayView<'_, T, N>,
) -> AttentionResult<()> {
    if view
        .iter()
        .copied()
        .any(|value| !NumericElement::is_finite(value))
    {
        return Err(AttentionError::NonFinite { operand });
    }
    Ok(())
}

fn expect_shape(
    operand: AttentionOperand,
    actual: [usize; 3],
    expected: [usize; 3],
) -> AttentionResult<()> {
    if actual != expected {
        return Err(AttentionError::Shape {
            operand,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_mask<T: RealScalar>(
    mask: AttentionMask<'_, T>,
    target: [usize; 3],
) -> AttentionResult<()> {
    let Some(mask_view) = mask.view() else {
        return Ok(());
    };
    validate_view(AttentionOperand::Mask, &mask_view)?;
    validate_finite(AttentionOperand::Mask, &mask_view)?;
    let actual = mask_view.shape();
    if actual
        .iter()
        .zip(target)
        .any(|(&dimension, target_dimension)| dimension != 1 && dimension != target_dimension)
    {
        return Err(AttentionError::MaskShape { actual, target });
    }
    Ok(())
}

fn read<T: Copy>(view: &ArrayView<'_, T, 3>, index: [usize; 3]) -> T {
    *view
        .get(index)
        .expect("invariant: validated attention input index is in bounds")
}

pub(super) fn mask_is_active<T: RealScalar>(
    mask: AttentionMask<'_, T>,
    batch: usize,
    query: usize,
    key: usize,
) -> bool {
    if mask.is_causal() && key > query {
        return false;
    }
    let Some(mask_view) = mask.view() else {
        return true;
    };
    let shape = mask_view.shape();
    let index = [
        if shape[0] == 1 { 0 } else { batch },
        if shape[1] == 1 { 0 } else { query },
        if shape[2] == 1 { 0 } else { key },
    ];
    read(&mask_view, index) != <T as NumericElement>::ZERO
}

fn validate_forward_arithmetic<T: RealScalar + RealField>(
    query: &ArrayView<'_, T, 3>,
    key: &ArrayView<'_, T, 3>,
    mask: AttentionMask<'_, T>,
    scale: T,
    plan: AttentionPlan,
) -> AttentionResult<()> {
    for batch in 0..plan.batch {
        for query_index in 0..plan.query_len {
            let mut active_keys = 0usize;
            for key_index in 0..plan.key_len {
                if !mask_is_active(mask, batch, query_index, key_index) {
                    continue;
                }
                active_keys = active_keys
                    .checked_add(1)
                    .ok_or(AttentionError::WorkspaceOverflow)?;
                let mut score = <T as NumericElement>::ZERO;
                for feature in 0..plan.key_dim {
                    score += read(query, [batch, query_index, feature])
                        * read(key, [batch, key_index, feature]);
                }
                score *= scale;
                if !NumericElement::is_finite(score) {
                    return Err(AttentionError::ArithmeticNonFinite {
                        operand: AttentionOperand::Weights,
                    });
                }
            }
            if active_keys > 0 {
                let support_bound = active_keys
                    .checked_next_power_of_two()
                    .ok_or(AttentionError::WorkspaceOverflow)?;
                if !NumericElement::is_finite(T::from_usize(support_bound)) {
                    return Err(AttentionError::ArithmeticNonFinite {
                        operand: AttentionOperand::Weights,
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_probability_weights<T: RealScalar + RealField>(
    weights: &ArrayView<'_, T, 3>,
    plan: AttentionPlan,
) -> AttentionResult<()> {
    let width = T::from_usize(plan.key_len);
    let tolerance = <T as RealField>::EPSILON * width * T::from_usize(4);
    if !NumericElement::is_finite(tolerance) || tolerance >= T::from_f64(0.5) {
        return Err(AttentionError::ArithmeticNonFinite {
            operand: AttentionOperand::Weights,
        });
    }
    for batch in 0..plan.batch {
        for query in 0..plan.query_len {
            let mut sum = <T as NumericElement>::ZERO;
            for key in 0..plan.key_len {
                let weight = read(weights, [batch, query, key]);
                if weight < <T as NumericElement>::ZERO || weight > <T as NumericElement>::ONE {
                    return Err(AttentionError::InvalidWeights { batch, query });
                }
                sum += weight;
            }
            if sum != <T as NumericElement>::ZERO
                && NumericElement::abs(sum - <T as NumericElement>::ONE) > tolerance
            {
                return Err(AttentionError::InvalidWeights { batch, query });
            }
        }
    }
    Ok(())
}

pub(super) fn validate_forward<T: RealScalar + RealField>(
    query: &ArrayView<'_, T, 3>,
    key: &ArrayView<'_, T, 3>,
    value: &ArrayView<'_, T, 3>,
    mask: AttentionMask<'_, T>,
    scale: T,
    output: &mut ArrayViewMut<'_, T, 3>,
    weights: &mut ArrayViewMut<'_, T, 3>,
) -> AttentionResult<AttentionPlan> {
    for (operand, view) in [
        (AttentionOperand::Query, query),
        (AttentionOperand::Key, key),
        (AttentionOperand::Value, value),
    ] {
        validate_view(operand, view)?;
        validate_finite(operand, view)?;
    }
    if !NumericElement::is_finite(scale) {
        return Err(AttentionError::NonFinite {
            operand: AttentionOperand::Scale,
        });
    }

    let [batch, query_len, key_dim] = query.shape();
    let [key_batch, key_len, actual_key_dim] = key.shape();
    let [value_batch, value_len, value_dim] = value.shape();
    if key_len == 0 {
        return Err(AttentionError::EmptyKeySequence);
    }
    expect_shape(
        AttentionOperand::Key,
        key.shape(),
        [batch, key_len, key_dim],
    )?;
    expect_shape(
        AttentionOperand::Value,
        value.shape(),
        [batch, key_len, value_dim],
    )?;
    debug_assert_eq!(key_batch, batch);
    debug_assert_eq!(actual_key_dim, key_dim);
    debug_assert_eq!(value_batch, batch);
    debug_assert_eq!(value_len, key_len);

    expect_shape(
        AttentionOperand::Output,
        output.shape(),
        [batch, query_len, value_dim],
    )?;
    expect_shape(
        AttentionOperand::Weights,
        weights.shape(),
        [batch, query_len, key_len],
    )?;
    validate_mut_view(AttentionOperand::Output, output)?;
    validate_mut_view(AttentionOperand::Weights, weights)?;
    validate_mask(mask, [batch, query_len, key_len])?;

    let plan = AttentionPlan {
        batch,
        query_len,
        key_len,
        key_dim,
        value_dim,
    };
    validate_forward_arithmetic(query, key, mask, scale, plan)?;
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_backward<T: RealScalar + RealField>(
    output_gradient: &ArrayView<'_, T, 3>,
    query: &ArrayView<'_, T, 3>,
    key: &ArrayView<'_, T, 3>,
    value: &ArrayView<'_, T, 3>,
    weights: &ArrayView<'_, T, 3>,
    scale: T,
    gradients: &mut AttentionGradients<'_, T>,
) -> AttentionResult<AttentionPlan> {
    if !gradients.has_any() {
        return Err(AttentionError::NoGradientTargets);
    }
    for (operand, view) in [
        (AttentionOperand::OutputGradient, output_gradient),
        (AttentionOperand::Query, query),
        (AttentionOperand::Key, key),
        (AttentionOperand::Value, value),
        (AttentionOperand::Weights, weights),
    ] {
        validate_view(operand, view)?;
        validate_finite(operand, view)?;
    }
    if !NumericElement::is_finite(scale) {
        return Err(AttentionError::NonFinite {
            operand: AttentionOperand::Scale,
        });
    }

    let [batch, query_len, key_dim] = query.shape();
    let [_, key_len, _] = key.shape();
    let [_, _, value_dim] = value.shape();
    if key_len == 0 {
        return Err(AttentionError::EmptyKeySequence);
    }
    expect_shape(
        AttentionOperand::Key,
        key.shape(),
        [batch, key_len, key_dim],
    )?;
    expect_shape(
        AttentionOperand::Value,
        value.shape(),
        [batch, key_len, value_dim],
    )?;
    expect_shape(
        AttentionOperand::Weights,
        weights.shape(),
        [batch, query_len, key_len],
    )?;
    expect_shape(
        AttentionOperand::OutputGradient,
        output_gradient.shape(),
        [batch, query_len, value_dim],
    )?;

    for (operand, view, expected) in [
        (
            AttentionOperand::QueryGradient,
            gradients.query.as_mut(),
            [batch, query_len, key_dim],
        ),
        (
            AttentionOperand::KeyGradient,
            gradients.key.as_mut(),
            [batch, key_len, key_dim],
        ),
        (
            AttentionOperand::ValueGradient,
            gradients.value.as_mut(),
            [batch, key_len, value_dim],
        ),
    ] {
        if let Some(view) = view {
            expect_shape(operand, view.shape(), expected)?;
            validate_mut_view(operand, view)?;
            if view
                .as_view()
                .iter()
                .copied()
                .any(|value| !NumericElement::is_finite(value))
            {
                return Err(AttentionError::NonFinite { operand });
            }
        }
    }

    let plan = AttentionPlan {
        batch,
        query_len,
        key_len,
        key_dim,
        value_dim,
    };
    if gradients.query.is_some() || gradients.key.is_some() {
        plan.score_elements()?;
    }
    validate_probability_weights(weights, plan)?;
    Ok(plan)
}
