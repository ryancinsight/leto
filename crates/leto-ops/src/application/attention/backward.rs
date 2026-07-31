use super::validation::validate_backward;
use super::{AttentionGradients, AttentionResult};
use crate::domain::real::RealScalar;
use eunomia::{NumericElement, RealField};
use leto::{ArrayView, ArrayViewMut};

#[inline(always)]
fn read<T: Copy>(view: &ArrayView<'_, T, 3>, index: [usize; 3]) -> T {
    *view
        .get(index)
        .expect("invariant: validated attention input index is in bounds")
}

#[inline(always)]
fn accumulate<T: Copy + core::ops::AddAssign>(
    view: &mut ArrayViewMut<'_, T, 3>,
    index: [usize; 3],
    value: T,
) {
    *view
        .get_mut(index)
        .expect("invariant: validated attention gradient index is in bounds") += value;
}

#[inline]
fn score_index(query_len: usize, key_len: usize, batch: usize, query: usize, key: usize) -> usize {
    (batch * query_len + query) * key_len + key
}

fn query_increment<T: RealScalar>(
    key: &ArrayView<'_, T, 3>,
    score_gradients: &[T],
    dimensions: (usize, usize),
    index: [usize; 3],
    scale: T,
) -> T {
    let [batch, query, feature] = index;
    let (query_len, key_len) = dimensions;
    let mut gradient = <T as NumericElement>::ZERO;
    for key_index in 0..key_len {
        gradient += score_gradients[score_index(query_len, key_len, batch, query, key_index)]
            * read(key, [batch, key_index, feature]);
    }
    gradient * scale
}

fn key_increment<T: RealScalar>(
    query: &ArrayView<'_, T, 3>,
    score_gradients: &[T],
    dimensions: (usize, usize),
    index: [usize; 3],
    scale: T,
) -> T {
    let [batch, key, feature] = index;
    let (query_len, key_len) = dimensions;
    let mut gradient = <T as NumericElement>::ZERO;
    for query_index in 0..query_len {
        gradient += score_gradients[score_index(query_len, key_len, batch, query_index, key)]
            * read(query, [batch, query_index, feature]);
    }
    gradient * scale
}

fn value_increment<T: RealScalar>(
    output_gradient: &ArrayView<'_, T, 3>,
    weights: &ArrayView<'_, T, 3>,
    query_len: usize,
    index: [usize; 3],
) -> T {
    let [batch, key, value] = index;
    let mut gradient = <T as NumericElement>::ZERO;
    for query in 0..query_len {
        gradient +=
            read(weights, [batch, query, key]) * read(output_gradient, [batch, query, value]);
    }
    gradient
}

fn validate_accumulation<T: RealScalar>(
    destination: &ArrayViewMut<'_, T, 3>,
    index: [usize; 3],
    increment: T,
    operand: super::AttentionOperand,
) -> AttentionResult<()> {
    let current = *destination
        .as_view()
        .get(index)
        .expect("invariant: validated attention gradient index is in bounds");
    if !NumericElement::is_finite(increment) || !NumericElement::is_finite(current + increment) {
        return Err(super::AttentionError::ArithmeticNonFinite { operand });
    }
    Ok(())
}

/// Accumulates scaled dot-product attention gradients into selected targets.
///
/// `weights` must be the post-softmax matrix returned by
/// [`super::scaled_dot_product_attention_into`]. Every supplied gradient is
/// incremented rather than overwritten. Query or key gradients use one dense
/// score-gradient workspace with `batch * query_sequence * key_sequence`
/// elements; value-only backward allocates no workspace.
///
/// # Errors
/// Returns [`super::AttentionError`] when no gradient is requested or any
/// layout, shape, finite-value, or workspace contract is invalid. Validation
/// completes before a gradient destination is mutated.
///
/// # Examples
/// ```
/// use leto::{Array, Layout, Storage, VecStorage};
/// use leto_ops::{
///     scaled_dot_product_attention_backward_accumulate, AttentionGradients,
/// };
///
/// let layout = Layout::c_contiguous([1, 1, 1])?;
/// let tensor = Array::new(layout, VecStorage::new(vec![2.0_f32]))?;
/// let weights = Array::new(layout, VecStorage::new(vec![1.0_f32]))?;
/// let output_gradient = Array::new(layout, VecStorage::new(vec![3.0_f32]))?;
/// let mut value_gradient = Array::new(layout, VecStorage::new(vec![4.0_f32]))?;
/// scaled_dot_product_attention_backward_accumulate(
///     &output_gradient.view(),
///     &tensor.view(),
///     &tensor.view(),
///     &tensor.view(),
///     &weights.view(),
///     1.0,
///     AttentionGradients::new(None, None, Some(value_gradient.view_mut())),
/// )?;
/// assert_eq!(value_gradient.storage().as_slice(), &[7.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_backward_accumulate<T: RealScalar + RealField>(
    output_gradient: &ArrayView<'_, T, 3>,
    query: &ArrayView<'_, T, 3>,
    key: &ArrayView<'_, T, 3>,
    value: &ArrayView<'_, T, 3>,
    weights: &ArrayView<'_, T, 3>,
    scale: T,
    mut gradients: AttentionGradients<'_, T>,
) -> AttentionResult<()> {
    let plan = validate_backward(
        output_gradient,
        query,
        key,
        value,
        weights,
        scale,
        &mut gradients,
    )?;
    let needs_score_gradients = gradients.query.is_some() || gradients.key.is_some();
    let score_gradients = if needs_score_gradients {
        let mut workspace = vec![<T as NumericElement>::ZERO; plan.score_elements()?];
        for batch in 0..plan.batch {
            for query_index in 0..plan.query_len {
                for key_index in 0..plan.key_len {
                    let mut attention_gradient = <T as NumericElement>::ZERO;
                    for value_index in 0..plan.value_dim {
                        attention_gradient +=
                            read(output_gradient, [batch, query_index, value_index])
                                * read(value, [batch, key_index, value_index]);
                    }
                    if !NumericElement::is_finite(attention_gradient) {
                        return Err(super::AttentionError::ArithmeticNonFinite {
                            operand: super::AttentionOperand::Weights,
                        });
                    }
                    workspace[score_index(
                        plan.query_len,
                        plan.key_len,
                        batch,
                        query_index,
                        key_index,
                    )] = attention_gradient;
                }

                let mut row_projection = <T as NumericElement>::ZERO;
                for key_index in 0..plan.key_len {
                    row_projection += read(weights, [batch, query_index, key_index])
                        * workspace[score_index(
                            plan.query_len,
                            plan.key_len,
                            batch,
                            query_index,
                            key_index,
                        )];
                }
                if !NumericElement::is_finite(row_projection) {
                    return Err(super::AttentionError::ArithmeticNonFinite {
                        operand: super::AttentionOperand::Weights,
                    });
                }
                for key_index in 0..plan.key_len {
                    let index =
                        score_index(plan.query_len, plan.key_len, batch, query_index, key_index);
                    workspace[index] = read(weights, [batch, query_index, key_index])
                        * (workspace[index] - row_projection);
                    if !NumericElement::is_finite(workspace[index]) {
                        return Err(super::AttentionError::ArithmeticNonFinite {
                            operand: super::AttentionOperand::Weights,
                        });
                    }
                }
            }
        }
        Some(workspace)
    } else {
        None
    };

    let dimensions = (plan.query_len, plan.key_len);
    if let Some(query_gradient) = gradients.query.as_ref() {
        let workspace = score_gradients
            .as_deref()
            .expect("invariant: query gradient requires score workspace");
        for batch in 0..plan.batch {
            for query_index in 0..plan.query_len {
                for feature in 0..plan.key_dim {
                    let index = [batch, query_index, feature];
                    validate_accumulation(
                        query_gradient,
                        index,
                        query_increment(key, workspace, dimensions, index, scale),
                        super::AttentionOperand::QueryGradient,
                    )?;
                }
            }
        }
    }
    if let Some(key_gradient) = gradients.key.as_ref() {
        let workspace = score_gradients
            .as_deref()
            .expect("invariant: key gradient requires score workspace");
        for batch in 0..plan.batch {
            for key_index in 0..plan.key_len {
                for feature in 0..plan.key_dim {
                    let index = [batch, key_index, feature];
                    validate_accumulation(
                        key_gradient,
                        index,
                        key_increment(query, workspace, dimensions, index, scale),
                        super::AttentionOperand::KeyGradient,
                    )?;
                }
            }
        }
    }
    if let Some(value_gradient) = gradients.value.as_ref() {
        for batch in 0..plan.batch {
            for key_index in 0..plan.key_len {
                for value_index in 0..plan.value_dim {
                    let index = [batch, key_index, value_index];
                    validate_accumulation(
                        value_gradient,
                        index,
                        value_increment(output_gradient, weights, plan.query_len, index),
                        super::AttentionOperand::ValueGradient,
                    )?;
                }
            }
        }
    }

    if let Some(query_gradient) = gradients.query.as_mut() {
        let workspace = score_gradients
            .as_deref()
            .expect("invariant: query gradient requires score workspace");
        for batch in 0..plan.batch {
            for query_index in 0..plan.query_len {
                for feature in 0..plan.key_dim {
                    let index = [batch, query_index, feature];
                    accumulate(
                        query_gradient,
                        index,
                        query_increment(key, workspace, dimensions, index, scale),
                    );
                }
            }
        }
    }

    if let Some(key_gradient) = gradients.key.as_mut() {
        let workspace = score_gradients
            .as_deref()
            .expect("invariant: key gradient requires score workspace");
        for batch in 0..plan.batch {
            for key_index in 0..plan.key_len {
                for feature in 0..plan.key_dim {
                    let index = [batch, key_index, feature];
                    accumulate(
                        key_gradient,
                        index,
                        key_increment(query, workspace, dimensions, index, scale),
                    );
                }
            }
        }
    }

    if let Some(value_gradient) = gradients.value.as_mut() {
        for batch in 0..plan.batch {
            for key_index in 0..plan.key_len {
                for value_index in 0..plan.value_dim {
                    let index = [batch, key_index, value_index];
                    accumulate(
                        value_gradient,
                        index,
                        value_increment(output_gradient, weights, plan.query_len, index),
                    );
                }
            }
        }
    }

    Ok(())
}
