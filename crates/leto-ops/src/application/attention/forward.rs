use super::validation::{mask_is_active, validate_forward};
use super::{AttentionMask, AttentionResult};
use crate::domain::real::RealScalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView, ArrayViewMut};

#[inline(always)]
fn read<T: Copy>(view: &ArrayView<'_, T, 3>, index: [usize; 3]) -> T {
    *view
        .get(index)
        .expect("invariant: validated attention input index is in bounds")
}

#[inline(always)]
fn write<T: Copy>(view: &mut ArrayViewMut<'_, T, 3>, index: [usize; 3], value: T) {
    *view
        .get_mut(index)
        .expect("invariant: validated attention output index is in bounds") = value;
}

#[inline(always)]
fn convex_combine<T: RealScalar>(current: T, value: T, value_weight: T) -> T {
    let one = <T as NumericElement>::ONE;
    let zero = <T as NumericElement>::ZERO;
    if (current >= zero) == (value >= zero) {
        if current <= value {
            current + value_weight * (value - current)
        } else {
            value + (one - value_weight) * (current - value)
        }
    } else {
        (one - value_weight) * current + value_weight * value
    }
}

/// Computes scaled dot-product attention into caller-owned views.
///
/// Query, key, and value use `[batch, sequence, feature]` order. `weights`
/// receives the post-softmax matrix `[batch, query_sequence, key_sequence]`
/// required by backward. All views may be strided or offset; mask dimensions
/// equal to one broadcast without materialization. A fully masked row produces
/// zero weights and a zero output row.
///
/// # Errors
/// Returns [`super::AttentionError`] when a layout, shape, finite-value, mask,
/// or workspace contract is invalid. Validation completes before either output
/// is mutated.
///
/// # Examples
/// ```
/// use leto::{Array, Layout, Storage, VecStorage};
/// use leto_ops::{scaled_dot_product_attention_into, AttentionMask};
///
/// let tensor = Array::new(
///     Layout::c_contiguous([1, 1, 1])?,
///     VecStorage::new(vec![2.0_f32]),
/// )?;
/// let mut output = Array::new(
///     Layout::c_contiguous([1, 1, 1])?,
///     VecStorage::new(vec![0.0_f32]),
/// )?;
/// let mut weights = output.clone();
/// scaled_dot_product_attention_into(
///     &tensor.view(),
///     &tensor.view(),
///     &tensor.view(),
///     AttentionMask::Unmasked,
///     1.0,
///     &mut output.view_mut(),
///     &mut weights.view_mut(),
/// )?;
/// assert_eq!(output.storage().as_slice(), &[2.0]);
/// assert_eq!(weights.storage().as_slice(), &[1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_into<T: RealScalar + RealField>(
    query: &ArrayView<'_, T, 3>,
    key: &ArrayView<'_, T, 3>,
    value: &ArrayView<'_, T, 3>,
    mask: AttentionMask<'_, T>,
    scale: T,
    output: &mut ArrayViewMut<'_, T, 3>,
    weights: &mut ArrayViewMut<'_, T, 3>,
) -> AttentionResult<()> {
    let plan = validate_forward(query, key, value, mask, scale, output, weights)?;

    for batch in 0..plan.batch {
        for query_index in 0..plan.query_len {
            let mut row_max = None;
            for key_index in 0..plan.key_len {
                let weight_index = [batch, query_index, key_index];
                if !mask_is_active(mask, batch, query_index, key_index) {
                    write(weights, weight_index, <T as NumericElement>::ZERO);
                    continue;
                }
                let mut score = <T as NumericElement>::ZERO;
                for feature in 0..plan.key_dim {
                    score += read(query, [batch, query_index, feature])
                        * read(key, [batch, key_index, feature]);
                }
                score *= scale;
                write(weights, weight_index, score);
                row_max = Some(match row_max {
                    Some(current) if current > score => current,
                    _ => score,
                });
            }

            let Some(row_max) = row_max else {
                for value_index in 0..plan.value_dim {
                    write(
                        output,
                        [batch, query_index, value_index],
                        <T as NumericElement>::ZERO,
                    );
                }
                continue;
            };

            let mut row_sum = <T as NumericElement>::ZERO;
            for key_index in 0..plan.key_len {
                if !mask_is_active(mask, batch, query_index, key_index) {
                    continue;
                }
                let index = [batch, query_index, key_index];
                let probability = FloatElement::exp(
                    *weights
                        .get(index)
                        .expect("invariant: validated weight index is in bounds")
                        - row_max,
                );
                write(weights, index, probability);
                row_sum += probability;
            }
            let inverse_sum = <T as NumericElement>::ONE / row_sum;
            for key_index in 0..plan.key_len {
                let index = [batch, query_index, key_index];
                let probability = *weights
                    .get(index)
                    .expect("invariant: validated weight index is in bounds")
                    * inverse_sum;
                write(weights, index, probability);
            }

            for value_index in 0..plan.value_dim {
                let mut result = <T as NumericElement>::ZERO;
                let mut total_weight = <T as NumericElement>::ZERO;
                for key_index in 0..plan.key_len {
                    let weight = *weights
                        .get([batch, query_index, key_index])
                        .expect("invariant: validated weight index is in bounds");
                    if weight == <T as NumericElement>::ZERO {
                        continue;
                    }
                    let next_total = total_weight + weight;
                    result = convex_combine(
                        result,
                        read(value, [batch, key_index, value_index]),
                        weight / next_total,
                    );
                    total_weight = next_total;
                }
                write(output, [batch, query_index, value_index], result);
            }
        }
    }
    Ok(())
}
