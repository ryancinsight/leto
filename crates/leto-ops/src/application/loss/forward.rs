use super::CrossEntropyResult;
use super::validation::validate_forward;
use crate::domain::real::RealScalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView, ArrayViewMut};

#[inline(always)]
fn read<T: Copy>(view: &ArrayView<'_, T, 2>, index: [usize; 2]) -> T {
    *view
        .get(index)
        .expect("invariant: validated cross-entropy input index is in bounds")
}

#[inline(always)]
fn write<T: Copy>(view: &mut ArrayViewMut<'_, T, 2>, index: [usize; 2], value: T) {
    *view
        .get_mut(index)
        .expect("invariant: validated cross-entropy output index is in bounds") = value;
}

/// Computes stable mean cross-entropy into caller-owned views.
///
/// `logits` and `probabilities` use `[batch, classes]` order. `targets`
/// contains one class index per batch row and `loss` has shape `[1]`. Every
/// view may be strided or offset. Probabilities are retained for backward
/// without an intermediate allocation.
///
/// # Errors
/// Returns [`super::CrossEntropyError`] for invalid layouts, shapes, empty
/// dimensions, target indices, or scalar-representation bounds. Validation
/// completes before either output changes.
///
/// # Examples
/// ```
/// use leto::{Array, Layout, Storage, VecStorage};
/// use leto_ops::cross_entropy_forward_into;
///
/// let logits = Array::new(
///     Layout::c_contiguous([1, 2])?,
///     VecStorage::new(vec![0.0_f32, 0.0]),
/// )?;
/// let mut probabilities = logits.clone();
/// let mut loss = Array::new(
///     Layout::c_contiguous([1])?,
///     VecStorage::new(vec![0.0_f32]),
/// )?;
/// cross_entropy_forward_into(
///     &logits.view(),
///     &[0],
///     &mut loss.view_mut(),
///     &mut probabilities.view_mut(),
/// )?;
/// assert_eq!(probabilities.storage().as_slice(), &[0.5, 0.5]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn cross_entropy_forward_into<T: RealScalar + RealField>(
    logits: &ArrayView<'_, T, 2>,
    targets: &[usize],
    loss: &mut ArrayViewMut<'_, T, 1>,
    probabilities: &mut ArrayViewMut<'_, T, 2>,
) -> CrossEntropyResult<()> {
    let plan = validate_forward(logits, targets, loss, probabilities)?;
    let mut mean_loss = <T as NumericElement>::ZERO;

    for (batch, &target) in targets.iter().enumerate() {
        let mut row_max = read(logits, [batch, 0]);
        for class in 1..plan.classes {
            let value = read(logits, [batch, class]);
            if value > row_max {
                row_max = value;
            }
        }

        let mut row_sum = <T as NumericElement>::ZERO;
        for class in 0..plan.classes {
            let shifted = read(logits, [batch, class]) - row_max;
            let exponential = FloatElement::exp(shifted);
            write(probabilities, [batch, class], exponential);
            row_sum += exponential;
        }
        let inverse_sum = <T as NumericElement>::ONE / row_sum;
        for class in 0..plan.classes {
            let probability = *probabilities
                .get([batch, class])
                .expect("invariant: validated probability index is in bounds")
                * inverse_sum;
            write(probabilities, [batch, class], probability);
        }
        let row_loss = FloatElement::ln(row_sum) + (row_max - read(logits, [batch, target]));
        let row_count = T::from_usize(batch + 1);
        mean_loss += (row_loss - mean_loss) / row_count;
    }

    *loss
        .get_mut([0])
        .expect("invariant: validated scalar loss index is in bounds") = mean_loss;
    Ok(())
}
