use super::CrossEntropyResult;
use super::validation::validate_backward;
use crate::domain::real::RealScalar;
use eunomia::{NumericElement, RealField};
use leto::{ArrayView, ArrayViewMut};

/// Adds the mean cross-entropy logit gradient into caller-owned storage.
///
/// `probabilities` is the provider-resident `[batch, classes]` output retained
/// by [`super::cross_entropy_forward_into`]. The upstream scalar scales the
/// complete mean-reduced gradient. The destination may be strided or offset.
///
/// # Errors
/// Returns [`super::CrossEntropyError`] for invalid layouts, shapes, empty
/// dimensions, target indices, or scalar-representation bounds. Validation
/// completes before the destination changes.
pub fn cross_entropy_backward_accumulate<T: RealScalar + RealField>(
    output_gradient: &ArrayView<'_, T, 1>,
    probabilities: &ArrayView<'_, T, 2>,
    targets: &[usize],
    logit_gradient: &mut ArrayViewMut<'_, T, 2>,
) -> CrossEntropyResult<()> {
    let plan = validate_backward(output_gradient, probabilities, targets, logit_gradient)?;
    let upstream = *output_gradient
        .get([0])
        .expect("invariant: validated output-gradient index is in bounds");
    let scale = upstream * plan.inverse_batch;

    for (batch, &target) in targets.iter().enumerate() {
        for class in 0..plan.classes {
            let probability = *probabilities
                .get([batch, class])
                .expect("invariant: validated probability index is in bounds");
            let indicator = if class == target {
                <T as NumericElement>::ONE
            } else {
                <T as NumericElement>::ZERO
            };
            let destination = logit_gradient
                .get_mut([batch, class])
                .expect("invariant: validated logit-gradient index is in bounds");
            *destination += (probability - indicator) * scale;
        }
    }
    Ok(())
}
