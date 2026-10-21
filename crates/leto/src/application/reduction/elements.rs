//! Shared element traversal for the reduction kernels.

use crate::application::view::ArrayView;

/// Iterate every logical element of `view` in row-major index order.
///
/// Fast-paths to a direct slice scan for C-contiguous layouts are not needed
/// here because `index_from_flat` + `offset_of` is branch-free per element.
/// The function is `pub(crate)` so leaf modules can import it without
/// re-implementing the contiguity check.
#[inline]
pub(crate) fn iter_elements<'a, T, const N: usize>(
    view: &'a ArrayView<'a, T, N>,
) -> impl Iterator<Item = &'a T> + 'a {
    view.iter()
}
