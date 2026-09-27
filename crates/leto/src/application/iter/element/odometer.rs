//! Row-major offset stepping and the mutable-aliasing proof shared by every
//! iterator in this family.

use crate::domain::layout::Layout;

#[inline]
pub(super) fn odometer_step<const N: usize>(
    index: &mut [usize; N],
    shape: &[usize; N],
    strides: &[isize; N],
    offset: &mut usize,
) {
    for i in (0..N).rev() {
        index[i] += 1;
        if index[i] < shape[i] {
            *offset = (*offset as isize + strides[i]) as usize;
            break;
        }
        *offset = (*offset as isize - (shape[i] - 1) as isize * strides[i]) as usize;
        index[i] = 0;
    }
}

#[inline]
pub(super) fn odometer_step_back<const N: usize>(
    index: &mut [usize; N],
    shape: &[usize; N],
    strides: &[isize; N],
    offset: &mut usize,
) {
    for i in (0..N).rev() {
        if index[i] > 0 {
            index[i] -= 1;
            *offset = (*offset as isize - strides[i]) as usize;
            break;
        }
        *offset = (*offset as isize + (shape[i] - 1) as isize * strides[i]) as usize;
        index[i] = shape[i] - 1;
    }
}

pub(super) fn layout_may_alias_mutable_offsets<const N: usize>(
    layout: &Layout<N>,
) -> crate::domain::error::Result<bool> {
    Ok(!layout.is_injective()?)
}

#[inline]
pub(super) fn last_logical_cursor<const N: usize>(
    layout: &Layout<N>,
    logical_len: usize,
) -> ([usize; N], usize) {
    if logical_len == 0 {
        return ([0usize; N], layout.offset());
    }

    let mut idx = [0usize; N];
    for (axis, item) in idx.iter_mut().enumerate() {
        *item = layout.shape()[axis] - 1;
    }
    let offset = layout
        .offset_of(idx)
        .expect("invariant: last index is valid");
    (idx, offset)
}

/// Private proof token for one bounds-valid, injective mutable layout.
///
/// The token is created once at the mutable partition boundary. It is not
/// exposed as a view or slice, so partition construction cannot duplicate the
/// raw layout proof outside this module family.
pub(super) struct MutableLayoutProof<'a, T, const N: usize> {
    pub(super) ptr: std::ptr::NonNull<T>,
    pub(super) layout: Layout<N>,
    pub(super) storage_len: usize,
    pub(super) _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize> MutableLayoutProof<'a, T, N> {
    pub(super) fn new(
        view: crate::application::view::ArrayViewMut<'a, T, N>,
        aliasing_reason: &'static str,
    ) -> crate::domain::error::Result<Self> {
        view.layout.validate_storage_len(view.len)?;
        if layout_may_alias_mutable_offsets(&view.layout)? {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: aliasing_reason.to_string(),
            });
        }
        Ok(Self {
            ptr: view.ptr,
            layout: view.layout,
            storage_len: view.len,
            _marker: std::marker::PhantomData,
        })
    }
}
