use crate::domain::layout::Layout;

/// Computes the physical `[offset, offset + size)` range covered by a layout
/// whose elements form a single dense block. Returns `None` only on size
/// overflow. Shared by the contiguous-slice accessors of both view types.
#[inline]
pub(super) fn dense_block_range<const N: usize>(
    layout: &Layout<N>,
) -> Option<core::ops::Range<usize>> {
    let start = layout.offset();
    let end = start.checked_add(layout.checked_size().ok()?)?;
    Some(start..end)
}

pub(super) const SHARED_WINDOW_ACCESS_MESSAGE: &str =
    "window is shared with sibling lane/axis views; a whole-window \
     slice would alias their elements (use per-element access instead)";

#[inline]
pub(super) fn checked_block_range<const N: usize>(
    layout: &Layout<N>,
    len: usize,
) -> Option<core::ops::Range<usize>> {
    let range = dense_block_range(layout)?;
    (range.end <= len).then_some(range)
}

#[inline]
pub(super) fn assert_exclusive_window(window_shared: bool) {
    assert!(!window_shared, "{SHARED_WINDOW_ACCESS_MESSAGE}");
}

#[inline]
pub(super) unsafe fn raw_slice_from_ptr<'a, T>(ptr: std::ptr::NonNull<T>, len: usize) -> &'a [T] {
    // SAFETY: callers guarantee `ptr` is valid for `len` elements.
    unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) }
}

#[inline]
pub(super) unsafe fn raw_slice_from_ptr_mut<'a, T>(
    ptr: std::ptr::NonNull<T>,
    len: usize,
) -> &'a mut [T] {
    // SAFETY: callers guarantee `ptr` is valid for `len` elements and uniquely borrowed.
    unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) }
}

#[inline]
pub(super) unsafe fn raw_range_from_ptr<'a, T>(
    ptr: std::ptr::NonNull<T>,
    range: core::ops::Range<usize>,
) -> &'a [T] {
    // SAFETY: callers guarantee the sub-range is in bounds for `ptr`.
    unsafe { std::slice::from_raw_parts(ptr.as_ptr().add(range.start), range.len()) }
}

#[inline]
pub(super) unsafe fn raw_range_from_ptr_mut<'a, T>(
    ptr: std::ptr::NonNull<T>,
    range: core::ops::Range<usize>,
) -> &'a mut [T] {
    // SAFETY: callers guarantee the sub-range is in bounds for `ptr` and uniquely borrowed.
    unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().add(range.start), range.len()) }
}

#[inline]
pub(super) unsafe fn raw_ref_from_ptr<'a, T>(ptr: std::ptr::NonNull<T>, offset: usize) -> &'a T {
    // SAFETY: callers guarantee `offset` is in bounds for `ptr`.
    unsafe { &*ptr.as_ptr().add(offset) }
}

#[inline]
pub(super) unsafe fn raw_mut_from_ptr<'a, T>(
    ptr: std::ptr::NonNull<T>,
    offset: usize,
) -> &'a mut T {
    // SAFETY: callers guarantee `offset` is in bounds for `ptr` and uniquely borrowed.
    unsafe { &mut *ptr.as_ptr().add(offset) }
}
