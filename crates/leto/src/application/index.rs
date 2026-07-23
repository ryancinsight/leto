/// Convert a flat row-major logical index into an N-dimensional index.
#[inline(always)]
pub(crate) fn index_from_flat<const N: usize>(flat: usize, shape: &[usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    crate::domain::layout::kernels::fill_index_from_flat(flat, shape, &mut index);
    index
}
