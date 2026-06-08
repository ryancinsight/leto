/// Convert a flat row-major logical index into an N-dimensional index.
#[inline(always)]
pub(crate) fn index_from_flat<const N: usize>(flat: usize, shape: &[usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    let mut temp = flat;
    for axis in (0..N).rev() {
        if shape[axis] > 0 {
            index[axis] = temp % shape[axis];
            temp /= shape[axis];
        }
    }
    index
}
