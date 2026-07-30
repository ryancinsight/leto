#[inline]
pub(super) fn decode_index<const N: usize>(
    mut flat: usize,
    shape: &[usize; N],
    index: &mut [usize; N],
) {
    for axis in (0..N).rev() {
        let extent = shape[axis];
        if extent == 0 {
            index[axis] = 0;
        } else {
            index[axis] = flat % extent;
            flat /= extent;
        }
    }
}
