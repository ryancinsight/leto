use leto::Complex;

use super::payloads::{assert_bits, expected, values, PayloadScalar};

pub(crate) fn assert_squares<T: PayloadScalar>(
    mut transpose: impl FnMut(&mut [Complex<T>], usize),
) {
    // Cover both sides of each supported register/tile boundary.
    for side in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        for offset in 0..4 {
            assert_case::<T>(side, offset, &mut transpose);
        }
    }
    // These are the square factors of the consumer's 65,536/262,144 FFTs.
    for side in [256, 512] {
        assert_case::<T>(side, 1, &mut transpose);
    }
}

fn assert_case<T: PayloadScalar>(
    side: usize,
    offset: usize,
    transpose: &mut impl FnMut(&mut [Complex<T>], usize),
) {
    let len = side * side;
    let mut storage = values::<T>(offset + len + 5);
    let before = storage.clone();
    let end = offset + len;
    let mut oracle = before.clone();
    oracle[offset..end].copy_from_slice(&expected(&before[offset..end], 1, side, side));

    transpose(&mut storage[offset..end], side);
    // The independent coordinate oracle also checks diagonal and guard bytes.
    assert_bits(&storage, &oracle);
    transpose(&mut storage[offset..end], side);
    assert_bits(&storage, &before);
}
