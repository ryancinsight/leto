//! Exact-width complex register tiles shared by matrix permutations.

use eunomia::{
    layout::{cast_slice, cast_slice_mut},
    Pod,
};
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel};
use leto::Complex;

/// Largest complex-square side admitted by the hardware dispatchers.
const MAX_COMPLEX_TILE_SIDE: usize = 8;

/// Complete strided rows followed by the final, unpadded register row.
#[inline(always)]
fn tile_extent<const SIDE: usize>(stride: usize, offset: usize) -> (usize, usize) {
    const { assert!(SIDE >= 2 && SIDE <= MAX_COMPLEX_TILE_SIDE) };
    assert!(
        stride >= SIDE,
        "invariant: complex tile rows do not overlap"
    );
    let head = (SIDE - 1)
        .checked_mul(stride)
        .expect("invariant: complex tile row span fits usize");
    let end = offset
        .checked_add(head)
        .and_then(|last| last.checked_add(SIDE))
        .expect("invariant: complex tile extent fits usize");
    (head, end)
}

/// Loads and transposes one complete tile without retaining source borrows.
/// The owned registers let an in-place caller read both halves before writing.
#[inline(always)]
pub(super) fn load_tile<T, A, const SIDE: usize>(
    simd: &Simd<T, A>,
    matrix: &[Complex<T>],
    stride: usize,
    offset: usize,
) -> [ComplexReg<T, A>; SIDE]
where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    assert_eq!(SIDE, ComplexReg::<T, A>::COMPLEX_COUNT);
    let (head, end) = tile_extent::<SIDE>(stride, offset);
    let span = matrix
        .get(offset..end)
        .expect("invariant: complex tile fits its matrix");
    let (rows, last) = span.split_at(head);
    let load_row = |row: &[Complex<T>]| {
        let scalars: &[T] = cast_slice(&row[..SIDE]);
        let view = simd.view(scalars);
        // Two scalars per complex sample give exactly one full register,
        // so this iterator has one chunk and no remainder.
        let chunk = view
            .simd_chunks()
            .next()
            .expect("invariant: complex tile row fills one register");
        ComplexReg::from_interleaved(chunk.load())
    };
    // Array construction through `from_fn` outlines across the AVX-512
    // capability boundary on the pinned compiler. Exact Copy storage keeps
    // row loads in the kernel. The final row has exactly SIDE elements;
    // preceding rows have stride elements, with stride >= SIDE. Their length
    // is (SIDE - 1) * stride, so chunks_exact has no remainder and the zip
    // fills every preceding register exactly once.
    let mut tile = [load_row(last); SIDE];
    for (row, register) in rows.chunks_exact(stride).zip(&mut tile[..SIDE - 1]) {
        *register = load_row(row);
    }
    ComplexReg::transpose_square(&mut tile);
    tile
}

/// Stores a transposed tile into complete, disjoint destination row segments.
#[inline(always)]
pub(super) fn store_tile<T, A, const SIDE: usize>(
    simd: &Simd<T, A>,
    matrix: &mut [Complex<T>],
    stride: usize,
    offset: usize,
    tile: &[ComplexReg<T, A>; SIDE],
) where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    assert_eq!(SIDE, ComplexReg::<T, A>::COMPLEX_COUNT);
    let (head, end) = tile_extent::<SIDE>(stride, offset);
    let span = matrix
        .get_mut(offset..end)
        .expect("invariant: complex tile fits its matrix");
    let (rows, last) = span.split_at_mut(head);
    let store_row = |row: &mut [Complex<T>], register: ComplexReg<T, A>| {
        let scalars: &mut [T] = cast_slice_mut(&mut row[..SIDE]);
        let view = simd.view_mut(scalars);
        let mut chunk = view
            .simd_chunks_mut()
            .next()
            .expect("invariant: complex tile row fills one register");
        chunk.store(register.into_interleaved());
    };
    // The same exact span decomposition as load_tile yields SIDE - 1 full
    // strides with no remainder and a disjoint final register row. Padding
    // within each stride is borrowed but never written.
    for (row, register) in rows.chunks_exact_mut(stride).zip(&tile[..SIDE - 1]) {
        store_row(row, *register);
    }
    store_row(last, tile[SIDE - 1]);
}
