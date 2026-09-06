//! Exact-width complex register tiles shared by matrix permutations.

use eunomia::{
    layout::{cast_slice, cast_slice_mut},
    Pod,
};
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel};
use leto::Complex;

/// Largest complex-square side admitted by the hardware dispatchers.
const MAX_COMPLEX_TILE_SIDE: usize = 8;

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
    const { assert!(SIDE >= 2 && SIDE <= MAX_COMPLEX_TILE_SIDE) };
    assert_eq!(SIDE, ComplexReg::<T, A>::COMPLEX_COUNT);
    let load_row = |row| {
        let start = offset + row * stride;
        let Some(row) = matrix.get(start..start + SIDE) else {
            tile_row_out_of_bounds(start, SIDE, matrix.len());
        };
        let scalars: &[T] = cast_slice(row);
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
    // row loads in the kernel; only the first row is read before the fill.
    let mut tile = [load_row(0); SIDE];
    for (row, register) in tile[1..].iter_mut().enumerate() {
        *register = load_row(row + 1);
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
    const { assert!(SIDE >= 2 && SIDE <= MAX_COMPLEX_TILE_SIDE) };
    assert_eq!(SIDE, ComplexReg::<T, A>::COMPLEX_COUNT);
    for (row, register) in tile.iter().copied().enumerate() {
        let start = offset + row * stride;
        let len = matrix.len();
        let Some(row) = matrix.get_mut(start..start + SIDE) else {
            tile_row_out_of_bounds(start, SIDE, len);
        };
        let scalars: &mut [T] = cast_slice_mut(row);
        let view = simd.view_mut(scalars);
        let mut chunk = view
            .simd_chunks_mut()
            .next()
            .expect("invariant: complex tile row fills one register");
        chunk.store(register.into_interleaved());
    }
}

// One failure site keeps panic-location data outside scalar/ISA instantiations.
// Caller locations must not be forwarded into this cold diagnostic boundary.
#[cold]
#[inline(never)]
fn tile_row_out_of_bounds(start: usize, width: usize, len: usize) -> ! {
    panic!(
        "invariant: complete complex tile row fits matrix storage: start={start}, width={width}, len={len}"
    );
}
