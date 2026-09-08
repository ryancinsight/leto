//! In-place square permutation with bounded register storage.

use super::tile::{load_tile, store_tile};
use eunomia::Pod;
use hermes_simd::{
    vectorize_hardware_lanes, ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel,
};
use leto::Complex;

use crate::domain::layout::SquareTransposeError;

// Preserve Apollo's baseline 16-by-16 cache blocks around the register tiles.
// Two blocks contain 2 * 16^2 * size_of::<Complex<T>>() payload bytes: at most
// 8 KiB for the four supported scalars. This bounds payload, not cache residency.
const CACHE_BLOCK_SIDE: usize = 16;

pub(super) fn transpose_square_inplace<T>(
    matrix: &mut [Complex<T>],
    side: usize,
) -> Result<(), SquareTransposeError>
where
    T: LaneScalar + Pod,
{
    let Some(expected) = side.checked_mul(side) else {
        return Err(SquareTransposeError::Overflow { side });
    };
    if matrix.len() != expected {
        return Err(SquareTransposeError::Length {
            side,
            expected,
            actual: matrix.len(),
        });
    }
    if expected == 0 {
        return Ok(());
    }

    let full_side = 'hardware: {
        if side >= 8 {
            if let Some(full_side) = vectorize_hardware_lanes::<16, T, _>(SquareTransposeKernel {
                matrix: &mut *matrix,
                side,
            }) {
                break 'hardware full_side;
            }
        }
        if side >= 4 {
            if let Some(full_side) = vectorize_hardware_lanes::<8, T, _>(SquareTransposeKernel {
                matrix: &mut *matrix,
                side,
            }) {
                break 'hardware full_side;
            }
        }
        if side >= 2 {
            if let Some(full_side) = vectorize_hardware_lanes::<4, T, _>(SquareTransposeKernel {
                matrix: &mut *matrix,
                side,
            }) {
                break 'hardware full_side;
            }
        }
        0
    };
    if full_side < side {
        transpose_tail(matrix, side, full_side);
    }
    Ok(())
}

struct SquareTransposeKernel<'a, T> {
    matrix: &'a mut [Complex<T>],
    side: usize,
}

impl<T> LaneKernel<T> for SquareTransposeKernel<'_, T>
where
    T: LaneScalar + Pod,
{
    type Output = usize;

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> usize {
        match ComplexReg::<T, A>::COMPLEX_COUNT {
            2 => transpose_tiled::<T, A, 2>(self.matrix, self.side, &simd),
            4 => transpose_tiled::<T, A, 4>(self.matrix, self.side, &simd),
            8 => transpose_tiled::<T, A, 8>(self.matrix, self.side, &simd),
            _ => {
                unreachable!("invariant: requested hardware widths hold 2, 4 or 8 complex samples")
            }
        }
    }
}

#[inline(always)]
fn transpose_tiled<T, A, const SIDE: usize>(
    matrix: &mut [Complex<T>],
    side: usize,
    simd: &Simd<T, A>,
) -> usize
where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    let full_side = side / SIDE * SIDE;
    for row in (0..full_side).step_by(SIDE) {
        let diagonal = row * side + row;
        let tile = load_tile::<T, A, SIDE>(simd, matrix, side, diagonal);
        store_tile(simd, matrix, side, diagonal, &tile);
    }

    // SIDE is 2, 4 or 8, so every block boundary and clipped full-side boundary
    // is a register-tile boundary. The strict upper triangle excludes the
    // diagonal tiles above and assigns every remaining tile pair exactly once.
    for block_row in (0..full_side).step_by(CACHE_BLOCK_SIDE) {
        let row_end = (block_row + CACHE_BLOCK_SIDE).min(full_side);
        for block_column in (block_row..full_side).step_by(CACHE_BLOCK_SIDE) {
            let column_end = (block_column + CACHE_BLOCK_SIDE).min(full_side);
            for row in (block_row..row_end).step_by(SIDE) {
                for column in (block_column.max(row + SIDE)..column_end).step_by(SIDE) {
                    let upper = row * side + column;
                    let lower = column * side + row;
                    // Both source tiles become owned register values before
                    // either destination is overwritten.
                    let upper_tile = load_tile::<T, A, SIDE>(simd, matrix, side, upper);
                    let lower_tile = load_tile::<T, A, SIDE>(simd, matrix, side, lower);
                    store_tile(simd, matrix, side, lower, &upper_tile);
                    store_tile(simd, matrix, side, upper, &lower_tile);
                }
            }
        }
    }

    full_side
}

fn transpose_tail<T>(matrix: &mut [T], side: usize, full_side: usize) {
    // Outside the complete square, the larger coordinate is at least
    // full_side. Visiting only row < column covers that border exactly once,
    // including its bottom-right partial square; diagonal values stay put.
    for row in 0..side {
        for column in (row + 1).max(full_side)..side {
            matrix.swap(row * side + column, column * side + row);
        }
    }
}

#[cfg(test)]
mod tests;
