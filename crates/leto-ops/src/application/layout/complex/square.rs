//! In-place square permutation with bounded register storage.

use super::tile::{load_tile, store_tile};
use eunomia::Pod;
use hermes_simd::{
    vectorize_hardware_lanes, ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel,
};
use leto::{Complex, LetoError, Result};

/// Transposes a row-major complex square in its existing storage.
///
/// Every sample at `(row, column)` moves to `(column, row)` without arithmetic
/// or allocation. All scalar bits survive, including signed zeros and NaN
/// payloads. Complete hardware tiles exchange through registers; remaining
/// pairs and targets without a suitable hardware width use scalar swaps.
///
/// # Errors
///
/// Returns [`LetoError::Overflow`] if `side * side` cannot be represented, or
/// [`LetoError::StorageError`] if `matrix.len()` differs from that extent.
/// Both errors leave the complete input unchanged. Side zero requires empty
/// storage and performs no work.
///
/// # Examples
///
/// ```
/// use leto::Complex;
/// use leto_ops::transpose_square_inplace;
///
/// let mut matrix = [1.0_f32, 2.0, 3.0, 4.0].map(|re| Complex::new(re, -re));
/// let original = matrix;
/// transpose_square_inplace(&mut matrix, 2)?;
/// assert_eq!(matrix, [original[0], original[2], original[1], original[3]]);
/// # Ok::<(), leto::LetoError>(())
/// ```
pub fn transpose_square_inplace<T>(matrix: &mut [Complex<T>], side: usize) -> Result<()>
where
    T: LaneScalar + Pod,
{
    let Some(expected) = side.checked_mul(side) else {
        return Err(LetoError::Overflow {
            reason: "complex square matrix element count",
        });
    };
    if matrix.len() != expected {
        return Err(LetoError::StorageError {
            reason: format!(
                "complex square transpose storage length {} does not match expected {expected}",
                matrix.len()
            ),
        });
    }
    if expected == 0 {
        return Ok(());
    }

    if side >= 8
        && vectorize_hardware_lanes::<16, T, _>(SquareTransposeKernel {
            matrix: &mut *matrix,
            side,
        })
        .is_some()
    {
        return Ok(());
    }
    if side >= 4
        && vectorize_hardware_lanes::<8, T, _>(SquareTransposeKernel {
            matrix: &mut *matrix,
            side,
        })
        .is_some()
    {
        return Ok(());
    }
    if side >= 2
        && vectorize_hardware_lanes::<4, T, _>(SquareTransposeKernel {
            matrix: &mut *matrix,
            side,
        })
        .is_some()
    {
        return Ok(());
    }
    transpose_scalar(matrix, side);
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
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
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
) where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    let full_side = side / SIDE * SIDE;
    for row in (0..full_side).step_by(SIDE) {
        let diagonal = row * side + row;
        let tile = load_tile::<T, A, SIDE>(simd, matrix, side, diagonal);
        store_tile(simd, matrix, side, diagonal, &tile);

        for column in (row + SIDE..full_side).step_by(SIDE) {
            let upper = row * side + column;
            let lower = column * side + row;
            // Both source tiles become owned register values before either
            // destination is overwritten. Each unordered tile pair occurs once.
            let upper_tile = load_tile::<T, A, SIDE>(simd, matrix, side, upper);
            let lower_tile = load_tile::<T, A, SIDE>(simd, matrix, side, lower);
            store_tile(simd, matrix, side, lower, &upper_tile);
            store_tile(simd, matrix, side, upper, &lower_tile);
        }
    }

    // Outside the complete square, the larger coordinate is at least
    // full_side. Visiting only row < column covers that border exactly once,
    // including its bottom-right partial square; diagonal values stay put.
    transpose_tail(matrix, side, full_side);
}

fn transpose_scalar<T>(matrix: &mut [T], side: usize) {
    transpose_tail(matrix, side, 0);
}

fn transpose_tail<T>(matrix: &mut [T], side: usize, full_side: usize) {
    for row in 0..side {
        for column in (row + 1).max(full_side)..side {
            matrix.swap(row * side + column, column * side + row);
        }
    }
}

#[cfg(test)]
mod tests;
