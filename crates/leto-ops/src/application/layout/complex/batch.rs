//! Allocation-free layout movement for homogeneous complex matrix batches.

use super::tile::{load_tile, store_tile};
use eunomia::Pod;
use hermes_simd::{
    vectorize_hardware_lanes, ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel,
};
use leto::{transpose_copy, Complex, LetoError, Result};

/// Pinned Apollo phase measurements show that dispatch amortizes at this batch
/// count while smaller batches remain with Leto's generic tiled transpose.
const REGISTER_TRANSPOSE_MIN_MATRICES: usize = 256;
/// The measured register-tile regime ends before matrices become cache-copy
/// dominated; larger matrices retain Leto's cache-budgeted generic kernel.
const REGISTER_TRANSPOSE_MAX_MATRIX_SIDE: usize = 16;

/// Transposes adjacent row-major complex matrices into adjacent row-major outputs.
///
/// Each source matrix has shape `[rows, columns]`; each destination matrix has
/// shape `[columns, rows]`. Matrix order is preserved. The operation validates
/// both complete slice lengths before writing any destination element.
///
/// High-count batches of small matrices use the widest exact Hermes hardware
/// width that fits a complete square tile. Other shapes and targets reuse
/// Leto's cache-budgeted [`transpose_copy`]. Neither route allocates after the
/// caller provides `source` and `destination`.
///
/// # Errors
///
/// Returns [`LetoError::Overflow`] when the matrix or batch element count does
/// not fit `usize`. Returns [`LetoError::StorageError`] when either slice length
/// differs from `matrix_count * rows * columns`. Validation completes before
/// mutation, in this order: matrix count, batch count, source length, then
/// destination length. Exact nonempty complex slices already bound each
/// matrix to the signed extent supported by [`transpose_copy`].
///
/// # Examples
///
/// ```
/// use leto::Complex;
/// use leto_ops::transpose_complex_matrices;
///
/// let source = [
///     Complex::new(1.0_f32, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0),
///     Complex::new(4.0, 0.0),
///     Complex::new(5.0, 0.0),
///     Complex::new(6.0, 0.0),
/// ];
/// let mut destination = [Complex::new(0.0_f32, 0.0); 6];
/// transpose_complex_matrices(&source, &mut destination, 1, 2, 3)?;
/// assert_eq!(
///     destination,
///     [source[0], source[3], source[1], source[4], source[2], source[5]]
/// );
/// # Ok::<(), leto::LetoError>(())
/// ```
pub fn transpose_complex_matrices<T>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) -> Result<()>
where
    T: LaneScalar + Pod,
{
    #[expect(
        clippy::unnecessary_lazy_evaluations,
        reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
    )]
    let matrix_len = rows
        .checked_mul(columns)
        .ok_or_else(|| LetoError::Overflow {
            reason: "complex matrix element count",
        })?;
    #[expect(
        clippy::unnecessary_lazy_evaluations,
        reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
    )]
    let total_len = matrix_count
        .checked_mul(matrix_len)
        .ok_or_else(|| LetoError::Overflow {
            reason: "complex matrix batch element count",
        })?;
    validate_length("source", source.len(), total_len)?;
    validate_length("destination", destination.len(), total_len)?;

    if total_len == 0 {
        return Ok(());
    }

    if uses_register_complex_tiles(matrix_count, rows, columns)
        && transpose_hardware(source, destination, matrix_len, rows, columns)
    {
        return Ok(());
    }

    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        transpose_copy(source_matrix, destination_matrix, rows, columns)?;
    }
    Ok(())
}

#[inline]
fn validate_length(role: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(LetoError::StorageError {
        reason: format!(
            "complex matrix transpose {role} length {actual} does not match expected {expected}"
        ),
    })
}

#[inline]
const fn uses_register_complex_tiles(matrix_count: usize, rows: usize, columns: usize) -> bool {
    matrix_count >= REGISTER_TRANSPOSE_MIN_MATRICES
        && rows <= REGISTER_TRANSPOSE_MAX_MATRIX_SIDE
        && columns <= REGISTER_TRANSPOSE_MAX_MATRIX_SIDE
}

#[inline]
fn transpose_hardware<T>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_len: usize,
    rows: usize,
    columns: usize,
) -> bool
where
    T: LaneScalar + Pod,
{
    let minimum_side = rows.min(columns);
    if minimum_side >= 8
        && vectorize_hardware_lanes::<16, T, _>(ComplexTransposeKernel {
            source,
            destination: &mut *destination,
            matrix_len,
            rows,
            columns,
        })
        .is_some()
    {
        return true;
    }
    if minimum_side >= 4
        && vectorize_hardware_lanes::<8, T, _>(ComplexTransposeKernel {
            source,
            destination: &mut *destination,
            matrix_len,
            rows,
            columns,
        })
        .is_some()
    {
        return true;
    }
    minimum_side >= 2
        && vectorize_hardware_lanes::<4, T, _>(ComplexTransposeKernel {
            source,
            destination,
            matrix_len,
            rows,
            columns,
        })
        .is_some()
}

struct ComplexTransposeKernel<'a, T> {
    source: &'a [Complex<T>],
    destination: &'a mut [Complex<T>],
    matrix_len: usize,
    rows: usize,
    columns: usize,
}

impl<T> LaneKernel<T> for ComplexTransposeKernel<'_, T>
where
    T: LaneScalar + Pod,
{
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        // Only the register side specializes storage; matrix dimensions stay
        // runtime values so shapes do not multiply the emitted kernels.
        match ComplexReg::<T, A>::COMPLEX_COUNT {
            2 => transpose_tiled::<T, A, 2>(self, &simd),
            4 => transpose_tiled::<T, A, 4>(self, &simd),
            8 => transpose_tiled::<T, A, 8>(self, &simd),
            _ => {
                unreachable!("invariant: requested hardware widths hold 2, 4 or 8 complex samples")
            }
        }
    }
}

#[inline(always)]
fn transpose_tiled<T, A, const SIDE: usize>(batch: ComplexTransposeKernel<'_, T>, simd: &Simd<T, A>)
where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    let ComplexTransposeKernel {
        source,
        destination,
        matrix_len,
        rows,
        columns,
    } = batch;
    let full_rows = rows / SIDE * SIDE;
    let full_columns = columns / SIDE * SIDE;

    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        for tile_row in (0..full_rows).step_by(SIDE) {
            for tile_column in (0..full_columns).step_by(SIDE) {
                let tile = load_tile::<T, A, SIDE>(
                    simd,
                    source_matrix,
                    columns,
                    tile_row * columns + tile_column,
                );
                store_tile(
                    simd,
                    destination_matrix,
                    rows,
                    tile_column * rows + tile_row,
                    &tile,
                );
            }
        }

        for row in 0..rows {
            let first_tail_column = if row < full_rows { full_columns } else { 0 };
            for column in first_tail_column..columns {
                destination_matrix[column * rows + row] = source_matrix[row * columns + column];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::uses_register_complex_tiles;

    #[test]
    fn selects_only_the_measured_small_matrix_regime() {
        assert!(uses_register_complex_tiles(256, 16, 16));
        assert!(!uses_register_complex_tiles(255, 16, 16));
        assert!(!uses_register_complex_tiles(256, 17, 16));
        assert!(!uses_register_complex_tiles(256, 16, 17));
    }
}
