//! Allocation-free layout movement for homogeneous complex matrix batches.

use eunomia::{
    layout::{cast_slice, cast_slice_mut},
    Pod,
};
use hermes_simd::{
    vectorize_hardware_lanes, ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel,
    Vector,
};
use leto::{ArrayView2, ArrayViewMut2, Complex, Layout, LetoError, Result};

/// Largest complex-square side supported by the current Hermes backends.
const MAX_COMPLEX_TILE_SIDE: usize = 8;
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
/// Leto's cache-budgeted generic assignment. Neither route allocates after the
/// caller provides `source` and `destination`.
///
/// # Errors
///
/// Returns [`LetoError::Overflow`] when the matrix or batch element count does
/// not fit `usize`. Returns [`LetoError::StorageError`] when either slice length
/// differs from `matrix_count * rows * columns`. Returns another [`LetoError`]
/// if the derived two-dimensional layouts are not representable. Validation
/// completes before mutation.
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
    let matrix_len = rows.checked_mul(columns).ok_or(LetoError::Overflow {
        reason: "complex matrix element count",
    })?;
    let total_len = matrix_count
        .checked_mul(matrix_len)
        .ok_or(LetoError::Overflow {
            reason: "complex matrix batch element count",
        })?;
    validate_length("source", source.len(), total_len)?;
    validate_length("destination", destination.len(), total_len)?;

    if total_len == 0 {
        return Ok(());
    }

    let source_layout = Layout::f_contiguous([columns, rows])?;
    let destination_layout = Layout::c_contiguous([columns, rows])?;
    if uses_register_complex_tiles(matrix_count, rows, columns)
        && transpose_hardware(source, destination, matrix_len, rows, columns)
    {
        return Ok(());
    }

    transpose_generic(
        source,
        destination,
        matrix_len,
        source_layout,
        destination_layout,
    );
    Ok(())
}

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
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) {
        let side = ComplexReg::<T, A>::COMPLEX_COUNT;
        debug_assert!((2..=MAX_COMPLEX_TILE_SIDE).contains(&side));
        transpose_tiled::<T, A>(
            self.source,
            self.destination,
            self.matrix_len,
            self.rows,
            self.columns,
            side,
        );
    }
}

#[inline(always)]
fn transpose_tiled<T, A>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_len: usize,
    rows: usize,
    columns: usize,
    side: usize,
) where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    let full_rows = rows / side * side;
    let full_columns = columns / side * side;

    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        for tile_row in (0..full_rows).step_by(side) {
            for tile_column in (0..full_columns).step_by(side) {
                let first_start = tile_row * columns + tile_column;
                let first_scalars = cast_slice(&source_matrix[first_start..first_start + side]);
                // SAFETY: the lane kernel runs inside A's proven target-feature
                // scope. `side == A::LANE_COUNT / 2`, and the source segment is
                // exactly one initialized interleaved complex register.
                let first = ComplexReg::from_interleaved(unsafe {
                    Vector::<T, A>::load_unaligned(first_scalars.as_ptr())
                });
                let mut tile = [first; MAX_COMPLEX_TILE_SIDE];
                for (local_row, register) in tile[..side].iter_mut().enumerate().skip(1) {
                    let start = (tile_row + local_row) * columns + tile_column;
                    let scalars = cast_slice(&source_matrix[start..start + side]);
                    // SAFETY: identical to the first-row load; every full tile
                    // row exposes `side` complete complex samples.
                    *register = ComplexReg::from_interleaved(unsafe {
                        Vector::<T, A>::load_unaligned(scalars.as_ptr())
                    });
                }

                ComplexReg::transpose_square(&mut tile[..side]);

                for (local_column, register) in tile[..side].iter().copied().enumerate() {
                    let start = (tile_column + local_column) * rows + tile_row;
                    let scalars = cast_slice_mut(&mut destination_matrix[start..start + side]);
                    // SAFETY: the lane kernel proves A, and the destination
                    // segment is exactly one writable interleaved register.
                    unsafe {
                        register
                            .into_interleaved()
                            .store_unaligned(scalars.as_mut_ptr());
                    }
                }
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

fn transpose_generic<T: Copy>(
    source: &[T],
    destination: &mut [T],
    matrix_len: usize,
    source_layout: Layout<2>,
    destination_layout: Layout<2>,
) {
    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        let source_view = ArrayView2::try_new(source_layout, source_matrix)
            .expect("invariant: validated complex matrix source fits its layout");
        let mut destination_view = ArrayViewMut2::try_new(destination_layout, destination_matrix)
            .expect("invariant: validated complex matrix destination fits its layout");
        destination_view.assign(&source_view);
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
