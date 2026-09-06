//! Scalar capability for checked complex matrix movement.

use eunomia::{Complex, Pod};
use hermes_simd::LaneScalar;
use leto::LetoError;

use super::SquareTransposeError;

/// Provider-owned complex layout operations for supported scalar representations.
///
/// The four scalar implementations bind one checked generic algorithm per
/// operation inside Leto. Selection is static; no trait object or allocation
/// is introduced. Scalar bits, including NaN payloads, are preserved.
#[diagnostic::on_unimplemented(
    message = "complex matrix movement requires a scalar implementing leto_ops::ComplexLayout"
)]
pub trait ComplexLayout: LaneScalar + Pod {
    /// Transposes adjacent row-major complex matrices into adjacent row-major outputs.
    ///
    /// Each source matrix has shape `[rows, columns]`; each destination matrix has
    /// shape `[columns, rows]`. Matrix order is preserved. The operation validates
    /// both complete slice lengths before writing any destination element.
    ///
    /// High-count batches of small matrices use the widest exact Hermes hardware
    /// width that fits a complete square tile. Other shapes and targets reuse
    /// Leto's cache-budgeted [`leto::transpose_copy`]. Neither route allocates after the
    /// caller provides `source` and `destination`.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] when the matrix or batch element count does
    /// not fit `usize`. Returns [`LetoError::StorageError`] when either slice length
    /// differs from `matrix_count * rows * columns`. Validation completes before
    /// mutation, in this order: matrix count, batch count, source length, then
    /// destination length. Exact nonempty complex slices already bound each
    /// matrix to the signed extent supported by [`leto::transpose_copy`].
    ///
    /// # Examples
    ///
    /// ```
    /// use leto::Complex;
    /// use leto_ops::ComplexLayout;
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
    /// f32::transpose_complex_matrices(&source, &mut destination, 1, 2, 3)?;
    /// assert_eq!(
    ///     destination,
    ///     [source[0], source[3], source[1], source[4], source[2], source[5]]
    /// );
    /// # Ok::<(), leto::LetoError>(())
    /// ```
    fn transpose_complex_matrices(
        source: &[Complex<Self>],
        destination: &mut [Complex<Self>],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(), LetoError>;

    /// Transposes a row-major complex square in its existing storage.
    ///
    /// Every sample at `(row, column)` moves to `(column, row)` without arithmetic
    /// or allocation. All scalar bits survive, including signed zeros and NaN
    /// payloads. Complete hardware tiles exchange through registers; remaining
    /// pairs and targets without a suitable hardware width use scalar swaps.
    ///
    /// # Errors
    ///
    /// Returns [`SquareTransposeError::Overflow`] if `side * side` cannot be
    /// represented, or [`SquareTransposeError::Length`] if `matrix.len()` differs
    /// from that extent. Errors carry dimensions without allocating storage.
    /// Both errors leave the complete input unchanged. Side zero requires empty
    /// storage and performs no work.
    ///
    /// # Examples
    ///
    /// ```
    /// use leto::Complex;
    /// use leto_ops::ComplexLayout;
    ///
    /// let mut matrix = [1.0_f32, 2.0, 3.0, 4.0].map(|re| Complex::new(re, -re));
    /// let original = matrix;
    /// f32::transpose_square_inplace(&mut matrix, 2)?;
    /// assert_eq!(matrix, [original[0], original[2], original[1], original[3]]);
    /// # Ok::<(), leto_ops::SquareTransposeError>(())
    /// ```
    fn transpose_square_inplace(
        matrix: &mut [Complex<Self>],
        side: usize,
    ) -> Result<(), SquareTransposeError>;
}
