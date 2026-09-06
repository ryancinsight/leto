//! Scalar entry points control layout operation specialization.
//!
//! Square entries bind one provider instantiation. Batch entries expose known
//! counts to callers so a single matrix eliminates batch dispatch and division.
//! ADR 0027 requires linked-map and timing evidence before retaining it.

use eunomia::{Bf16, Complex, F16};
use leto::LetoError;

use crate::domain::layout::{ComplexLayout, SquareTransposeError};

use super::{batch, square};

impl ComplexLayout for f32 {
    #[inline]
    fn transpose_complex_matrices(
        source: &[Complex<Self>],
        destination: &mut [Complex<Self>],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(), LetoError> {
        batch::transpose_complex_matrices(source, destination, matrix_count, rows, columns)
    }

    #[inline(never)]
    fn transpose_square_inplace(
        matrix: &mut [Complex<Self>],
        side: usize,
    ) -> Result<(), SquareTransposeError> {
        square::transpose_square_inplace(matrix, side)
    }
}

impl ComplexLayout for f64 {
    #[inline]
    fn transpose_complex_matrices(
        source: &[Complex<Self>],
        destination: &mut [Complex<Self>],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(), LetoError> {
        batch::transpose_complex_matrices(source, destination, matrix_count, rows, columns)
    }

    #[inline(never)]
    fn transpose_square_inplace(
        matrix: &mut [Complex<Self>],
        side: usize,
    ) -> Result<(), SquareTransposeError> {
        square::transpose_square_inplace(matrix, side)
    }
}

impl ComplexLayout for F16 {
    #[inline]
    fn transpose_complex_matrices(
        source: &[Complex<Self>],
        destination: &mut [Complex<Self>],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(), LetoError> {
        batch::transpose_complex_matrices(source, destination, matrix_count, rows, columns)
    }

    #[inline(never)]
    fn transpose_square_inplace(
        matrix: &mut [Complex<Self>],
        side: usize,
    ) -> Result<(), SquareTransposeError> {
        square::transpose_square_inplace(matrix, side)
    }
}

impl ComplexLayout for Bf16 {
    #[inline]
    fn transpose_complex_matrices(
        source: &[Complex<Self>],
        destination: &mut [Complex<Self>],
        matrix_count: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(), LetoError> {
        batch::transpose_complex_matrices(source, destination, matrix_count, rows, columns)
    }

    #[inline(never)]
    fn transpose_square_inplace(
        matrix: &mut [Complex<Self>],
        side: usize,
    ) -> Result<(), SquareTransposeError> {
        square::transpose_square_inplace(matrix, side)
    }
}
