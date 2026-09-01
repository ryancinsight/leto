use core::fmt::Debug;
use eunomia::{Pod, F16};
use hermes_simd::LaneScalar;
use leto::{Complex, LetoError};
use leto_ops::transpose_complex_matrices;

fn expected<T: Copy + Default>(
    source: &[Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) -> Vec<Complex<T>> {
    let matrix_len = rows * columns;
    let mut output = vec![Complex::default(); source.len()];
    for matrix in 0..matrix_count {
        let base = matrix * matrix_len;
        for row in 0..rows {
            for column in 0..columns {
                output[base + column * rows + row] = source[base + row * columns + column];
            }
        }
    }
    output
}

fn assert_case<T>(
    matrix_count: usize,
    rows: usize,
    columns: usize,
    value: impl Fn(usize) -> Complex<T>,
) where
    T: LaneScalar + Pod + Default + PartialEq + Debug,
{
    let len = matrix_count * rows * columns;
    let source = (0..len).map(value).collect::<Vec<_>>();
    let expected = expected(&source, matrix_count, rows, columns);
    let mut destination = vec![Complex::default(); len];
    transpose_complex_matrices(&source, &mut destination, matrix_count, rows, columns)
        .expect("valid complex matrix batch transposes");
    assert_eq!(destination, expected);
}

#[test]
fn complex_matrix_batches_preserve_values_across_full_and_ragged_tiles() {
    let cases = [(256, 15, 13), (256, 16, 16), (3, 5, 7), (1, 35, 67)];
    for (matrix_count, rows, columns) in cases {
        assert_case(matrix_count, rows, columns, |index| {
            Complex::new(index as f32 + 0.25, -(index as f32) - 0.5)
        });
        assert_case(matrix_count, rows, columns, |index| {
            Complex::new(index as f64 + 0.25, -(index as f64) - 0.5)
        });
        assert_case(matrix_count, rows, columns, |index| {
            let value = (index % 127) as f32;
            Complex::new(F16::from_f32(value + 0.25), F16::from_f32(-value - 0.5))
        });
    }
}

#[test]
fn complex_matrix_batch_validation_is_failure_atomic() {
    let source = (0..12)
        .map(|index| Complex::new(index as f64, -1.0))
        .collect::<Vec<_>>();
    let mut short_destination = vec![Complex::new(17.0_f64, 19.0); 11];
    let before = short_destination.clone();
    let error = transpose_complex_matrices(&source, &mut short_destination, 2, 2, 3)
        .expect_err("short destination must be rejected before mutation");
    assert!(matches!(error, LetoError::StorageError { .. }));
    assert_eq!(short_destination, before);

    let mut destination = vec![Complex::new(23.0_f64, 29.0); 12];
    let before = destination.clone();
    let error = transpose_complex_matrices(&source[..11], &mut destination, 2, 2, 3)
        .expect_err("short source must be rejected before mutation");
    assert!(matches!(error, LetoError::StorageError { .. }));
    assert_eq!(destination, before);

    let error = transpose_complex_matrices::<f64>(&[], &mut [], usize::MAX, 2, 2)
        .expect_err("overflowing batch dimensions must be rejected");
    assert_eq!(
        error,
        LetoError::Overflow {
            reason: "complex matrix batch element count"
        }
    );
}

#[test]
fn complex_matrix_batch_empty_shapes_are_no_ops() {
    transpose_complex_matrices::<f32>(&[], &mut [], 0, 7, 5).expect("zero matrices are valid");
    transpose_complex_matrices::<f64>(&[], &mut [], 3, 0, 5).expect("zero rows are valid");
    transpose_complex_matrices::<f64>(&[], &mut [], 3, 5, 0).expect("zero columns are valid");
}
