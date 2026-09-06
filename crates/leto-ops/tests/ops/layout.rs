mod payloads;
mod square;
mod square_contract;

use eunomia::{Bf16, F16};
use leto::LetoError;
use leto_ops::ComplexLayout;

use payloads::{assert_bits, expected, values, PayloadScalar};

fn assert_batches<T: PayloadScalar + ComplexLayout>() {
    for (matrix_count, rows, columns) in [(256, 15, 13), (256, 16, 16), (3, 5, 7), (1, 35, 67)] {
        let source = values::<T>(matrix_count * rows * columns);
        let expected = expected(&source, matrix_count, rows, columns);
        let mut destination = source.clone();
        T::transpose_complex_matrices(&source, &mut destination, matrix_count, rows, columns)
            .expect("valid complex matrix batch transposes");
        assert_bits(&destination, &expected);
    }
}

#[test]
fn complex_matrix_batches_preserve_values_across_full_and_ragged_tiles() {
    assert_batches::<f32>();
    assert_batches::<f64>();
    assert_batches::<F16>();
    assert_batches::<Bf16>();
}

fn assert_batch_validation<T: PayloadScalar + ComplexLayout>() {
    let source = values::<T>(13);
    for (source_len, destination_len) in [(12, 11), (12, 13), (11, 12), (13, 12)] {
        let mut destination = values::<T>(destination_len);
        let before = destination.clone();
        let error = T::transpose_complex_matrices(&source[..source_len], &mut destination, 2, 2, 3)
            .expect_err("inexact storage must be rejected before mutation");
        assert!(matches!(error, LetoError::StorageError { .. }));
        assert_bits(&destination, &before);
    }
    for (count, rows, columns, reason) in [
        (usize::MAX, 2, 2, "complex matrix batch element count"),
        (1, usize::MAX, 2, "complex matrix element count"),
        (0, usize::MAX, 2, "complex matrix element count"),
    ] {
        let mut destination = values::<T>(3);
        let before = destination.clone();
        let error = T::transpose_complex_matrices(&[], &mut destination, count, rows, columns)
            .expect_err("overflowing dimensions must be rejected before mutation");
        assert_eq!(error, LetoError::Overflow { reason });
        assert_bits(&destination, &before);
    }
}

#[test]
fn complex_matrix_batch_validation_is_failure_atomic() {
    assert_batch_validation::<f32>();
    assert_batch_validation::<f64>();
    assert_batch_validation::<F16>();
    assert_batch_validation::<Bf16>();
}

fn assert_empty_batches<T: PayloadScalar + ComplexLayout>() {
    for (count, rows, columns) in [
        (0, 7, 5),
        (3, 0, 5),
        (3, 5, 0),
        (usize::MAX, 0, usize::MAX),
        (usize::MAX, usize::MAX, 0),
    ] {
        let mut storage = values::<T>(3);
        let before = storage.clone();
        T::transpose_complex_matrices(&[], &mut storage[1..1], count, rows, columns)
            .expect("zero total extent accepts empty slices");
        assert_bits(&storage, &before);
    }
}

#[test]
fn complex_matrix_batch_empty_shapes_are_no_ops() {
    assert_empty_batches::<f32>();
    assert_empty_batches::<f64>();
    assert_empty_batches::<F16>();
    assert_empty_batches::<Bf16>();
}
