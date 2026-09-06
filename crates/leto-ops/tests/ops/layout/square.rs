use eunomia::{Bf16, F16};
use leto::LetoError;
use leto_ops::transpose_square_inplace;

use super::payloads::{assert_bits, values, PayloadScalar};
use super::square_contract::assert_squares;

fn assert_dispatch<T: PayloadScalar>() {
    assert_squares::<T>(|matrix, side| {
        transpose_square_inplace(matrix, side).expect("exact square storage transposes");
    });
}

#[test]
fn square_transpose_preserves_coordinate_payloads_and_guards() {
    assert_dispatch::<f32>();
    assert_dispatch::<f64>();
    assert_dispatch::<F16>();
    assert_dispatch::<Bf16>();
}

fn assert_validation<T: PayloadScalar>() {
    for (side, len) in [
        (0, 1),
        (1, 0),
        (1, 2),
        (3, 8),
        (3, 10),
        (16, 255),
        (16, 257),
    ] {
        let mut storage = values::<T>(len + 4);
        let before = storage.clone();
        let error = transpose_square_inplace(&mut storage[1..=len], side)
            .expect_err("inexact square storage is rejected");
        assert!(matches!(error, LetoError::StorageError { .. }));
        assert_bits(&storage, &before);
    }
    // usize has an even bit width; this is the first side whose square overflows.
    let first_overflow = 1usize << (usize::BITS / 2);
    for side in [first_overflow, usize::MAX] {
        let mut storage = values::<T>(7);
        let before = storage.clone();
        let error = transpose_square_inplace(&mut storage[1..6], side)
            .expect_err("overflowing square extent is rejected");
        assert_eq!(
            error,
            LetoError::Overflow {
                reason: "complex square matrix element count"
            }
        );
        assert_bits(&storage, &before);
    }
}

#[test]
fn square_transpose_rejects_invalid_extent_before_mutation() {
    assert_validation::<f32>();
    assert_validation::<f64>();
    assert_validation::<F16>();
    assert_validation::<Bf16>();
}
