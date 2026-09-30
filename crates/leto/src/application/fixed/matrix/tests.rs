//! Unit tests for [`super::FixedMatrix`].

use super::FixedMatrix;

#[test]
fn fixed_matrix_multiplies_on_stack() {
    let lhs = FixedMatrix::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rhs = FixedMatrix::from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let product = lhs * rhs;

    assert_eq!(
        product,
        FixedMatrix::from_rows([[58.0, 64.0], [139.0, 154.0]])
    );
}

#[test]
fn fixed_matrix_determinant_matches_known_value() {
    let matrix = FixedMatrix::from_rows([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]]);

    assert_eq!(matrix.determinant(), -306.0);
}

#[test]
fn fixed_3x3_inverse_matches_known_value() {
    let matrix = FixedMatrix::from_rows([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
    let inv = matrix.try_inverse().unwrap();
    let expected =
        FixedMatrix::from_rows([[-24.0, 18.0, 5.0], [20.0, -15.0, -4.0], [-5.0, 4.0, 1.0]]);
    assert_eq!(inv, expected);
}

#[test]
fn fixed_3x3_inverse_times_original_is_identity() {
    let m = FixedMatrix::from_rows([[4.0, 7.0, 2.0], [2.0, 6.0, 1.0], [3.0, 5.0, 8.0]]);
    let inv = m.try_inverse().unwrap();
    let product = m * inv;
    let identity: FixedMatrix<f64, 3, 3> = FixedMatrix::identity();
    for row in 0..3 {
        for col in 0..3 {
            assert!((product[(row, col)] - identity[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn fixed_3x3_inverse_returns_none_for_singular() {
    let singular = FixedMatrix::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    assert!(singular.try_inverse().is_none());
}

#[test]
fn fixed_3x3_inverse_identity() {
    let identity = FixedMatrix::<f64, 3, 3>::identity();
    let inv = identity.try_inverse().unwrap();
    for row in 0..3 {
        for col in 0..3 {
            assert!((inv[(row, col)] - identity[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn fixed_2x2_inverse_matches_known_value() {
    let m = FixedMatrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    let inv = m.try_inverse().unwrap();
    let expected = FixedMatrix::from_rows([[-2.0, 1.0], [1.5, -0.5]]);
    for row in 0..2 {
        for col in 0..2 {
            assert!((inv[(row, col)] - expected[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn fixed_2x2_inverse_times_original_is_identity() {
    let m = FixedMatrix::from_rows([[5.0, 3.0], [2.0, 1.0]]);
    let inv = m.try_inverse().unwrap();
    let product = m * inv;
    let identity = FixedMatrix::<f64, 2, 2>::identity();
    for row in 0..2 {
        for col in 0..2 {
            assert!((product[(row, col)] - identity[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn fixed_2x2_inverse_returns_none_for_singular() {
    let singular = FixedMatrix::from_rows([[1.0, 2.0], [2.0, 4.0]]);
    assert!(singular.try_inverse().is_none());
}

#[test]
fn fixed_matrix_iterates_in_row_major_order() {
    let matrix = FixedMatrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);

    assert_eq!(
        matrix.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn fixed_matrix_converts_row_and_column_major_storage() {
    let row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let column_major = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

    let matrix = FixedMatrix::<f64, 3, 3>::from_row_major(row_major);

    assert_eq!(matrix.into_row_major(), row_major);
    assert_eq!(matrix.into_column_major(), column_major);
    assert_eq!(
        FixedMatrix::<f64, 3, 3>::from_column_major(column_major),
        matrix
    );
}

#[test]
fn fixed_4x4_diagonal_inverse_preserves_value_contract() {
    let matrix = FixedMatrix::<f64, 4, 4>::from_row_major([
        1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    assert_eq!(matrix.determinant(), 8.0);
    assert_eq!(
        matrix.into_column_major(),
        [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );

    let inverse = matrix
        .try_inverse()
        .expect("diagonal matrix is nonsingular");
    let expected = FixedMatrix::<f64, 4, 4>::from_row_major([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    assert_eq!(inverse, expected);
    assert_eq!(
        FixedMatrix::<f64, 4, 4>::from_row_major([
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
        .try_inverse(),
        None
    );
}
