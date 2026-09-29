//! `zip_fold`/`indexed_fold` logical-order tests.

use super::*;

#[test]
fn test_zip_fold_contiguous_inputs() {
    let lhs = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);

    let dot = zip_fold(&lhs.view(), &rhs.view(), 0.0, |acc, &x, &y| acc + x * y).unwrap();

    assert_eq!(dot, 300.0);
}

#[test]
fn test_zip_fold_strided_inputs_follow_logical_order() {
    let lhs_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_src = arr([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let lhs = lhs_src.transpose([1, 0]).unwrap();
    let rhs = rhs_src.transpose([1, 0]).unwrap();

    let weighted = zip_fold(&lhs, &rhs, 0.0, |acc, &x, &y| acc + x * y).unwrap();

    assert_eq!(weighted, 910.0);
}

#[test]
fn test_zip_fold_rejects_shape_mismatch() {
    let lhs = arr([2, 2], vec![0.0; 4]);
    let rhs = arr([1, 4], vec![0.0; 4]);

    assert!(zip_fold(&lhs.view(), &rhs.view(), 0.0, |acc, &x, &y| acc + x + y).is_err());
}

#[test]
fn test_indexed_fold_uses_logical_index() {
    let input = arr([2, 3], vec![1.0, -4.0, 2.0, 8.0, -7.0, 3.0]);

    let peak = indexed_fold(
        &input.view(),
        (0.0_f64, [0usize; 2]),
        |(best, best_index), index, &value| {
            let magnitude = value.abs();
            if magnitude > best {
                (magnitude, index)
            } else {
                (best, best_index)
            }
        },
    )
    .unwrap();

    assert_eq!(peak, (8.0, [1, 0]));
}

#[test]
fn test_indexed_fold_strided_input_follows_logical_order() {
    let input = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 9.0, 6.0]);
    let transposed = input.transpose([1, 0]).unwrap();

    let peak = indexed_fold(
        &transposed,
        (0.0_f64, [0usize; 2]),
        |(best, best_index), index, &value| {
            if value > best {
                (value, index)
            } else {
                (best, best_index)
            }
        },
    )
    .unwrap();

    assert_eq!(peak, (9.0, [1, 1]));
}

#[test]
fn test_indexed_fold_fortran_uses_column_major_logical_order() {
    let input = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let visited = indexed_fold_fortran(
        &input.view(),
        Vec::<([usize; 2], f64)>::new(),
        |mut acc, index, &value| {
            acc.push((index, value));
            acc
        },
    )
    .unwrap();

    assert_eq!(
        visited,
        vec![
            ([0, 0], 1.0),
            ([1, 0], 4.0),
            ([0, 1], 2.0),
            ([1, 1], 5.0),
            ([0, 2], 3.0),
            ([1, 2], 6.0),
        ]
    );
}

