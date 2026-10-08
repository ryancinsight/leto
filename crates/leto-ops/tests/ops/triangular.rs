#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, SliceArg, Storage, VecStorage};
use leto_ops::{triangular, triangular_into, triangular_keeps, TriangularMode};

#[test]
fn keeps_matches_the_device_predicate_on_unit_cases() {
    assert!(triangular_keeps(TriangularMode::Lower, 0, 0, 0));
    assert!(triangular_keeps(TriangularMode::Lower, 1, 0, 0));
    assert!(!triangular_keeps(TriangularMode::Lower, 0, 1, 0));
    assert!(triangular_keeps(TriangularMode::Upper, 0, 0, 0));
    assert!(triangular_keeps(TriangularMode::Upper, 0, 1, 0));
    assert!(!triangular_keeps(TriangularMode::Upper, 1, 0, 0));
    assert!(triangular_keeps(TriangularMode::Lower, 0, 1, 1));
    assert!(!triangular_keeps(TriangularMode::Lower, 0, 2, 1));
    assert!(!triangular_keeps(TriangularMode::Lower, 0, 0, -1));
    assert!(triangular_keeps(TriangularMode::Lower, 1, 0, -1));
}

#[test]
fn keeps_matches_the_widened_mathematical_comparison() {
    for row in 0..=4 {
        for col in 0..=4 {
            for diagonal in [i64::MIN, -5, -1, 0, 1, 5, i64::MAX] {
                let boundary = i128::try_from(row).unwrap() + i128::from(diagonal);
                let coordinate = i128::try_from(col).unwrap();
                assert_eq!(
                    triangular_keeps(TriangularMode::Lower, row, col, diagonal),
                    coordinate <= boundary,
                    "lower row={row} col={col} diagonal={diagonal}"
                );
                assert_eq!(
                    triangular_keeps(TriangularMode::Upper, row, col, diagonal),
                    coordinate >= boundary,
                    "upper row={row} col={col} diagonal={diagonal}"
                );
            }
        }
    }
}

#[test]
fn triangular_masks_match_numpy_layout() {
    let input = Array::from_shape_vec([3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
    let lower = triangular(&input.view(), TriangularMode::Lower, 0).unwrap();
    assert_eq!(
        lower.storage().as_slice(),
        &[1, 0, 0, 0, 5, 6, 0, 0, 9, 10, 11, 0]
    );
    let upper = triangular(&input.view(), TriangularMode::Upper, 0).unwrap();
    assert_eq!(
        upper.storage().as_slice(),
        &[1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12]
    );
    let shifted = triangular(&input.view(), TriangularMode::Lower, 1).unwrap();
    assert_eq!(
        shifted.storage().as_slice(),
        &[1, 2, 0, 0, 5, 6, 7, 0, 9, 10, 11, 12]
    );
    let grown = triangular(&input.view(), TriangularMode::Upper, -1).unwrap();
    assert_eq!(
        grown.storage().as_slice(),
        &[1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12]
    );
}

#[test]
fn triangular_serves_strided_views() {
    let input = Array::from_shape_vec([2, 4], vec![1, 100, 2, 200, 3, 300, 4, 400]).unwrap();
    let selected = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(0), None, 2)])
        .unwrap();
    assert_eq!(selected.shape(), [2, 2]);
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(4, 0)).unwrap();
    triangular_into(&selected, TriangularMode::Lower, 0, &mut output.view_mut()).unwrap();
    // [[1,2],[3,4]] tril -> [[1,0],[3,4]].
    assert_eq!(output.storage().as_slice(), &[1, 0, 3, 4]);
}

#[test]
fn contiguous_and_strided_paths_agree_on_diagonal_extremes() {
    // The contiguous fast path must compute the same mask as the
    // per-element strided path for every mode and diagonal, including the
    // extremes where the split point saturates to an empty or full row.
    let input = Array::from_shape_vec([4, 5], (1..=20).collect::<Vec<_>>()).unwrap();
    for mode in [TriangularMode::Lower, TriangularMode::Upper] {
        for diagonal in [i64::MIN, -100, -5, -1, 0, 1, 5, 100, i64::MAX] {
            let layout = Layout::c_contiguous([4, 5]).unwrap();
            let mut dense = Array::new(layout, VecStorage::fill(20, 0)).unwrap();
            triangular_into(&input.view(), mode, diagonal, &mut dense.view_mut()).unwrap();
            let mut backing = Array::from_shape_vec([4, 10], vec![0; 40]).unwrap();
            let mut stepped = backing
                .slice_with_mut::<2>(&[SliceArg::All, SliceArg::range(Some(0), None, 2)])
                .unwrap();
            triangular_into(&input.view(), mode, diagonal, &mut stepped).unwrap();
            let stepped_dense: Vec<_> = backing
                .storage()
                .as_slice()
                .chunks_exact(10)
                .flat_map(|row| [row[0], row[2], row[4], row[6], row[8]])
                .collect();
            assert_eq!(
                dense.storage().as_slice(),
                stepped_dense.as_slice(),
                "mode={mode:?} diagonal={diagonal}"
            );
        }
    }
}

// The predicate evaluates at compile time.
const _: () = {
    assert!(triangular_keeps(TriangularMode::Lower, 1, 0, 0));
    assert!(!triangular_keeps(TriangularMode::Upper, 1, 0, 0));
};

#[test]
fn triangular_rejects_shape_mismatch() {
    let input = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let bad = Layout::c_contiguous([3, 2]).unwrap();
    let mut output = Array::new(bad, VecStorage::fill(6, 0)).unwrap();
    assert!(triangular_into(
        &input.view(),
        TriangularMode::Lower,
        0,
        &mut output.view_mut()
    )
    .is_err());
}
