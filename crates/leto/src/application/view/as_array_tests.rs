#![allow(clippy::unwrap_used, reason = "test scope")]

use crate::application::array::Array;
use crate::infrastructure::storage::VecStorage;

#[test]
fn as_array_matches_view_indexing_including_strided() {
    // 3x4 source; take a strided sub-view (every other column) and confirm
    // the zero-copy borrowed array indexes identically to the view.
    let src = Array::<f64, VecStorage<f64>, 2>::from_shape_vec(
        [3, 4],
        (0..12).map(|i| i as f64).collect(),
    )
    .unwrap();
    let view = src.view();
    let borrowed = view.as_array();
    assert_eq!(borrowed.shape(), view.shape());
    for r in 0..3 {
        for c in 0..4 {
            assert_eq!(*borrowed.get([r, c]).unwrap(), *view.get([r, c]).unwrap());
            assert_eq!(borrowed[[r, c]], src[[r, c]]);
        }
    }

    // Strided sub-view: columns [1,3) step is exercised via slice.
    let strided = src.view().slice(&[(0, 3, 1), (1, 4, 2)]).unwrap();
    let strided_borrowed = strided.as_array();
    assert_eq!(strided_borrowed.shape(), strided.shape());
    for r in 0..strided.shape()[0] {
        for c in 0..strided.shape()[1] {
            assert_eq!(
                *strided_borrowed.get([r, c]).unwrap(),
                *strided.get([r, c]).unwrap(),
                "strided borrowed array must match the view at [{r},{c}]"
            );
        }
    }
}
