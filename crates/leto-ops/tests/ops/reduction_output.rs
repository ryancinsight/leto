#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{max_axis, product_axis};

/// Every expected element differs from zero, so an output cell the kernel
/// skipped would read as the zero-filled allocation and fail the comparison.
#[test]
fn test_allocating_axis_reductions_overwrite_every_output_element() {
    let empty = Array::new(
        Layout::c_contiguous([2, 0]).unwrap(),
        VecStorage::new(Vec::<f32>::new()),
    )
    .unwrap();
    let empty_product = product_axis(&empty.view(), 1).unwrap();
    assert_eq!(empty_product.storage().as_slice(), &[1.0, 1.0]);

    let negative = Array::from_shape_vec(
        [3, 4],
        vec![
            -1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0,
        ],
    )
    .unwrap();
    let column_max = max_axis(&negative.view(), 0).unwrap();
    assert_eq!(column_max.storage().as_slice(), &[-1.0, -2.0, -3.0, -4.0]);
    let row_max = max_axis(&negative.view(), 1).unwrap();
    assert_eq!(row_max.storage().as_slice(), &[-1.0, -5.0, -9.0]);
}
