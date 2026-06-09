use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::matmul;

#[test]
fn test_matmul() {
    // 2x3 matrix
    let lhs_layout = Layout::c_contiguous([2, 3]).unwrap();
    let lhs_storage = VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let lhs = Array::new(lhs_layout, lhs_storage).unwrap();

    // 3x2 matrix
    let rhs_layout = Layout::c_contiguous([3, 2]).unwrap();
    let rhs_storage = VecStorage::new(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let rhs = Array::new(rhs_layout, rhs_storage).unwrap();

    // 2x2 output matrix
    let out_layout = Layout::c_contiguous([2, 2]).unwrap();
    let out_storage = VecStorage::fill(4, 0.0f32);
    let mut out = Array::new(out_layout, out_storage).unwrap();

    matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    assert_eq!(out.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_strided_and_transposed() {
    // 2x3 matrix
    let lhs_storage = VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let lhs = Array::new(Layout::c_contiguous([2, 3]).unwrap(), lhs_storage).unwrap();

    // 2x3 matrix to be transposed into 3x2
    let rhs_storage = VecStorage::new(vec![7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]);
    let rhs_t = Array::new(Layout::c_contiguous([2, 3]).unwrap(), rhs_storage).unwrap();
    let rhs = rhs_t.transpose([1, 0]).unwrap(); // shape [3, 2] with strides [1, 3]

    // 2x2 output matrix with stride 1 column
    let mut out = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::fill(4, 0.0f32),
    )
    .unwrap();

    matmul(&lhs.view(), &rhs, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    // Transposed lhs and transposed rhs
    let lhs_storage2 = VecStorage::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let lhs2_t = Array::new(Layout::c_contiguous([3, 2]).unwrap(), lhs_storage2).unwrap();
    let lhs2 = lhs2_t.transpose([1, 0]).unwrap(); // shape [2, 3] with strides [1, 2]

    let rhs2 = Array::new(
        Layout::c_contiguous([3, 2]).unwrap(),
        VecStorage::new(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )
    .unwrap();

    let mut out2 = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::fill(4, 0.0f32),
    )
    .unwrap();
    matmul(&lhs2, &rhs2.view(), &mut out2.view_mut()).unwrap();
    assert_eq!(out2.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}
