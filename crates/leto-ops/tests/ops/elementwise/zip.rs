//! Mapping and zipping entry points over strided views.

use super::*;

#[test]
fn test_mapping_and_zipping() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let arr = Array::new(layout, VecStorage::new(vec![1, 2, 3, 4, 5, 6])).unwrap();

    // Map by reference
    let mapped = map(&arr.view(), |x| x * 10).unwrap();
    assert_eq!(mapped.storage().as_slice(), &[10, 20, 30, 40, 50, 60]);
    assert!(mapped.layout().is_c_contiguous());

    // Map on a transposed strided view
    let transposed = arr.transpose([1, 0]).unwrap();
    let mapped_t = map(&transposed, |x| x + 1).unwrap();
    assert_eq!(mapped_t.storage().as_slice(), &[2, 5, 3, 6, 4, 7]);
    assert!(mapped_t.layout().is_c_contiguous());

    // Zip-mapping in place
    let mut dest = Array::new(layout, VecStorage::fill(6, 100)).unwrap();
    zip_mut_with(dest.view_mut(), &arr.view(), |d, &s| {
        *d += s;
    })
    .unwrap();
    assert_eq!(dest.storage().as_slice(), &[101, 102, 103, 104, 105, 106]);

    // Shape mismatch validation
    let wrong_layout = Layout::c_contiguous([3, 2]).unwrap();
    let wrong_arr = Array::new(wrong_layout, VecStorage::fill(6, 0)).unwrap();
    let mut dest_mut = dest.view_mut();
    assert!(zip_mut_with(&mut dest_mut, &wrong_arr.view(), |_, _| {}).is_err());
}

#[test]
fn test_zip_mut_with_handles_strided_transposed_views() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let lhs_base = Array::new(layout, VecStorage::new(vec![1i32, 2, 3, 4, 5, 6])).unwrap();
    let rhs_base = Array::new(layout, VecStorage::new(vec![10i32, 20, 30, 40, 50, 60])).unwrap();
    let mut lhs_storage = lhs_base.into_vec();
    let mut lhs_view = leto::ArrayViewMut::try_new(
        Layout::c_contiguous([2, 3]).unwrap(),
        lhs_storage.as_mut_slice(),
    )
    .unwrap()
    .transpose_mut([1, 0])
    .unwrap();
    let rhs_view = rhs_base.transpose([1, 0]).unwrap();

    zip_mut_with(&mut lhs_view, &rhs_view, |left, right| {
        *left += *right;
    })
    .unwrap();

    assert_eq!(lhs_storage.as_slice(), &[11, 22, 33, 44, 55, 66]);
}

#[test]
fn test_indexed_zip_mut_with_uses_logical_indices() {
    let rhs = Array::from_shape_vec([2, 3], vec![10i32, 20, 30, 40, 50, 60]).unwrap();
    let mut lhs = Array::zeros([2, 3]);

    indexed_zip_mut_with(lhs.view_mut(), &rhs.view(), |[row, col], left, right| {
        *left = *right + (row as i32) * 100 + (col as i32);
    })
    .unwrap();

    assert_eq!(lhs.storage().as_slice(), &[10, 21, 32, 140, 151, 162]);
}

#[test]
fn test_indexed_zip_mut_with_handles_strided_transposed_views() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![1i32, 2, 3, 4, 5, 6])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![10i32, 20, 30, 40, 50, 60])).unwrap();
    let mut out_storage = vec![0i32; 6];
    let mut out = leto::ArrayViewMut::try_new(layout, out_storage.as_mut_slice())
        .unwrap()
        .transpose_mut([1, 0])
        .unwrap();
    let a_t = a.transpose([1, 0]).unwrap();
    let b_t = b.transpose([1, 0]).unwrap();

    indexed_zip_mut_with(&mut out, (&a_t, &b_t), |[row, col], left, (av, bv)| {
        *left = *av + *bv + (row as i32) * 10 + (col as i32);
    })
    .unwrap();

    assert_eq!(out_storage.as_slice(), &[11, 32, 53, 45, 66, 87]);
}

#[test]
fn integer_scalar_elementwise_ops_are_value_semantic() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let lhs = Array::new(layout, VecStorage::new(vec![1i32, -2, 3, 4, -5, 6])).unwrap();
    let rhs = Array::new(layout, VecStorage::new(vec![10i32, 20, -30, 40, 50, -60])).unwrap();
    let mut sum = Array::new(layout, VecStorage::fill(6, 0i32)).unwrap();
    let mut product = Array::new(layout, VecStorage::fill(6, 0i32)).unwrap();

    add(&lhs.view(), &rhs.view(), &mut sum.view_mut()).unwrap();
    mul(&lhs.view(), &rhs.view(), &mut product.view_mut()).unwrap();
    let shifted = scalar_map::<AddOp, _, 2>(&lhs.view(), 7).unwrap();

    assert_eq!(sum.storage().as_slice(), &[11, 18, -27, 44, 45, -54]);
    assert_eq!(
        product.storage().as_slice(),
        &[10, -40, -90, 160, -250, -360]
    );
    assert_eq!(shifted.storage().as_slice(), &[8, 5, 10, 11, 2, 13]);
}
