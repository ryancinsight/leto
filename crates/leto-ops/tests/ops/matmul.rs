#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{matmul, matmul_accumulate, matmul_with_tile_policy, matvec, MatmulTilePolicy};

#[test]
fn test_matvec_contiguous() {
    // A = [[1,2,3],[4,5,6]] (2x3), x = [7,8,9] → A·x = [50, 122].
    let a = Array::new(
        Layout::c_contiguous([2, 3]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let x = Array::new(
        Layout::c_contiguous([3]).unwrap(),
        VecStorage::new(vec![7.0f64, 8.0, 9.0]),
    )
    .unwrap();
    let mut out = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::fill(2, 0.0f64),
    )
    .unwrap();

    matvec(&a.view(), &x.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[50.0, 122.0]);
}

#[test]
fn test_matvec_transposed_strided() {
    // Aᵀ·x via a transposed (strided) view: A = [[1,2,3],[4,5,6]],
    // Aᵀ = [[1,4],[2,5],[3,6]] (3x2), x = [1,1] → Aᵀ·x = [5, 7, 9].
    let a = Array::new(
        Layout::c_contiguous([2, 3]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let a_t = a.transpose([1, 0]).unwrap(); // shape [3, 2], strides [1, 3]
    let x = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::new(vec![1.0f64, 1.0]),
    )
    .unwrap();
    let mut out = Array::new(
        Layout::c_contiguous([3]).unwrap(),
        VecStorage::fill(3, 0.0f64),
    )
    .unwrap();

    matvec(&a_t, &x.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_matvec_shape_mismatch() {
    let a = Array::new(
        Layout::c_contiguous([2, 3]).unwrap(),
        VecStorage::fill(6, 1.0f64),
    )
    .unwrap();
    let x = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::fill(2, 1.0f64),
    )
    .unwrap();
    let mut out = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::fill(2, 0.0f64),
    )
    .unwrap();
    assert!(matvec(&a.view(), &x.view(), &mut out.view_mut()).is_err());
}

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

#[test]
fn matmul_matches_dense_on_sparse_and_dense_inputs() {
    let b = Array::from_shape_vec([5, 3], (0..15).map(|i| i as f64 + 1.0).collect()).unwrap();

    // Sparse lhs: 2 of 25 nonzero (8% < 10% threshold).
    let mut sparse = vec![0.0f64; 25];
    sparse[1] = 3.0; // (0, 1)
    sparse[3 * 5 + 2] = -2.0; // (3, 2)
    let a = Array::from_shape_vec([5, 5], sparse).unwrap();

    let mut out_auto = Array::zeros([5, 3]);
    matmul(&a.view(), &b.view(), &mut out_auto.view_mut()).unwrap();

    // Compare with manual/direct spmm:
    let csr = leto_ops::CsrMatrix::from_dense(&a.view());
    let mut out_spmm = vec![0.0f64; 15];
    leto_ops::spmm_into(&csr, &b.view(), &mut out_spmm).unwrap();
    assert_eq!(
        out_auto.storage().as_slice(),
        &out_spmm[..],
        "sparse-routed matmul must match spmm"
    );

    // Dense lhs (all nonzero) → dense route.
    let dense = Array::from_shape_vec([5, 5], (1..=25).map(|i| i as f64).collect()).unwrap();
    let mut o1 = Array::zeros([5, 3]);
    let mut o2 = Array::zeros([5, 3]);
    matmul(&dense.view(), &b.view(), &mut o1.view_mut()).unwrap();
    matmul(&dense.view(), &b.view(), &mut o2.view_mut()).unwrap();
    assert_eq!(o1.storage().as_slice(), o2.storage().as_slice());
}

#[test]
fn dense_common_route_honors_explicit_tile_policy() {
    let n = 64usize;
    let lhs = Array::from_shape_vec(
        [n, n],
        (0..n * n)
            .map(|i| (i as f64 * 0.03125 + 1.0) * 1.0e-3)
            .collect(),
    )
    .unwrap();
    let rhs = Array::from_shape_vec(
        [n, n],
        (0..n * n)
            .map(|i| (i as f64 * 0.0175 + 0.5) * 1.0e-3)
            .collect(),
    )
    .unwrap();
    let mut one_row = Array::zeros([n, n]);
    let mut thirty_two_rows = Array::zeros([n, n]);

    matmul_with_tile_policy(
        &lhs.view(),
        &rhs.view(),
        &mut one_row.view_mut(),
        MatmulTilePolicy::fixed(1).unwrap(),
    )
    .unwrap();
    matmul_with_tile_policy(
        &lhs.view(),
        &rhs.view(),
        &mut thirty_two_rows.view_mut(),
        MatmulTilePolicy::fixed(32).unwrap(),
    )
    .unwrap();

    assert_eq!(
        one_row.storage().as_slice(),
        thirty_two_rows.storage().as_slice()
    );
}

#[test]
fn dense_common_route_accumulates_without_overwriting() {
    let lhs = Array::from_shape_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Array::from_shape_vec([2, 2], vec![5.0f64, 6.0, 7.0, 8.0]).unwrap();
    let mut product = Array::zeros([2, 2]);
    let mut out = Array::from_shape_vec([2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

    matmul_with_tile_policy(
        &lhs.view(),
        &rhs.view(),
        &mut product.view_mut(),
        MatmulTilePolicy::fixed(1).unwrap(),
    )
    .unwrap();
    matmul_accumulate(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let expected: Vec<_> = [10.0, 20.0, 30.0, 40.0]
        .into_iter()
        .zip(product.storage().as_slice().iter().copied())
        .map(|(initial, value)| initial + value)
        .collect();
    assert_eq!(out.storage().as_slice(), expected.as_slice());
}

#[test]
fn test_matmul_non_contiguous_all() {
    use leto::SliceArg;

    // Both lhs, rhs, and out are non-contiguous.
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = Array::from_shape_vec([4, 2], a_data).unwrap();
    let a_sliced = a
        .slice_with::<2>(&[SliceArg::range(None, None, 2), SliceArg::All])
        .unwrap(); // shape [2, 2]

    let b_data = vec![9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let b = Array::from_shape_vec([4, 2], b_data).unwrap();
    let b_sliced = b
        .slice_with::<2>(&[SliceArg::range(None, None, 2), SliceArg::All])
        .unwrap(); // shape [2, 2]

    // Non-contiguous output
    let mut out_base = Array::zeros([4, 2]);
    {
        let mut out_sliced = out_base
            .slice_with_mut::<2>(&[SliceArg::range(None, None, 2), SliceArg::All])
            .unwrap(); // shape [2, 2]
        matmul(&a_sliced, &b_sliced, &mut out_sliced).unwrap();
    }

    assert_eq!(
        out_base.storage().as_slice(),
        &[35.0, 38.0, 0.0, 0.0, 123.0, 134.0, 0.0, 0.0]
    );
}

#[test]
fn test_matmul_hang_reproduction() {
    let n = 256usize;
    let k = 32usize;

    let mut dense_a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if (i * 7 + j * 13) % 20 == 0 {
                dense_a[i * n + j] = ((i + j) % 7 + 1) as f64;
            }
        }
    }

    let a = Array::from_shape_vec([n, n], dense_a).unwrap();
    let b = Array::from_shape_vec(
        [n, k],
        (0..n * k)
            .map(|i| (i as f64 * 0.731 + 1.0) * 1.0e-3)
            .collect(),
    )
    .unwrap();
    let mut out = Array::zeros([n, k]);

    matmul(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
}

/// A C-dense output view at a non-zero offset (a sliced sub-array, as produced
/// per-batch by `batched_matmul`) must route through the in-place fast path:
/// the result is correct, no scratch/copy-back corrupts it, and elements
/// outside the view are left untouched. Pins the offset-independent
/// `is_c_dense` routing that replaced the offset-pinned `is_c_contiguous`.
#[test]
fn test_matmul_into_offset_c_dense_view_writes_in_place() {
    let lhs = Array::from_shape_vec([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Array::from_shape_vec([3, 2], vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    // 4x2 backing buffer; rows 0..2 are a sentinel that must stay untouched.
    let mut out = Array::from_shape_vec([4, 2], vec![-1.0f32; 8]).unwrap();

    {
        // Rows 2..4 form a [2,2] C-dense view at offset 4 (strides [2,1]):
        // is_c_dense() == true while is_c_contiguous() == false.
        let mut out_sub = out.slice_mut(&[(2, 4, 1), (0, 2, 1)]).unwrap();
        assert!(out_sub.is_c_dense());
        assert!(!out_sub.is_c_contiguous());
        matmul(&lhs.view(), &rhs.view(), &mut out_sub).unwrap();
    }

    let data = out.storage().as_slice();
    // Sentinel rows untouched (no over-write outside the view).
    assert_eq!(&data[0..4], &[-1.0, -1.0, -1.0, -1.0]);
    // Product written in place at the view's offset.
    assert_eq!(&data[4..8], &[58.0, 64.0, 139.0, 154.0]);
}
