use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    batched_matmul, cumsum, normal_with_seed, scan_axis, uniform_with_seed, zip2_mut_with,
    CumProdOp, ScanDirection,
};

fn arr<const N: usize>(shape: [usize; N], data: Vec<f64>) -> Array<f64, VecStorage<f64>, N> {
    Array::new(Layout::c_contiguous(shape).unwrap(), VecStorage::new(data)).unwrap()
}

#[test]
fn test_batched_matmul_two_batches() {
    // batch 0: [[1,2],[3,4]] x [[1,0],[0,1]] = identity-mul -> same
    // batch 1: [[1,1],[1,1]] x [[2,0],[0,2]] = [[2,2],[2,2]]
    let lhs = arr([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]);
    let rhs = arr([2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let mut out = arr([2, 2, 2], vec![0.0; 8]);
    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn test_batched_matmul_broadcasts_rhs_batch() {
    // rhs batch dim is 1, broadcast across both lhs batches.
    let lhs = arr([2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = arr([1, 2, 1], vec![1.0, 1.0]);
    let mut out = arr([2, 1, 1], vec![0.0; 2]);
    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();
    // [1,2]·[1,1]=3 ; [3,4]·[1,1]=7
    assert_eq!(out.storage().as_slice(), &[3.0, 7.0]);
}

#[test]
fn test_batched_matmul_rejects_shape_mismatch() {
    let lhs = arr([2, 2, 3], vec![0.0; 12]);
    let rhs = arr([2, 2, 2], vec![0.0; 8]);
    let mut out = arr([2, 2, 2], vec![0.0; 8]);
    assert!(batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).is_err());
}

#[test]
fn test_cumsum_forward_axis1() {
    let a = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = cumsum(&a.view(), 1).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn test_cumsum_forward_axis0() {
    let a = arr([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = cumsum(&a.view(), 0).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_scan_reverse_and_cumprod() {
    let a = arr([4], vec![1.0, 2.0, 3.0, 4.0]);
    let suffix = scan_axis::<CumProdOp, _, 1>(&a.view(), 0, ScanDirection::Reverse).unwrap();
    // reverse cumulative product: [24, 24, 12, 4]
    assert_eq!(suffix.storage().as_slice(), &[24.0, 24.0, 12.0, 4.0]);
}

#[test]
fn test_zip2_mut_with_fused_multiply_add() {
    // out = out + a * b, three-operand fused update.
    let mut out = arr([2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let a = arr([2, 2], vec![2.0, 3.0, 4.0, 5.0]);
    let b = arr([2, 2], vec![10.0, 10.0, 10.0, 10.0]);
    zip2_mut_with(&mut out.view_mut(), &a.view(), &b.view(), |o, &x, &y| {
        *o += x * y;
    })
    .unwrap();
    assert_eq!(out.storage().as_slice(), &[21.0, 31.0, 41.0, 51.0]);
}

#[test]
fn test_zip2_mut_with_strided_input() {
    // a is a transposed (strided) view; traversal must follow logical order.
    let mut out = arr([3, 2], vec![0.0; 6]);
    let a_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a = a_src.transpose([1, 0]).unwrap(); // logical [[1,4],[2,5],[3,6]]
    let b = arr([3, 2], vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
    zip2_mut_with(&mut out.view_mut(), &a, &b.view(), |o, &x, &y| {
        *o = x + y;
    })
    .unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[101.0, 104.0, 102.0, 105.0, 103.0, 106.0]
    );
}

#[test]
fn test_uniform_is_deterministic_and_in_range() {
    let a = uniform_with_seed([1000], -2.0, 5.0, 42).unwrap();
    let b = uniform_with_seed([1000], -2.0, 5.0, 42).unwrap();
    assert_eq!(a.storage().as_slice(), b.storage().as_slice());
    for &v in a.storage().as_slice() {
        assert!((-2.0..5.0).contains(&v), "out of range: {v}");
    }
    // Different seed yields a different stream.
    let c = uniform_with_seed([1000], -2.0, 5.0, 43).unwrap();
    assert_ne!(a.storage().as_slice(), c.storage().as_slice());
}

#[test]
fn test_uniform_mean_matches_closed_form() {
    let n = 100_000usize;
    let a = uniform_with_seed([n], 0.0, 1.0, 7).unwrap();
    let mean: f64 = a.storage().as_slice().iter().sum::<f64>() / n as f64;
    // Closed-form mean of U(0,1) is 0.5; sampling error well under 0.02 at this n.
    assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
}

#[test]
fn test_normal_mean_and_std_match_closed_form() {
    let n = 100_000usize;
    let a = normal_with_seed([n], 1.0, 2.0, 11).unwrap();
    let data = a.storage().as_slice();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    assert!((mean - 1.0).abs() < 0.05, "mean {mean}");
    assert!((std - 2.0).abs() < 0.05, "std {std}");
}
