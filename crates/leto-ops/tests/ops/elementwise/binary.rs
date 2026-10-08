//! Binary map, comparison and broadcast semantics.

use super::*;

#[test]
fn test_elementwise_binary_ops() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let a_storage = VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_storage = VecStorage::new(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let out_storage = VecStorage::fill(6, 0.0f32);

    let a = Array::new(layout, a_storage).unwrap();
    let b = Array::new(layout, b_storage).unwrap();
    let mut out = Array::new(layout, out_storage).unwrap();

    add(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    );

    // For subtraction, write into out2
    let out2_storage = VecStorage::fill(6, 0.0f32);
    let mut out2 = Array::new(layout, out2_storage).unwrap();
    sub(&out.view(), &a.view(), &mut out2.view_mut()).unwrap();
    assert_eq!(
        out2.storage().as_slice(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    // For multiplication, write into out3
    let out3_storage = VecStorage::fill(6, 0.0f32);
    let mut out3 = Array::new(layout, out3_storage).unwrap();
    mul(&out2.view(), &a.view(), &mut out3.view_mut()).unwrap();
    assert_eq!(
        out3.storage().as_slice(),
        &[10.0, 40.0, 90.0, 160.0, 250.0, 360.0]
    );

    // For division, write into out4
    let out4_storage = VecStorage::fill(6, 0.0f32);
    let mut out4 = Array::new(layout, out4_storage).unwrap();
    div(&out3.view(), &a.view(), &mut out4.view_mut()).unwrap();
    assert_eq!(
        out4.storage().as_slice(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn test_binary_map_zst_operation_entry_point() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![5.0f32, 6.0, 7.0, 8.0])).unwrap();
    let mut out = Array::new(layout, VecStorage::fill(4, 0.0f32)).unwrap();

    binary_map::<AddOp, _, 1>(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[6.0, 8.0, 10.0, 12.0]);

    binary_map::<MulOp, _, 1>(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_binary_map_strided_transposed_views() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let a = Array::new(
        layout,
        VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let b = Array::new(
        layout,
        VecStorage::new(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]),
    )
    .unwrap();
    let out_layout = Layout::c_contiguous([3, 2]).unwrap();
    let mut out = Array::new(out_layout, VecStorage::fill(6, 0.0f32)).unwrap();

    let a_t = a.transpose([1, 0]).unwrap();
    let b_t = b.transpose([1, 0]).unwrap();
    add(&a_t, &b_t, &mut out.view_mut()).unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[11.0, 44.0, 22.0, 55.0, 33.0, 66.0]
    );
}

#[test]
fn test_binary_map_broadcasts_inputs_to_output_shape() {
    let lhs = Array::from_shape_vec([2, 1], vec![1.0f32, 10.0]).unwrap();
    let rhs = Array::from_shape_vec([1, 3], vec![2.0f32, 3.0, 4.0]).unwrap();
    let mut out = Array::zeros([2, 3]);

    add(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    assert_eq!(out.storage().as_slice(), &[3.0, 4.0, 5.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_binary_comparisons_broadcast_inputs_to_output_shape() {
    let lhs = Array::from_shape_vec([2, 1], vec![1.0f32, 10.0]).unwrap();
    let rhs = Array::from_shape_vec([1, 3], vec![1.0f32, 3.0, 10.0]).unwrap();

    macro_rules! assert_comparison {
        ($operation:ty, $expected:expr) => {
            let mut out = Array::zeros([2, 3]);
            binary_map::<$operation, _, 2>(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();
            assert_eq!(out.storage().as_slice(), $expected);
        };
    }

    assert_comparison!(EqOp, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_comparison!(NeOp, &[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    assert_comparison!(LtOp, &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert_comparison!(GtOp, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    assert_comparison!(LeOp, &[1.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    assert_comparison!(GeOp, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_binary_map_broadcasts_strided_input_to_output_shape() {
    let lhs_base = Array::from_shape_vec([3, 2], vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
    let lhs = lhs_base
        .transpose([1, 0])
        .unwrap()
        .slice(&[(0, 1, 1), (0, 3, 1)])
        .unwrap();
    let rhs = Array::from_shape_vec([2, 3], vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let mut out = Array::zeros([2, 3]);

    mul(&lhs, &rhs.view(), &mut out.view_mut()).unwrap();

    assert_eq!(out.storage().as_slice(), &[2.0, 6.0, 12.0, 5.0, 12.0, 21.0]);
}

#[test]
fn test_map_into_uses_caller_owned_output() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1.0f32, -2.0, 3.5, 4.0])).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(4, 0.0f32)).unwrap();

    map_into(&input.view(), &mut output.view_mut(), |value| value * value).unwrap();

    assert_eq!(output.storage().as_slice(), &[1.0, 4.0, 12.25, 16.0]);
}

#[test]
fn test_binary_map_same_order_f_dense_operands_match_reference() {
    // Three F-dense operands with identical strides take the memory-order
    // slice fast path; values must match the logical per-element reference.
    let f_layout = Layout::f_contiguous([2, 3]).unwrap();
    let lhs = Array::new(
        f_layout,
        VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let rhs = Array::new(
        f_layout,
        VecStorage::new(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]),
    )
    .unwrap();
    let mut out = Array::new(f_layout, VecStorage::fill(6, 0.0f32)).unwrap();

    binary_map::<AddOp, f32, 2>(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    for r in 0..2 {
        for c in 0..3 {
            let expected = *lhs.get([r, c]).unwrap() + *rhs.get([r, c]).unwrap();
            assert_eq!(*out.get([r, c]).unwrap(), expected, "diverges at [{r},{c}]");
        }
    }
}

#[test]
fn test_outputs_reject_non_injective_layouts() {
    // Shape [2, 2], strides [1, 1] is zero-stride-free yet non-injective:
    // logical (0, 1) and (1, 0) share physical offset 1. Serial kernels would
    // double-apply and parallel kernels would race, so every mutable-output
    // entry point must reject it with a typed error.
    let aliased = Layout::try_new([2, 2], [1, 1], 0).unwrap();
    let dense = Layout::c_contiguous([2, 2]).unwrap();
    let input = Array::new(dense, VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0])).unwrap();

    let mut out = Array::new(aliased, VecStorage::fill(4, 0.0f32)).unwrap();
    assert!(map_into(&input.view(), &mut out.view_mut(), |v| v + 1.0).is_err());
    assert!(
        binary_map::<AddOp, f32, 2>(&input.view(), &input.view(), &mut out.view_mut()).is_err()
    );
}
