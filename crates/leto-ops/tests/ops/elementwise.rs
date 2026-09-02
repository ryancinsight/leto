#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    add, binary_map, div, indexed_zip_mut_with, map, map_into, mapv, mul, scalar_map, sub,
    unary_map, zip_mut_with, AddOp, EqOp, ErfOp, ErfcOp, GeOp, GtOp, LeOp, LgammaOp, LtOp, MulOp,
    NeOp,
};

fn assert_scalar_supertrait<T>()
where
    T: leto_ops::Scalar + eunomia::NumericElement,
{
}

fn assert_real_supertrait<T>()
where
    T: leto_ops::RealScalar + eunomia::FloatElement,
{
}

#[test]
fn scalar_traits_are_eunomia_extensions() {
    assert_scalar_supertrait::<f32>();
    assert_scalar_supertrait::<f64>();
    assert_scalar_supertrait::<eunomia::F16>();
    assert_scalar_supertrait::<eunomia::Bf16>();
    assert_scalar_supertrait::<i32>();
    assert_scalar_supertrait::<u64>();
    assert_scalar_supertrait::<isize>();
    assert_scalar_supertrait::<usize>();

    assert_real_supertrait::<f32>();
    assert_real_supertrait::<f64>();
    assert_real_supertrait::<eunomia::F16>();
    assert_real_supertrait::<eunomia::Bf16>();

    assert_eq!(<f64 as leto_ops::Scalar>::from_usize(3), 3.0);
    assert_eq!(<isize as leto_ops::Scalar>::from_usize(5), 5_isize);
    assert_eq!(<usize as leto_ops::Scalar>::from_usize(6), 6_usize);
    assert_eq!(
        <eunomia::F16 as leto_ops::Scalar>::from_usize(4),
        eunomia::F16::from_f32(4.0)
    );
    assert_eq!(
        <eunomia::Bf16 as leto_ops::Scalar>::from_usize(4),
        eunomia::Bf16::from_f32(4.0)
    );
}

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

#[test]
fn special_unary_ops_match_eunomia_reference_values() {
    let input = Array::from_shape_vec([4], vec![0.0f64, 0.5, 1.0, 5.0]).unwrap();

    let erf = unary_map(ErfOp, &input.view()).unwrap();
    let erfc = unary_map(ErfcOp, &input.view()).unwrap();
    let lgamma = unary_map(LgammaOp, &input.view()).unwrap();

    let erf_expected = [
        0.0,
        0.520_499_877_813_046_5,
        0.842_700_792_949_714_9,
        0.999_999_999_998_462_6,
    ];
    let erfc_expected = [
        1.0,
        0.479_500_122_186_953_5,
        0.157_299_207_050_285_13,
        1.537_459_794_428_034_7e-12,
    ];
    let lgamma_expected = [f64::INFINITY, 0.572_364_942_924_700_1, 0.0, 24.0_f64.ln()];

    for index in 0..4 {
        assert!(
            (erf.storage().as_slice()[index] - erf_expected[index]).abs() <= 2.0e-15,
            "erf[{index}]"
        );
        assert!(
            (erfc.storage().as_slice()[index] - erfc_expected[index]).abs() <= 2.0e-15,
            "erfc[{index}]"
        );
        if lgamma_expected[index].is_infinite() {
            assert!(
                lgamma.storage().as_slice()[index].is_infinite(),
                "lgamma[{index}]"
            );
        } else {
            assert!(
                (lgamma.storage().as_slice()[index] - lgamma_expected[index]).abs() <= 2.0e-15,
                "lgamma[{index}]"
            );
        }
    }
}

#[test]
fn test_mapv_allocates_c_contiguous_output_with_explicit_conversion() {
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1.25f64, 2.5, 3.75, 4.0])).unwrap();

    let output = mapv(&input.view(), |value| value as f32).unwrap();

    assert_eq!(output.shape(), [2, 2]);
    assert!(output.layout().is_c_contiguous());
    assert_eq!(output.storage().as_slice(), &[1.25f32, 2.5, 3.75, 4.0]);
}

#[test]
fn test_map_into_handles_strided_transposed_input() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1i32, 2, 3, 4, 5, 6])).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let out_layout = Layout::c_contiguous([3, 2]).unwrap();
    let mut output = Array::new(out_layout, VecStorage::fill(6, 0i32)).unwrap();

    map_into(&transposed, &mut output.view_mut(), |value| value * 10).unwrap();

    assert_eq!(output.storage().as_slice(), &[10, 40, 20, 50, 30, 60]);
}

#[test]
fn test_map_into_handles_cache_line_transposed_input() {
    let n = 16usize;
    let input =
        Array::from_shape_vec([n, n], (0..n * n).map(|value| value as f64).collect()).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let mut output = Array::zeros([n, n]);

    map_into(&transposed, &mut output.view_mut(), |value| {
        value * 2.0 + 1.0
    })
    .unwrap();

    let expected = (0..n)
        .flat_map(|row| (0..n).map(move |col| ((col * n + row) as f64) * 2.0 + 1.0))
        .collect::<Vec<_>>();
    assert_eq!(output.storage().as_slice(), expected.as_slice());
}

#[test]
fn test_map_into_strided_zero_sized_input() {
    let input = Array::from_shape_vec([2, 2], vec![(); 4]).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let mut output = Array::zeros([2, 2]);

    map_into(&transposed, &mut output.view_mut(), |_| 7usize).unwrap();

    assert_eq!(output.storage().as_slice(), &[7, 7, 7, 7]);
}

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

/// `F16` takes the hermes route like the machine floats. The elementwise
/// operations round once from an `f32` intermediate on every backend, as the
/// scalar `F16` operators do, so they are bitwise equal; the reductions may
/// reorder, so they are held to `n · u` with `u = 2⁻¹¹` on the summed
/// magnitudes (Higham ASNA 2nd ed. §4.2, naive summation), and min/max are
/// exact selections.
#[test]
fn f16_slice_operations_route_through_hermes_and_match_scalar_semantics() {
    use eunomia::F16;
    use leto_ops::domain::strategy::{SimdOperations, SimdStrategy};
    let n = 259usize;
    let a: Vec<F16> = (0..n)
        .map(|i| F16::from_f32(((i * 37 % 101) as f32 - 50.0) / 8.0))
        .collect();
    let b: Vec<F16> = (0..n)
        .map(|i| F16::from_f32(((i * 53 % 97) as f32 - 48.0) / 16.0 + 0.25))
        .collect();
    let mut out = vec![F16::from_f32(0.0); n];
    for (name, op, scalar) in [
        (
            "add",
            <SimdStrategy as SimdOperations<F16>>::add_slice
                as fn(&[F16], &[F16], &mut [F16]) -> Result<(), &'static str>,
            (|x: F16, y: F16| x + y) as fn(F16, F16) -> F16,
        ),
        (
            "sub",
            <SimdStrategy as SimdOperations<F16>>::sub_slice,
            |x, y| x - y,
        ),
        (
            "mul",
            <SimdStrategy as SimdOperations<F16>>::mul_slice,
            |x, y| x * y,
        ),
        (
            "div",
            <SimdStrategy as SimdOperations<F16>>::div_slice,
            |x, y| x / y,
        ),
    ] {
        op(&a, &b, &mut out)
            .unwrap_or_else(|e| panic!("{name}: F16 declined by the strategy: {e}"));
        for (i, ((&x, &y), &got)) in a.iter().zip(&b).zip(&out).enumerate() {
            let want = scalar(x, y);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name}[{i}]: hermes {got:?} vs scalar {want:?}"
            );
        }
    }
    let sum = <SimdStrategy as SimdOperations<F16>>::sum_slice(&a).expect("F16 sum routed");
    let dot = <SimdStrategy as SimdOperations<F16>>::dot_slice(&a, &b).expect("F16 dot routed");
    let sum_ref: f64 = a.iter().map(|x| f64::from(x.to_f32())).sum();
    let dot_ref: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(x.to_f32()) * f64::from(y.to_f32()))
        .sum();
    let u = 2f64.powi(-11);
    let sum_scale: f64 = a.iter().map(|x| f64::from(x.to_f32()).abs()).sum();
    let dot_scale: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f64::from(x.to_f32()) * f64::from(y.to_f32())).abs())
        .sum();
    assert!(
        (f64::from(sum.to_f32()) - sum_ref).abs() <= (n as f64) * u * sum_scale,
        "sum {sum:?} vs {sum_ref}"
    );
    assert!(
        (f64::from(dot.to_f32()) - dot_ref).abs() <= (n as f64) * u * dot_scale,
        "dot {dot:?} vs {dot_ref}"
    );
    let min = <SimdStrategy as SimdOperations<F16>>::min_slice(&a).expect("F16 min routed");
    let max = <SimdStrategy as SimdOperations<F16>>::max_slice(&a).expect("F16 max routed");
    let min_ref = a.iter().copied().fold(
        F16::from_f32(f32::INFINITY),
        |m, x| if x < m { x } else { m },
    );
    let max_ref =
        a.iter().copied().fold(
            F16::from_f32(f32::NEG_INFINITY),
            |m, x| if x > m { x } else { m },
        );
    assert_eq!(min.to_bits(), min_ref.to_bits());
    assert_eq!(max.to_bits(), max_ref.to_bits());

    // The public op reaches the same route through the `Scalar` impl.
    let arr_a = Array::from_shape_vec([n], a.clone()).unwrap();
    let arr_b = Array::from_shape_vec([n], b.clone()).unwrap();
    let mut via_public = Array::from_shape_vec([n], vec![F16::from_f32(0.0); n]).unwrap();
    add(&arr_a.view(), &arr_b.view(), &mut via_public.view_mut()).unwrap();
    for (i, ((&x, &y), &got)) in a
        .iter()
        .zip(&b)
        .zip(via_public.storage().as_slice())
        .enumerate()
    {
        assert_eq!(got.to_bits(), (x + y).to_bits(), "public add[{i}]");
    }
}
