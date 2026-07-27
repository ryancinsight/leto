use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    add, binary_map, div, indexed_zip2_mut_with, indexed_zip_mut_with, map, map_into, mapv, mul,
    scalar_map, sub, unary_map, zip_mut_with, AddOp, EqOp, ErfOp, ErfcOp, GeOp, GtOp, LeOp,
    LgammaOp, LtOp, MulOp, NeOp,
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
    zip_mut_with(&mut dest.view_mut(), &arr.view(), |d, &s| {
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

    indexed_zip_mut_with(
        &mut lhs.view_mut(),
        &rhs.view(),
        |[row, col], left, right| {
            *left = *right + (row as i32) * 100 + (col as i32);
        },
    )
    .unwrap();

    assert_eq!(lhs.storage().as_slice(), &[10, 21, 32, 140, 151, 162]);
}

#[test]
fn test_indexed_zip2_mut_with_handles_strided_transposed_views() {
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

    indexed_zip2_mut_with(&mut out, &a_t, &b_t, |[row, col], left, av, bv| {
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
