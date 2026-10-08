//! Special unary maps and caller-owned output handling.

use super::*;

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
fn explicit_cache_geometry_preserves_strided_unary_and_binary_semantics() {
    let n = 32usize;
    let input = Array::from_shape_vec(
        [n, n],
        (0..n * n).map(|value| value as f64 + 0.25).collect(),
    )
    .unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let rhs = Array::from_shape_vec([n, n], vec![2.0f64; n * n]).unwrap();
    let narrow = CacheGeometry::with_cache_line_bytes(64).unwrap();
    let wide = CacheGeometry::with_cache_line_bytes(128).unwrap();

    let mut unary_narrow = Array::zeros([n, n]);
    let mut unary_wide = Array::zeros([n, n]);
    map_into_with_cache_geometry(
        &transposed,
        &mut unary_narrow.view_mut(),
        |value| value.mul_add(2.0, 1.0),
        narrow,
    )
    .unwrap();
    map_into_with_cache_geometry(
        &transposed,
        &mut unary_wide.view_mut(),
        |value| value.mul_add(2.0, 1.0),
        wide,
    )
    .unwrap();
    assert_eq!(
        unary_narrow.storage().as_slice(),
        unary_wide.storage().as_slice()
    );

    let mut binary_narrow = Array::zeros([n, n]);
    let mut binary_wide = Array::zeros([n, n]);
    binary_map_with_cache_geometry::<AddOp, _, 2>(
        &transposed,
        &rhs.view(),
        &mut binary_narrow.view_mut(),
        narrow,
    )
    .unwrap();
    binary_map_with_cache_geometry::<AddOp, _, 2>(
        &transposed,
        &rhs.view(),
        &mut binary_wide.view_mut(),
        wide,
    )
    .unwrap();
    assert_eq!(
        binary_narrow.storage().as_slice(),
        binary_wide.storage().as_slice()
    );
}

#[test]
fn test_map_into_strided_zero_sized_input() {
    let input = Array::from_shape_vec([2, 2], vec![(); 4]).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let mut output = Array::zeros([2, 2]);

    map_into(&transposed, &mut output.view_mut(), |_| 7usize).unwrap();

    assert_eq!(output.storage().as_slice(), &[7, 7, 7, 7]);
}
