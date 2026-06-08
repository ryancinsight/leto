use half::f16;
use leto::{Array1, Array2, Array3, Storage};
use leto_ops::{add, mapv, matmul, mul, sum_axis_into};
use num_complex::{Complex32, Complex64};

fn assert_close(lhs: f32, rhs: f32) {
    assert!(
        (lhs - rhs).abs() <= 1.0e-5,
        "left {lhs} differs from right {rhs}"
    );
}

#[test]
fn apollo_fft_1d_complex_arrays_support_generation_mapping_and_output_storage() {
    let signal = Array1::from_shape_fn([8], |[index]| {
        let x = index as f64 * 0.25;
        Complex64::new(x.cos(), x.sin())
    });
    let converted = mapv(&signal.view(), |value| {
        Complex32::new(value.re as f32, value.im as f32)
    })
    .unwrap();
    let mut spectrum = Array1::<Complex32>::zeros([8]);

    for index in 0..8 {
        *spectrum.get_mut([index]).unwrap() = *converted.get([index]).unwrap();
    }

    assert_eq!(spectrum.shape(), [8]);
    assert_close(spectrum.get([0]).unwrap().re, 1.0);
    assert_close(spectrum.get([0]).unwrap().im, 0.0);
    assert_close(spectrum.get([4]).unwrap().re, 1.0f32.cos());
    assert_close(spectrum.get([4]).unwrap().im, 1.0f32.sin());
}

#[test]
fn apollo_fft_2d_and_3d_real_fields_support_generated_and_zeroed_outputs() {
    let field_2d = Array2::from_shape_fn([4, 5], |[row, col]| ((row + col) as f64 * 0.3).sin());
    let mut spectrum_2d = Array2::<Complex64>::zeros([4, 5]);
    for row in 0..4 {
        for col in 0..5 {
            let value = *field_2d.get([row, col]).unwrap();
            *spectrum_2d.get_mut([row, col]).unwrap() = Complex64::new(value, 0.0);
        }
    }

    let field_3d = Array3::from_shape_fn([3, 4, 5], |[x, y, z]| ((x + y + z) as f64 * 0.2).cos());
    let recovered_3d = mapv(&field_3d.view(), |value| value).unwrap();

    assert_eq!(spectrum_2d.shape(), [4, 5]);
    assert_eq!(recovered_3d.shape(), [3, 4, 5]);
    assert_eq!(
        recovered_3d.storage().as_slice(),
        field_3d.storage().as_slice()
    );
}

#[test]
fn apollo_precision_conversion_fixture_supports_half_pair_storage() {
    let input = Array1::from_shape_fn([4], |[index]| {
        let value = index as f64 + 0.5;
        Complex64::new(value, -value)
    });

    let half_pairs = mapv(&input.view(), |value| {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    })
    .unwrap();

    assert_eq!(half_pairs.shape(), [4]);
    assert_eq!(half_pairs.get([2]).unwrap()[0], f16::from_f32(2.5));
    assert_eq!(half_pairs.get([2]).unwrap()[1], f16::from_f32(-2.5));
}

#[test]
fn coeus_keepdim_reduction_broadcast_and_elementwise_fixture() {
    let activations =
        Array2::from_shape_vec([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut row_sums = Array2::<f32>::zeros([2, 1]);
    sum_axis_into(&activations.view(), 1, &mut row_sums.view_mut()).unwrap();

    let row_sums_broadcast = row_sums.view().broadcast([2, 3]).unwrap();
    let mut normalized = Array2::<f32>::zeros([2, 3]);
    let mut scaled = Array2::<f32>::zeros([2, 3]);

    add(
        &activations.view(),
        &row_sums_broadcast,
        &mut normalized.view_mut(),
    )
    .unwrap();
    mul(
        &activations.view(),
        &row_sums_broadcast,
        &mut scaled.view_mut(),
    )
    .unwrap();

    assert_eq!(row_sums.storage().as_slice(), &[6.0, 15.0]);
    assert_eq!(
        normalized.storage().as_slice(),
        &[7.0, 8.0, 9.0, 19.0, 20.0, 21.0]
    );
    assert_eq!(
        scaled.storage().as_slice(),
        &[6.0, 12.0, 18.0, 60.0, 75.0, 90.0]
    );
}

#[test]
fn coeus_tensor_matmul_fixture_matches_dense_layer_shape() {
    let batch = Array2::from_shape_vec([2, 3], vec![1.0f32, -2.0, 3.0, 4.0, 0.5, -6.0]).unwrap();
    let weights = Array2::from_shape_vec(
        [3, 4],
        vec![
            0.5, 1.0, -1.0, 2.0, 1.5, -0.5, 0.25, 3.0, -2.0, 4.0, 0.75, -1.5,
        ],
    )
    .unwrap();
    let mut output = Array2::<f32>::zeros([2, 4]);

    matmul(&batch.view(), &weights.view(), &mut output.view_mut()).unwrap();

    assert_eq!(output.shape(), [2, 4]);
    assert_eq!(
        output.storage().as_slice(),
        &[-8.5, 14.0, 0.75, -8.5, 14.75, -20.25, -8.375, 18.5]
    );
}
