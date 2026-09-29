//! Padding, shape-validation and rejection-path tests.

use super::*;

#[test]
fn forward_two_dimensional_padding_and_dilation() {
    let input = array(
        [1, 1, 3, 3],
        (1..=9).map(f32::from_usize).collect::<Vec<_>>(),
    );
    let weight = array([1, 1, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
    let mut output = array([1, 1, 3, 3], vec![0.0_f32; 9]);
    let parameters = ConvolutionParameters::new([1, 1], [1, 1], [2, 2]).unwrap();

    convolution_forward_into(
        &input.view(),
        &weight.view(),
        None,
        parameters,
        &mut output.view_mut(),
    )
    .unwrap();

    assert_eq!(
        output.storage().as_slice(),
        &[20.0, 36.0, 15.0, 36.0, 64.0, 26.0, 10.0, 16.0, 5.0]
    );
}

#[test]
fn forward_three_dimensional_identity_kernel() {
    let input = array(
        [1, 1, 2, 2, 2],
        (1..=8).map(f64::from_usize).collect::<Vec<_>>(),
    );
    let weight = array([1, 1, 1, 1, 1], vec![2.0_f64]);
    let mut output = array([1, 1, 2, 2, 2], vec![0.0_f64; 8]);
    let parameters = ConvolutionParameters::new([1; 3], [0; 3], [1; 3]).unwrap();

    convolution_forward_into(
        &input.view(),
        &weight.view(),
        None,
        parameters,
        &mut output.view_mut(),
    )
    .unwrap();

    assert_eq!(
        output.storage().as_slice(),
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    );
}

#[test]
fn invalid_output_shape_preserves_output() {
    let input = array([1, 1, 3], vec![1.0_f32, 2.0, 3.0]);
    let weight = array([1, 1, 2], vec![1.0_f32, 1.0]);
    let mut output = array([1, 1, 3], vec![17.0_f32; 3]);
    let parameters = ConvolutionParameters::new([1], [0], [1]).unwrap();

    let error = convolution_forward_into(
        &input.view(),
        &weight.view(),
        None,
        parameters,
        &mut output.view_mut(),
    )
    .expect_err("three outputs violate the derived two-output contract");

    assert_eq!(
        error,
        LetoError::ShapeMismatch {
            lhs: vec![1, 1, 3],
            rhs: vec![1, 1, 2],
        }
    );
    assert_eq!(output.storage().as_slice(), &[17.0; 3]);
}

#[test]
fn zero_stride_is_rejected_at_parameter_construction() {
    assert_eq!(
        ConvolutionParameters::new([0], [0], [1]),
        Err(LetoError::InvalidInput(
            "convolution stride must be nonzero".to_string()
        ))
    );
}

#[test]
fn invalid_backward_target_preserves_all_gradients() {
    let input = array([1, 1, 3], vec![1.0_f32, 2.0, 3.0]);
    let weight = array([1, 1, 2], vec![2.0_f32, 3.0]);
    let grad_output = array([1, 1, 2], vec![5.0_f32, 7.0]);
    let mut grad_input = array([1, 1, 3], vec![17.0_f32; 3]);
    let mut grad_weight = array([1, 1, 3], vec![19.0_f32; 3]);
    let mut grad_bias = array([1], vec![23.0_f32]);
    let parameters = ConvolutionParameters::new([1], [0], [1]).unwrap();

    let error = convolution_backward_accumulate(
        &input.view(),
        &weight.view(),
        &grad_output.view(),
        parameters,
        Some(&mut grad_input.view_mut()),
        Some(&mut grad_weight.view_mut()),
        Some(&mut grad_bias.view_mut()),
    )
    .expect_err("the weight gradient target has the wrong shape");

    assert_eq!(
        error,
        LetoError::ShapeMismatch {
            lhs: vec![1, 1, 3],
            rhs: vec![1, 1, 2],
        }
    );
    assert_eq!(grad_input.storage().as_slice(), &[17.0; 3]);
    assert_eq!(grad_weight.storage().as_slice(), &[19.0; 3]);
    assert_eq!(grad_bias.storage().as_slice(), &[23.0]);
}

#[test]
fn invalid_transposed_backward_target_preserves_all_gradients() {
    let input = array([1, 1, 2], vec![1.0_f32, 2.0]);
    let weight = array([1, 1, 2], vec![3.0_f32, 4.0]);
    let grad_output = array([1, 1, 4], vec![5.0_f32, 6.0, 7.0, 8.0]);
    let mut grad_input = array([1, 1, 2], vec![17.0_f32; 2]);
    let mut grad_weight = array([1, 1, 3], vec![19.0_f32; 3]);
    let mut grad_bias = array([1], vec![23.0_f32]);
    let parameters = TransposedConvolutionParameters::new([2], [0], [0], [1]).unwrap();

    let error = convolution_transposed_backward_accumulate(
        &input.view(),
        &weight.view(),
        &grad_output.view(),
        parameters,
        TransposedConvolutionGradients::new(
            Some(&mut grad_input.view_mut()),
            Some(&mut grad_weight.view_mut()),
            Some(&mut grad_bias.view_mut()),
        ),
    )
    .expect_err("the weight gradient target has the wrong shape");

    assert_eq!(
        error,
        LetoError::ShapeMismatch {
            lhs: vec![1, 1, 3],
            rhs: vec![1, 1, 2],
        }
    );
    assert_eq!(grad_input.storage().as_slice(), &[17.0; 2]);
    assert_eq!(grad_weight.storage().as_slice(), &[19.0; 3]);
    assert_eq!(grad_bias.storage().as_slice(), &[23.0]);
}
