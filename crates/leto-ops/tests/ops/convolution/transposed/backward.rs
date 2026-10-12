use eunomia::{Bf16, FloatElement, F16};
use leto::{LetoError, Storage};
use leto_ops::{
    convolution_transposed_backward_accumulate, Scalar, TransposedConvolutionGradients,
    TransposedConvolutionParameters,
};

use super::super::fixtures::array;

fn transposed_backward_contract<T>()
where
    T: Scalar + FloatElement,
{
    let input = array([1, 1, 2], vec![T::ONE, T::from_count(2)]);
    let weight = array([1, 1, 2], vec![T::from_count(3), T::from_count(4)]);
    let grad_output = array([1, 1, 4], (5..=8).map(T::from_count).collect::<Vec<_>>());
    let mut grad_input = array([1, 1, 2], vec![T::ONE; 2]);
    let mut grad_weight = array([1, 1, 2], vec![T::ONE; 2]);
    let mut grad_bias = array([1], vec![T::ONE]);
    let parameters = TransposedConvolutionParameters::new([2], [0], [0], [1]).unwrap();

    convolution_transposed_backward_accumulate(
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
    .unwrap();

    assert_eq!(
        grad_input.storage().as_slice(),
        &[T::from_count(40), T::from_count(54)]
    );
    assert_eq!(
        grad_weight.storage().as_slice(),
        &[T::from_count(20), T::from_count(23)]
    );
    assert_eq!(grad_bias.storage().as_slice(), &[T::from_count(27)]);
}

#[test]
fn transposed_backward_contract_all_scalars() {
    transposed_backward_contract::<f32>();
    transposed_backward_contract::<f64>();
    transposed_backward_contract::<F16>();
    transposed_backward_contract::<Bf16>();
}

#[test]
fn transposed_backward_two_dimensional_scatter() {
    let input = array([1, 1, 2, 2], vec![1.0_f32, 3.0, 2.0, 4.0]);
    let weight = array([1, 1, 2, 2], vec![1.0_f32, 3.0, 2.0, 4.0]);
    let grad_output = array(
        [1, 1, 3, 3],
        vec![1.0_f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
    );
    let mut grad_input = array([1, 1, 2, 2], vec![0.0_f32; 4]);
    let mut grad_weight = array([1, 1, 2, 2], vec![0.0_f32; 4]);
    let mut grad_bias = array([1], vec![0.0_f32]);
    let parameters = TransposedConvolutionParameters::new([1; 2], [0; 2], [0; 2], [1; 2]).unwrap();

    {
        let input_view = input.transpose([0, 1, 3, 2]).unwrap();
        let weight_view = weight.transpose([0, 1, 3, 2]).unwrap();
        let grad_output_view = grad_output.transpose([0, 1, 3, 2]).unwrap();
        let mut grad_input_view = grad_input.transpose_mut([0, 1, 3, 2]).unwrap();
        let mut grad_weight_view = grad_weight.transpose_mut([0, 1, 3, 2]).unwrap();
        convolution_transposed_backward_accumulate(
            &input_view,
            &weight_view,
            &grad_output_view,
            parameters,
            TransposedConvolutionGradients::new(
                Some(&mut grad_input_view),
                Some(&mut grad_weight_view),
                Some(&mut grad_bias.view_mut()),
            ),
        )
        .unwrap();
    }

    assert_eq!(grad_input.storage().as_slice(), &[37.0, 67.0, 47.0, 77.0]);
    assert_eq!(grad_weight.storage().as_slice(), &[37.0, 67.0, 47.0, 77.0]);
    assert_eq!(grad_bias.storage().as_slice(), &[45.0]);
}

#[test]
fn transposed_backward_three_dimensional_identity_kernel() {
    let input = array(
        [1, 1, 2, 2, 2],
        (1..=8).map(f64::from_count).collect::<Vec<_>>(),
    );
    let weight = array([1, 1, 1, 1, 1], vec![2.0_f64]);
    let grad_output = array([1, 1, 2, 2, 2], vec![1.0_f64; 8]);
    let mut grad_input = array([1, 1, 2, 2, 2], vec![0.0_f64; 8]);
    let mut grad_weight = array([1, 1, 1, 1, 1], vec![0.0_f64]);
    let mut grad_bias = array([1], vec![0.0_f64]);
    let parameters = TransposedConvolutionParameters::new([1; 3], [0; 3], [0; 3], [1; 3]).unwrap();

    convolution_transposed_backward_accumulate(
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
    .unwrap();

    assert_eq!(grad_input.storage().as_slice(), &[2.0; 8]);
    assert_eq!(grad_weight.storage().as_slice(), &[36.0]);
    assert_eq!(grad_bias.storage().as_slice(), &[8.0]);
}

#[test]
fn transposed_backward_output_padding_has_no_weight_contribution() {
    let input = array([1, 1, 2], vec![1.0_f64, 2.0]);
    let weight = array([1, 1, 1], vec![3.0_f64]);
    let grad_output = array([1, 1, 4], vec![5.0_f64, 6.0, 7.0, 8.0]);
    let mut grad_input = array([1, 1, 2], vec![0.0_f64; 2]);
    let mut grad_weight = array([1, 1, 1], vec![0.0_f64]);
    let mut grad_bias = array([1], vec![0.0_f64]);
    let parameters = TransposedConvolutionParameters::new([2], [0], [1], [1]).unwrap();

    convolution_transposed_backward_accumulate(
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
    .unwrap();

    assert_eq!(grad_input.storage().as_slice(), &[15.0, 21.0]);
    assert_eq!(grad_weight.storage().as_slice(), &[19.0]);
    assert_eq!(grad_bias.storage().as_slice(), &[26.0]);
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
