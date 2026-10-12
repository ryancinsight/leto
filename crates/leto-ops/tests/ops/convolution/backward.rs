use eunomia::{Bf16, FloatElement, F16};
use leto::{LetoError, Storage};
use leto_ops::{convolution_backward_accumulate, ConvolutionParameters, Scalar};

use super::fixtures::array;

fn backward_contract<T>()
where
    T: Scalar + FloatElement,
{
    let input = array([1, 1, 3], (1..=3).map(T::from_count).collect::<Vec<_>>());
    let weight = array([1, 1, 2], vec![T::from_count(2), T::from_count(3)]);
    let grad_output = array([1, 1, 2], vec![T::from_count(5), T::from_count(7)]);
    let mut grad_input = array([1, 1, 3], vec![T::ONE; 3]);
    let mut grad_weight = array([1, 1, 2], vec![T::ONE; 2]);
    let mut grad_bias = array([1], vec![T::ONE]);
    let parameters = ConvolutionParameters::new([1], [0], [1]).unwrap();

    convolution_backward_accumulate(
        &input.view(),
        &weight.view(),
        &grad_output.view(),
        parameters,
        Some(&mut grad_input.view_mut()),
        Some(&mut grad_weight.view_mut()),
        Some(&mut grad_bias.view_mut()),
    )
    .unwrap();

    assert_eq!(
        grad_input.storage().as_slice(),
        &[T::from_count(11), T::from_count(30), T::from_count(22)]
    );
    assert_eq!(
        grad_weight.storage().as_slice(),
        &[T::from_count(20), T::from_count(32)]
    );
    assert_eq!(grad_bias.storage().as_slice(), &[T::from_count(13)]);
}

#[test]
fn backward_contract_all_scalars() {
    backward_contract::<f32>();
    backward_contract::<f64>();
    backward_contract::<F16>();
    backward_contract::<Bf16>();
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
