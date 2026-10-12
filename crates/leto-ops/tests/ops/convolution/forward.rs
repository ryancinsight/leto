use eunomia::{Bf16, FloatElement, F16};
use leto::{LetoError, Storage};
use leto_ops::{convolution_forward_into, ConvolutionParameters, Scalar};

use super::fixtures::array;

fn forward_contract<T>()
where
    T: Scalar + FloatElement,
{
    let input = array([1, 1, 4], (1..=4).map(T::from_count).collect::<Vec<_>>());
    let weight = array([1, 1, 2], vec![T::from_count(2), T::ONE]);
    let bias = array([1], vec![T::ONE]);
    let mut output = array([1, 1, 3], vec![T::ZERO; 3]);
    let parameters = ConvolutionParameters::new([1], [0], [1]).unwrap();

    convolution_forward_into(
        &input.view(),
        &weight.view(),
        Some(&bias.view()),
        parameters,
        &mut output.view_mut(),
    )
    .unwrap();

    assert_eq!(
        output.storage().as_slice(),
        &[T::from_count(5), T::from_count(8), T::from_count(11)]
    );
}

#[test]
fn forward_contract_all_scalars() {
    forward_contract::<f32>();
    forward_contract::<f64>();
    forward_contract::<F16>();
    forward_contract::<Bf16>();
}

#[test]
fn forward_two_dimensional_padding_and_dilation() {
    let input = array(
        [1, 1, 3, 3],
        (1..=9).map(f32::from_count).collect::<Vec<_>>(),
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
        (1..=8).map(f64::from_count).collect::<Vec<_>>(),
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
