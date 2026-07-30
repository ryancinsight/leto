use core::fmt::Debug;

use eunomia::{Bf16, F16};
use leto::{Array, Layout, LetoError, Storage, VecStorage};
use leto_ops::{
    ConvolutionParameters, Scalar, TransposedConvolutionGradients, TransposedConvolutionParameters,
    convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
};

fn array<T: Clone, const R: usize>(
    shape: [usize; R],
    values: Vec<T>,
) -> Array<T, VecStorage<T>, R> {
    Array::new(
        Layout::c_contiguous(shape).expect("test shape is representable"),
        VecStorage::new(values),
    )
    .expect("test storage matches its shape")
}

fn forward_contract<T>()
where
    T: Scalar + Clone + Debug + PartialEq,
{
    let input = array([1, 1, 4], (1..=4).map(T::from_usize).collect::<Vec<_>>());
    let weight = array([1, 1, 2], vec![T::from_usize(2), T::from_usize(1)]);
    let bias = array([1], vec![T::from_usize(1)]);
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
        &[T::from_usize(5), T::from_usize(8), T::from_usize(11)]
    );
}

fn backward_contract<T>()
where
    T: Scalar + Clone + Debug + PartialEq,
{
    let input = array([1, 1, 3], (1..=3).map(T::from_usize).collect::<Vec<_>>());
    let weight = array([1, 1, 2], vec![T::from_usize(2), T::from_usize(3)]);
    let grad_output = array([1, 1, 2], vec![T::from_usize(5), T::from_usize(7)]);
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
        &[T::from_usize(11), T::from_usize(30), T::from_usize(22)]
    );
    assert_eq!(
        grad_weight.storage().as_slice(),
        &[T::from_usize(20), T::from_usize(32)]
    );
    assert_eq!(grad_bias.storage().as_slice(), &[T::from_usize(13)]);
}

fn transposed_contract<T>()
where
    T: Scalar + Clone + Debug + PartialEq,
{
    let input = array([1, 1, 2], vec![T::from_usize(1), T::from_usize(2)]);
    let weight = array([1, 1, 2], vec![T::from_usize(3), T::from_usize(4)]);
    let bias = array([1], vec![T::ONE]);
    let mut output = array([1, 1, 4], vec![T::ZERO; 4]);
    let parameters = TransposedConvolutionParameters::new([2], [0], [0], [1]).unwrap();

    convolution_transposed_forward_into(
        &input.view(),
        &weight.view(),
        Some(&bias.view()),
        parameters,
        &mut output.view_mut(),
    )
    .unwrap();

    assert_eq!(
        output.storage().as_slice(),
        &[
            T::from_usize(4),
            T::from_usize(5),
            T::from_usize(7),
            T::from_usize(9),
        ]
    );
}

fn transposed_backward_contract<T>()
where
    T: Scalar + Clone + Debug + PartialEq,
{
    let input = array([1, 1, 2], vec![T::from_usize(1), T::from_usize(2)]);
    let weight = array([1, 1, 2], vec![T::from_usize(3), T::from_usize(4)]);
    let grad_output = array([1, 1, 4], (5..=8).map(T::from_usize).collect::<Vec<_>>());
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
        &[T::from_usize(40), T::from_usize(54)]
    );
    assert_eq!(
        grad_weight.storage().as_slice(),
        &[T::from_usize(20), T::from_usize(23)]
    );
    assert_eq!(grad_bias.storage().as_slice(), &[T::from_usize(27)]);
}

#[test]
fn forward_contract_all_scalars() {
    forward_contract::<f32>();
    forward_contract::<f64>();
    forward_contract::<F16>();
    forward_contract::<Bf16>();
}

#[test]
fn backward_contract_all_scalars() {
    backward_contract::<f32>();
    backward_contract::<f64>();
    backward_contract::<F16>();
    backward_contract::<Bf16>();
}

#[test]
fn transposed_contract_all_scalars() {
    transposed_contract::<f32>();
    transposed_contract::<f64>();
    transposed_contract::<F16>();
    transposed_contract::<Bf16>();
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
        (1..=8).map(f64::from_usize).collect::<Vec<_>>(),
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
fn transposed_three_dimensional_identity_kernel() {
    let input = array(
        [1, 1, 2, 2, 2],
        (1..=8).map(f32::from_usize).collect::<Vec<_>>(),
    );
    let weight = array([1, 1, 1, 1, 1], vec![2.0_f32]);
    let mut output = array([1, 1, 2, 2, 2], vec![0.0_f32; 8]);
    let parameters = TransposedConvolutionParameters::new([1; 3], [0; 3], [0; 3], [1; 3]).unwrap();

    convolution_transposed_forward_into(
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
fn transposed_two_dimensional_scatter_matches_reference() {
    let input = array([1, 1, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
    let weight = array([1, 1, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
    let mut output = array([1, 1, 3, 3], vec![0.0_f32; 9]);
    let parameters = TransposedConvolutionParameters::new([1; 2], [0; 2], [0; 2], [1; 2]).unwrap();

    convolution_transposed_forward_into(
        &input.view(),
        &weight.view(),
        None,
        parameters,
        &mut output.view_mut(),
    )
    .unwrap();

    assert_eq!(
        output.storage().as_slice(),
        &[1.0, 4.0, 4.0, 6.0, 20.0, 16.0, 9.0, 24.0, 16.0]
    );
}

#[test]
fn transposed_strided_views_match_logical_reference() {
    let input = array([1, 1, 2, 2], vec![1.0_f32, 3.0, 2.0, 4.0]);
    let weight = array([1, 1, 2, 2], vec![1.0_f32, 3.0, 2.0, 4.0]);
    let input_view = input.transpose([0, 1, 3, 2]).unwrap();
    let weight_view = weight.transpose([0, 1, 3, 2]).unwrap();
    let mut output = array([1, 1, 3, 3], vec![0.0_f32; 9]);
    let parameters = TransposedConvolutionParameters::new([1; 2], [0; 2], [0; 2], [1; 2]).unwrap();

    {
        let mut output_view = output.transpose_mut([0, 1, 3, 2]).unwrap();
        convolution_transposed_forward_into(
            &input_view,
            &weight_view,
            None,
            parameters,
            &mut output_view,
        )
        .unwrap();
    }

    assert_eq!(
        output.storage().as_slice(),
        &[1.0, 6.0, 9.0, 4.0, 20.0, 24.0, 4.0, 16.0, 16.0]
    );
}

#[test]
fn transposed_output_padding_changes_only_shape() {
    let input = array([1, 1, 2], vec![1.0_f64, 2.0]);
    let weight = array([1, 1, 1], vec![3.0_f64]);
    let mut output = array([1, 1, 4], vec![17.0_f64; 4]);
    let parameters = TransposedConvolutionParameters::new([2], [0], [1], [1]).unwrap();

    convolution_transposed_forward_into(
        &input.view(),
        &weight.view(),
        None,
        parameters,
        &mut output.view_mut(),
    )
    .unwrap();

    assert_eq!(output.storage().as_slice(), &[3.0, 0.0, 6.0, 0.0]);
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
fn invalid_transposed_output_shape_preserves_output() {
    let input = array([1, 1, 2], vec![1.0_f32, 2.0]);
    let weight = array([1, 1, 2], vec![3.0_f32, 4.0]);
    let mut output = array([1, 1, 3], vec![17.0_f32; 3]);
    let parameters = TransposedConvolutionParameters::new([2], [0], [0], [1]).unwrap();

    let error = convolution_transposed_forward_into(
        &input.view(),
        &weight.view(),
        None,
        parameters,
        &mut output.view_mut(),
    )
    .expect_err("three outputs violate the derived four-output contract");

    assert_eq!(
        error,
        LetoError::ShapeMismatch {
            lhs: vec![1, 1, 3],
            rhs: vec![1, 1, 4],
        }
    );
    assert_eq!(output.storage().as_slice(), &[17.0; 3]);
}

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
