use super::*;

#[test]
fn validation_is_typed_and_failure_atomic() {
    let query = array([1, 1, 1], vec![1.0_f64]);
    let key = array([1, 1, 1], vec![1.0_f64]);
    let value = array([1, 1, 1], vec![1.0_f64]);
    let mut output = array([1, 1, 2], vec![7.0_f64; 2]);
    let mut weights = array([1, 1, 1], vec![8.0_f64]);

    let error = scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect_err("wrong output shape must fail");
    assert_eq!(
        error,
        AttentionError::Shape {
            operand: AttentionOperand::Output,
            expected: [1, 1, 1],
            actual: [1, 1, 2],
        }
    );
    assert_eq!(output.storage().as_slice(), &[7.0, 7.0]);
    assert_eq!(weights.storage().as_slice(), &[8.0]);
}

#[test]
fn non_finite_input_and_invalid_mask_are_typed_failures() {
    let query = array([1, 1, 1], vec![f64::NAN]);
    let key = array([1, 1, 1], vec![1.0_f64]);
    let value = array([1, 1, 1], vec![1.0_f64]);
    let invalid_mask = array([2, 1, 1], vec![1.0_f64; 2]);
    let mut output = array([1, 1, 1], vec![7.0_f64]);
    let mut weights = array([1, 1, 1], vec![8.0_f64]);

    let error = scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect_err("non-finite query must fail");
    assert_eq!(
        error,
        AttentionError::NonFinite {
            operand: AttentionOperand::Query,
        }
    );

    let finite_query = array([1, 1, 1], vec![1.0_f64]);
    let error = scaled_dot_product_attention_into(
        &finite_query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Keep(invalid_mask.view()),
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect_err("non-broadcast mask must fail");
    assert_eq!(
        error,
        AttentionError::MaskShape {
            actual: [2, 1, 1],
            target: [1, 1, 1],
        }
    );
    assert_eq!(output.storage().as_slice(), &[7.0]);
    assert_eq!(weights.storage().as_slice(), &[8.0]);
}

#[test]
fn derived_overflow_is_typed_and_failure_atomic() {
    let query = array([1, 1, 1], vec![f32::MAX]);
    let key = array([1, 1, 1], vec![f32::MAX]);
    let value = array([1, 1, 1], vec![1.0_f32]);
    let mut output = array([1, 1, 1], vec![7.0_f32]);
    let mut weights = array([1, 1, 1], vec![8.0_f32]);

    let error = scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect_err("finite operands whose score overflows must fail");
    assert_eq!(
        error,
        AttentionError::ArithmeticNonFinite {
            operand: AttentionOperand::Weights,
        }
    );
    assert_eq!(output.storage().as_slice(), &[7.0]);
    assert_eq!(weights.storage().as_slice(), &[8.0]);

    let mask = array([1, 1, 1], vec![0.0_f32]);
    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Keep(mask.view()),
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("masked scores do not participate in arithmetic");
    assert_eq!(output.storage().as_slice(), &[0.0]);
    assert_eq!(weights.storage().as_slice(), &[0.0]);
}

#[test]
fn finite_extreme_convex_output_is_not_rejected() {
    let query = array([1, 1, 1], vec![0.0_f32]);
    let key = array([1, 2, 1], vec![0.0_f32; 2]);
    let expected = 0.75 * f32::MAX;
    let value = array([1, 2, 1], vec![expected; 2]);
    let mut output = array([1, 1, 1], vec![0.0_f32]);
    let mut weights = array([1, 1, 2], vec![0.0_f32; 2]);
    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("a finite convex combination remains representable");
    assert_eq!(output.storage().as_slice(), &[expected]);
    assert_eq!(weights.storage().as_slice(), &[0.5, 0.5]);
}

#[test]
fn backward_rejects_invalid_probability_rows_and_overflow_atomically() {
    let tensor = array([1, 1, 1], vec![1.0_f32]);
    let invalid_weights = array([1, 1, 1], vec![-0.25_f32]);
    let mut gradient = array([1, 1, 1], vec![3.0_f32]);
    let error = scaled_dot_product_attention_backward_accumulate(
        &tensor.view(),
        &tensor.view(),
        &tensor.view(),
        &tensor.view(),
        &invalid_weights.view(),
        1.0,
        AttentionGradients::new(None, None, Some(gradient.view_mut())),
    )
    .expect_err("negative weights are not a softmax row");
    assert_eq!(error, AttentionError::InvalidWeights { batch: 0, query: 0 });
    assert_eq!(gradient.storage().as_slice(), &[3.0]);

    let weights = array([1, 1, 1], vec![1.0_f32]);
    let output_gradient = array([1, 1, 1], vec![f32::MAX]);
    let mut gradient = array([1, 1, 1], vec![f32::MAX]);
    let error = scaled_dot_product_attention_backward_accumulate(
        &output_gradient.view(),
        &tensor.view(),
        &tensor.view(),
        &tensor.view(),
        &weights.view(),
        1.0,
        AttentionGradients::new(None, None, Some(gradient.view_mut())),
    )
    .expect_err("additive gradient overflow must fail before mutation");
    assert_eq!(
        error,
        AttentionError::ArithmeticNonFinite {
            operand: AttentionOperand::ValueGradient,
        }
    );
    assert_eq!(gradient.storage().as_slice(), &[f32::MAX]);
}

#[test]
fn backward_rejects_an_empty_target_set() {
    let tensor = array([1, 1, 1], vec![1.0_f64]);
    let error = scaled_dot_product_attention_backward_accumulate(
        &tensor.view(),
        &tensor.view(),
        &tensor.view(),
        &tensor.view(),
        &tensor.view(),
        1.0,
        AttentionGradients::new(None, None, None),
    )
    .expect_err("at least one gradient target is required");
    assert_eq!(error, AttentionError::NoGradientTargets);
}
