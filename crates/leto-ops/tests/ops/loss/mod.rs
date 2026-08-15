use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array, ArrayView, ArrayViewMut, Layout, Storage, VecStorage};
use leto_ops::{
    cross_entropy_backward_accumulate, cross_entropy_forward_into, CrossEntropyError,
    CrossEntropyOperand, RealScalar,
};

fn array<T: Clone, const N: usize>(
    shape: [usize; N],
    values: Vec<T>,
) -> Array<T, VecStorage<T>, N> {
    Array::new(
        Layout::c_contiguous(shape).expect("test shape is representable"),
        VecStorage::new(values),
    )
    .expect("test storage matches its shape")
}

fn assert_close<T: NumericElement + RealField>(actual: &[T], expected: &[f64]) {
    // Per element, the longest tested path contains three shifted
    // exponentiations, two additions, reciprocal normalization, division,
    // logarithm, two online-mean operations, and three backward operations.
    // Thirty-two rounding units cover those fourteen operations plus one-ulp
    // elementary-function error and composition slack. Scaling by 1 + |x|
    // converts the relative bound into an absolute bound near zero.
    const ROUNDING_UNITS: f64 = 32.0;
    let epsilon = <T as RealField>::EPSILON.to_f64();
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual.to_f64() - expected).abs();
        let tolerance = ROUNDING_UNITS * epsilon * (1.0 + expected.abs());
        assert!(
            error <= tolerance,
            "index {index}: actual {}, expected {expected}, error {error}, tolerance {tolerance}",
            actual.to_f64()
        );
    }
}

fn forward_backward_contract<T: RealScalar + RealField>() {
    let logits = array(
        [2, 3],
        [2.0, 1.0, 0.0, 0.0, 1.0, 2.0]
            .map(FloatElement::from_f64)
            .to_vec(),
    );
    let mut probabilities = array([2, 3], vec![<T as NumericElement>::ZERO; 6]);
    let mut loss = array([1], vec![<T as NumericElement>::ZERO]);
    cross_entropy_forward_into(
        &logits.view(),
        &[0, 2],
        &mut loss.view_mut(),
        &mut probabilities.view_mut(),
    )
    .expect("valid cross-entropy forward contract");

    let normalization = 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp();
    let expected_probability = [
        1.0 / normalization,
        (-1.0_f64).exp() / normalization,
        (-2.0_f64).exp() / normalization,
    ];
    assert_close(loss.storage().as_slice(), &[normalization.ln()]);
    assert_close(
        probabilities.storage().as_slice(),
        &[
            expected_probability[0],
            expected_probability[1],
            expected_probability[2],
            expected_probability[2],
            expected_probability[1],
            expected_probability[0],
        ],
    );

    let mut gradient = array([2, 3], vec![<T as NumericElement>::ZERO; 6]);
    let output_gradient = array([1], vec![T::from_usize(2)]);
    cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probabilities.view(),
        &[0, 2],
        &mut gradient.view_mut(),
    )
    .expect("valid cross-entropy backward contract");
    assert_close(
        gradient.storage().as_slice(),
        &[
            expected_probability[0] - 1.0,
            expected_probability[1],
            expected_probability[2],
            expected_probability[2],
            expected_probability[1],
            expected_probability[0] - 1.0,
        ],
    );
}

#[test]
fn forward_and_backward_match_closed_form_for_native_precisions() {
    forward_backward_contract::<f32>();
    forward_backward_contract::<f64>();
}

#[test]
fn strided_forward_and_backward_preserve_padding() {
    let logits_data = [
        99.0_f64, 2.0, 99.0, 1.0, 99.0, 0.0, 99.0, 0.0, 99.0, 1.0, 99.0, 2.0,
    ];
    let layout = Layout::try_new([2, 3], [6, 2], 1).expect("valid test layout");
    let logits = ArrayView::try_new(layout, &logits_data).expect("valid strided logits");
    let mut probability_data = [77.0_f64; 12];
    let mut probabilities =
        ArrayViewMut::try_new(layout, &mut probability_data).expect("valid strided probabilities");
    let mut loss_data = [55.0_f64, 55.0];
    let mut loss = ArrayViewMut::try_new(
        Layout::try_new([1], [1], 1).expect("valid test layout"),
        &mut loss_data,
    )
    .expect("valid offset loss");
    cross_entropy_forward_into(&logits, &[0, 2], &mut loss, &mut probabilities)
        .expect("valid strided forward");
    let denominator = 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp();
    let expected = [
        1.0 / denominator,
        (-1.0_f64).exp() / denominator,
        (-2.0_f64).exp() / denominator,
    ];
    for batch in 0..2 {
        for class in 0..3 {
            let expected_class = if batch == 0 { class } else { 2 - class };
            assert_close(
                &[*probabilities
                    .get([batch, class])
                    .expect("logical probability is reachable")],
                &[expected[expected_class]],
            );
        }
    }
    assert_close(
        &[*loss.get([0]).expect("logical loss is reachable")],
        &[-expected[0].ln()],
    );
    assert_eq!(probabilities.data()[0], 77.0);
    assert_eq!(probabilities.data()[2], 77.0);
    assert_eq!(loss.data()[0], 55.0);

    let probability_view = probabilities.as_view();
    let output_gradient = array([1], vec![1.0_f64]);
    let mut gradient_data = [33.0_f64; 12];
    let mut gradient =
        ArrayViewMut::try_new(layout, &mut gradient_data).expect("valid strided gradient");
    cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probability_view,
        &[0, 2],
        &mut gradient,
    )
    .expect("valid strided backward");
    for batch in 0..2 {
        for class in 0..3 {
            let expected_class = if batch == 0 { class } else { 2 - class };
            let target_delta = if class == [0, 2][batch] { 1.0 } else { 0.0 };
            assert_close(
                &[*gradient
                    .get([batch, class])
                    .expect("logical gradient is reachable")],
                &[33.0 + (expected[expected_class] - target_delta) / 2.0],
            );
        }
    }
    assert_eq!(gradient_data[0], 33.0);
    assert_eq!(gradient_data[2], 33.0);
}

#[test]
fn permuted_layout_matches_logical_row_order() {
    let logits_data = [2.0_f64, 0.0, 1.0, 1.0, 0.0, 2.0];
    let layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let logits = ArrayView::try_new(layout, &logits_data).expect("valid permuted logits");
    let mut probability_data = [0.0_f64; 6];
    let mut probabilities =
        ArrayViewMut::try_new(layout, &mut probability_data).expect("valid permuted probabilities");
    let mut loss_data = [0.0_f64];
    let mut loss = ArrayViewMut::try_new(
        Layout::try_new([1], [1], 0).expect("valid test layout"),
        &mut loss_data,
    )
    .expect("valid scalar loss");

    cross_entropy_forward_into(&logits, &[0, 2], &mut loss, &mut probabilities)
        .expect("valid permuted forward");

    let denominator = 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp();
    let expected = [
        1.0 / denominator,
        (-1.0_f64).exp() / denominator,
        (-2.0_f64).exp() / denominator,
    ];
    for batch in 0..2 {
        for class in 0..3 {
            let expected_class = if batch == 0 { class } else { 2 - class };
            assert_close(
                &[*probabilities
                    .get([batch, class])
                    .expect("logical probability is reachable")],
                &[expected[expected_class]],
            );
        }
    }
}

#[test]
fn invalid_target_is_typed_and_failure_atomic() {
    let logits = array([1, 2], vec![0.0_f64, 1.0]);
    let mut probabilities = array([1, 2], vec![7.0_f64, 7.0]);
    let mut loss = array([1], vec![9.0_f64]);
    let error = cross_entropy_forward_into(
        &logits.view(),
        &[2],
        &mut loss.view_mut(),
        &mut probabilities.view_mut(),
    )
    .expect_err("target outside class range must fail");
    assert_eq!(
        error,
        CrossEntropyError::TargetOutOfRange {
            batch: 0,
            target: 2,
            classes: 2,
        }
    );
    assert_eq!(loss.storage().as_slice(), &[9.0]);
    assert_eq!(probabilities.storage().as_slice(), &[7.0, 7.0]);
}

#[test]
fn aliased_probability_destination_is_rejected_before_loss_mutation() {
    let logits = array([1, 2], vec![0.0_f64, 1.0]);
    let mut probability_data = [7.0_f64];
    let mut probabilities = ArrayViewMut::try_new(
        Layout::try_new([1, 2], [0, 0], 0).expect("valid test layout"),
        &mut probability_data,
    )
    .expect("broadcast layout is storage-reachable");
    let mut loss = array([1], vec![9.0_f64]);
    let error = cross_entropy_forward_into(
        &logits.view(),
        &[0],
        &mut loss.view_mut(),
        &mut probabilities,
    )
    .expect_err("aliased output must fail");
    assert!(matches!(
        error,
        CrossEntropyError::Layout {
            operand: CrossEntropyOperand::Probabilities,
            ..
        }
    ));
    assert_eq!(loss.storage().as_slice(), &[9.0]);
    assert_eq!(probability_data, [7.0]);
}

#[test]
fn non_finite_logits_are_rejected_before_outputs_change() {
    let logits = array([1, 2], vec![0.0_f64, f64::NAN]);
    let mut probabilities = array([1, 2], vec![7.0_f64, 7.0]);
    let mut loss = array([1], vec![9.0_f64]);
    let error = cross_entropy_forward_into(
        &logits.view(),
        &[0],
        &mut loss.view_mut(),
        &mut probabilities.view_mut(),
    )
    .expect_err("non-finite logits must fail");
    assert_eq!(
        error,
        CrossEntropyError::NonFinite {
            operand: CrossEntropyOperand::Logits,
        }
    );
    assert_eq!(loss.storage().as_slice(), &[9.0]);
    assert_eq!(probabilities.storage().as_slice(), &[7.0, 7.0]);
}

#[test]
fn invalid_saved_probabilities_are_rejected_before_gradient_changes() {
    let probabilities = array([1, 2], vec![0.75_f64, 0.5]);
    let output_gradient = array([1], vec![1.0_f64]);
    let mut gradient = array([1, 2], vec![4.0_f64, 5.0]);
    let error = cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probabilities.view(),
        &[0],
        &mut gradient.view_mut(),
    )
    .expect_err("probability row must sum to one");
    assert_eq!(error, CrossEntropyError::InvalidProbabilities { batch: 0 });
    assert_eq!(gradient.storage().as_slice(), &[4.0, 5.0]);
}

#[test]
fn non_finite_gradient_destination_is_rejected_before_mutation() {
    let probabilities = array([1, 2], vec![0.25_f64, 0.75]);
    let output_gradient = array([1], vec![1.0_f64]);
    let mut gradient = array([1, 2], vec![f64::NAN, 5.0]);
    let error = cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probabilities.view(),
        &[0],
        &mut gradient.view_mut(),
    )
    .expect_err("non-finite additive input must fail");
    assert_eq!(
        error,
        CrossEntropyError::NonFinite {
            operand: CrossEntropyOperand::LogitGradient,
        }
    );
    assert!(gradient.storage().as_slice()[0].is_nan());
    assert_eq!(gradient.storage().as_slice()[1], 5.0);
}

#[test]
fn overflowing_gradient_addition_is_failure_atomic() {
    let probabilities = array([1, 2], vec![0.0_f64, 1.0]);
    let output_gradient = array([1], vec![f64::MAX]);
    let mut gradient = array([1, 2], vec![-f64::MAX, 5.0]);
    let error = cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probabilities.view(),
        &[0],
        &mut gradient.view_mut(),
    )
    .expect_err("overflowing additive update must fail before mutation");
    assert_eq!(error, CrossEntropyError::ArithmeticNonFinite { batch: 0 });
    assert_eq!(gradient.storage().as_slice(), &[-f64::MAX, 5.0]);
}

#[test]
fn backward_adds_scaled_gradient_to_existing_destination() {
    let probabilities = array([1, 2], vec![0.25_f64, 0.75]);
    let output_gradient = array([1], vec![2.0_f64]);
    let mut gradient = array([1, 2], vec![3.0_f64, 4.0]);
    cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probabilities.view(),
        &[1],
        &mut gradient.view_mut(),
    )
    .expect("valid additive backward");
    assert_eq!(gradient.storage().as_slice(), &[3.5, 3.5]);
}

#[test]
fn stable_loss_avoids_log_sum_exp_overflow() {
    let logits = array([1, 2], vec![-1.0e300_f64, 1.0e300]);
    let mut probabilities = array([1, 2], vec![0.0_f64; 2]);
    let mut loss = array([1], vec![0.0_f64]);
    cross_entropy_forward_into(
        &logits.view(),
        &[0],
        &mut loss.view_mut(),
        &mut probabilities.view_mut(),
    )
    .expect("finite dynamic range is representable");
    assert_eq!(loss.storage().as_slice(), &[2.0e300]);
    assert_eq!(probabilities.storage().as_slice(), &[0.0, 1.0]);
}

#[test]
fn mean_reduction_avoids_representable_sum_overflow() {
    for magnitude in [1.0e38_f64, 1.0e308_f64] {
        let logits = array([2, 2], vec![0.0, -magnitude, 0.0, -magnitude]);
        let mut probabilities = array([2, 2], vec![0.0_f64; 4]);
        let mut loss = array([1], vec![0.0_f64]);
        cross_entropy_forward_into(
            &logits.view(),
            &[1, 1],
            &mut loss.view_mut(),
            &mut probabilities.view_mut(),
        )
        .expect("representable mean must not overflow through its sum");
        assert_eq!(loss.storage().as_slice(), &[magnitude]);
    }

    let logits = array([2, 2], vec![0.0_f32, -1.0e38, 0.0, -1.0e38]);
    let mut probabilities = array([2, 2], vec![0.0_f32; 4]);
    let mut loss = array([1], vec![0.0_f32]);
    cross_entropy_forward_into(
        &logits.view(),
        &[1, 1],
        &mut loss.view_mut(),
        &mut probabilities.view_mut(),
    )
    .expect("representable f32 mean must not overflow through its sum");
    assert_eq!(loss.storage().as_slice(), &[1.0e38_f32]);
}

#[test]
fn probability_validation_rejects_large_mass_deficit() {
    const CLASSES: usize = 1 << 19;
    let mut probability_values = vec![0.0_f32; CLASSES];
    probability_values[0] = 0.75;
    let probabilities = array([1, CLASSES], probability_values);
    let output_gradient = array([1], vec![1.0_f32]);
    let mut gradient = array([1, CLASSES], vec![0.0_f32; CLASSES]);
    let error = cross_entropy_backward_accumulate(
        &output_gradient.view(),
        &probabilities.view(),
        &[0],
        &mut gradient.view_mut(),
    )
    .expect_err("a one-quarter probability-mass deficit is not rounding error");
    assert_eq!(error, CrossEntropyError::InvalidProbabilities { batch: 0 });
}
