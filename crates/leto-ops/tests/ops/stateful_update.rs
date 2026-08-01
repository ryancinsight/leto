use eunomia::{FloatElement, NumericElement};
use leto::{ArrayView, ArrayViewMut, Layout, LetoError};
use leto_ops::{
    stateful_update, AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW, AdamWParameters,
    RealScalar, RmsProp, RmsPropParameters, Sgd, SgdParameters,
};

const LOGICAL: [usize; 4] = [1, 2, 4, 5];

fn scalar<T: FloatElement>(value: f64) -> T {
    T::from_f64(value)
}

fn values<T: NumericElement>(storage: &[T; 6]) -> [f64; 4] {
    LOGICAL.map(|index| storage[index].to_f64())
}

fn assert_close(actual: [f64; 4], expected: [f64; 4], epsilon: f64) {
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        // Each oracle expression has at most 16 rounded elementary operations.
        // A factor of two covers first-order composition without assuming exact
        // operation reassociation, yielding 32 * epsilon at unit scale.
        let tolerance = 2.0 * 16.0 * epsilon * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn assert_guards<T: NumericElement>(storage: &[T; 6], first: f64, second: f64) {
    assert_eq!(storage[0].to_f64(), first);
    assert_eq!(storage[3].to_f64(), second);
}

fn run<T: RealScalar>(epsilon: f64) {
    let layout = Layout::new([2, 2], [3, 1], 1);
    let gradient_storage = [
        scalar(81.0),
        scalar(0.1),
        scalar(0.2),
        scalar(82.0),
        scalar(0.3),
        scalar(0.4),
    ];
    let gradient = ArrayView::new(layout, &gradient_storage);
    let parameter_initial = [1.0, 2.0, 3.0, 4.0];
    let gradient_values = [0.1, 0.2, 0.3, 0.4];
    let state_initial = [0.5, 0.6, 0.7, 0.8];

    let mut parameter_storage = [
        scalar(91.0),
        scalar(1.0),
        scalar(2.0),
        scalar(92.0),
        scalar(3.0),
        scalar(4.0),
    ];
    let mut state_storage = [
        scalar(71.0),
        scalar(0.5),
        scalar(0.6),
        scalar(72.0),
        scalar(0.7),
        scalar(0.8),
    ];
    stateful_update::<T, Sgd, 2>(
        ArrayViewMut::new(layout, &mut parameter_storage),
        gradient,
        ArrayViewMut::new(layout, &mut state_storage),
        SgdParameters::new(scalar(0.05), scalar(0.9)).expect("SGD parameters"),
    )
    .expect("SGD update");
    let velocity = core::array::from_fn(|i| 0.9 * state_initial[i] + gradient_values[i]);
    let parameter = core::array::from_fn(|i| parameter_initial[i] - 0.05 * velocity[i]);
    assert_close(values(&state_storage), velocity, epsilon);
    assert_close(values(&parameter_storage), parameter, epsilon);
    assert_guards(&parameter_storage, 91.0, 92.0);
    assert_guards(&state_storage, 71.0, 72.0);

    let mut parameter_storage = [
        scalar(91.0),
        scalar(1.0),
        scalar(2.0),
        scalar(92.0),
        scalar(3.0),
        scalar(4.0),
    ];
    let mut first_storage = [
        scalar(71.0),
        scalar(0.5),
        scalar(0.6),
        scalar(72.0),
        scalar(0.7),
        scalar(0.8),
    ];
    let mut second_storage = [
        scalar(61.0),
        scalar(0.25),
        scalar(0.36),
        scalar(62.0),
        scalar(0.49),
        scalar(0.64),
    ];
    let adam = AdamParameters::new(scalar(0.01), scalar(0.9), scalar(0.99), scalar(1.0e-6), 3)
        .expect("Adam parameters");
    stateful_update::<T, Adam, 2>(
        ArrayViewMut::new(layout, &mut parameter_storage),
        gradient,
        (
            ArrayViewMut::new(layout, &mut first_storage),
            ArrayViewMut::new(layout, &mut second_storage),
        ),
        adam,
    )
    .expect("Adam update");
    let first = core::array::from_fn(|i| 0.9 * state_initial[i] + 0.1 * gradient_values[i]);
    let second_initial = [0.25, 0.36, 0.49, 0.64];
    let second = core::array::from_fn(|i| {
        0.99 * second_initial[i] + 0.01 * gradient_values[i] * gradient_values[i]
    });
    let bias_one = 1.0 - 0.9_f64.powi(3);
    let bias_two = 1.0 - 0.99_f64.powi(3);
    let expected = core::array::from_fn(|i| {
        parameter_initial[i]
            - 0.01 * (first[i] / bias_one) / ((second[i] / bias_two).sqrt() + 1.0e-6)
    });
    assert_close(values(&first_storage), first, epsilon);
    assert_close(values(&second_storage), second, epsilon);
    assert_close(values(&parameter_storage), expected, epsilon);
    assert_guards(&parameter_storage, 91.0, 92.0);
    assert_guards(&first_storage, 71.0, 72.0);
    assert_guards(&second_storage, 61.0, 62.0);

    let mut parameter_storage = [
        scalar(91.0),
        scalar(1.0),
        scalar(2.0),
        scalar(92.0),
        scalar(3.0),
        scalar(4.0),
    ];
    let mut first_storage = [
        scalar(71.0),
        scalar(0.5),
        scalar(0.6),
        scalar(72.0),
        scalar(0.7),
        scalar(0.8),
    ];
    let mut second_storage = [
        scalar(61.0),
        scalar(0.25),
        scalar(0.36),
        scalar(62.0),
        scalar(0.49),
        scalar(0.64),
    ];
    stateful_update::<T, AdamW, 2>(
        ArrayViewMut::new(layout, &mut parameter_storage),
        gradient,
        (
            ArrayViewMut::new(layout, &mut first_storage),
            ArrayViewMut::new(layout, &mut second_storage),
        ),
        AdamWParameters::new(
            scalar(0.01),
            scalar(0.9),
            scalar(0.99),
            scalar(1.0e-6),
            scalar(0.1),
            3,
        )
        .expect("AdamW parameters"),
    )
    .expect("AdamW update");
    let expected = core::array::from_fn(|i| {
        parameter_initial[i] * (1.0 - 0.01 * 0.1)
            - 0.01 * (first[i] / bias_one) / ((second[i] / bias_two).sqrt() + 1.0e-6)
    });
    assert_close(values(&parameter_storage), expected, epsilon);
    assert_guards(&parameter_storage, 91.0, 92.0);
    assert_guards(&first_storage, 71.0, 72.0);
    assert_guards(&second_storage, 61.0, 62.0);

    let mut parameter_storage = [
        scalar(91.0),
        scalar(1.0),
        scalar(2.0),
        scalar(92.0),
        scalar(3.0),
        scalar(4.0),
    ];
    let mut state_storage = [
        scalar(71.0),
        scalar(0.5),
        scalar(0.6),
        scalar(72.0),
        scalar(0.7),
        scalar(0.8),
    ];
    stateful_update::<T, RmsProp, 2>(
        ArrayViewMut::new(layout, &mut parameter_storage),
        gradient,
        ArrayViewMut::new(layout, &mut state_storage),
        RmsPropParameters::new(scalar(0.05), scalar(0.9), scalar(1.0e-6))
            .expect("RMSProp parameters"),
    )
    .expect("RMSProp update");
    let average = core::array::from_fn(|i| {
        0.9 * state_initial[i] + 0.1 * gradient_values[i] * gradient_values[i]
    });
    let expected = core::array::from_fn(|i| {
        parameter_initial[i] - 0.05 * gradient_values[i] / (average[i].sqrt() + 1.0e-6)
    });
    assert_close(values(&state_storage), average, epsilon);
    assert_close(values(&parameter_storage), expected, epsilon);
    assert_guards(&parameter_storage, 91.0, 92.0);
    assert_guards(&state_storage, 71.0, 72.0);

    let mut parameter_storage = [
        scalar(91.0),
        scalar(1.0),
        scalar(2.0),
        scalar(92.0),
        scalar(3.0),
        scalar(4.0),
    ];
    let mut state_storage = [
        scalar(71.0),
        scalar(0.5),
        scalar(0.6),
        scalar(72.0),
        scalar(0.7),
        scalar(0.8),
    ];
    stateful_update::<T, AdaGrad, 2>(
        ArrayViewMut::new(layout, &mut parameter_storage),
        gradient,
        ArrayViewMut::new(layout, &mut state_storage),
        AdaGradParameters::new(scalar(0.05), scalar(1.0e-6)).expect("AdaGrad parameters"),
    )
    .expect("AdaGrad update");
    let sum = core::array::from_fn(|i| state_initial[i] + gradient_values[i] * gradient_values[i]);
    let expected = core::array::from_fn(|i| {
        parameter_initial[i] - 0.05 * gradient_values[i] / (sum[i].sqrt() + 1.0e-6)
    });
    assert_close(values(&state_storage), sum, epsilon);
    assert_close(values(&parameter_storage), expected, epsilon);
    assert_guards(&parameter_storage, 91.0, 92.0);
    assert_guards(&state_storage, 71.0, 72.0);
}

#[test]
fn all_rules_match_scalar_oracles_for_each_admitted_precision() {
    run::<f32>(f32::EPSILON as f64);
    run::<f64>(f64::EPSILON);
}

#[test]
fn validation_is_failure_atomic() {
    let parameter_before = [1.0_f32, 2.0];
    let state_before = [0.5_f32, 0.6];
    let mut parameter = parameter_before;
    let gradient = [0.1_f32, 0.2];
    let mut state = state_before;
    let parameter_layout = Layout::c_contiguous([2]).expect("parameter layout");
    let gradient_layout = Layout::c_contiguous([1]).expect("gradient layout");
    let error = stateful_update::<f32, Sgd, 1>(
        ArrayViewMut::new(parameter_layout, &mut parameter),
        ArrayView::new(gradient_layout, &gradient),
        ArrayViewMut::new(parameter_layout, &mut state),
        SgdParameters::new(0.1, 0.9).expect("parameters"),
    )
    .expect_err("shape mismatch");
    assert!(matches!(error, LetoError::ShapeMismatch { .. }));
    assert_eq!(parameter, parameter_before);
    assert_eq!(state, state_before);

    assert_invalid(
        SgdParameters::new(f32::NAN, 0.9),
        "SGD learning rate must be finite and positive",
    );
    assert_invalid(
        AdamParameters::new(0.1, 0.9, 0.99, 1.0e-6, 0),
        "Adam step must be positive",
    );
    assert_invalid(
        AdamWParameters::new(0.1, 0.9, 0.99, 1.0e-6, -0.1, 1),
        "AdamW weight decay must be finite and non-negative",
    );
    assert_invalid(
        RmsPropParameters::new(0.1, 1.0, 1.0e-6),
        "RMSProp alpha must be finite in [0, 1)",
    );
    assert_invalid(
        AdaGradParameters::new(0.1, 0.0),
        "AdaGrad epsilon must be finite and positive",
    );
}

fn assert_invalid<T>(result: Result<T, LetoError>, expected: &str) {
    match result {
        Err(LetoError::InvalidInput(reason)) => assert_eq!(reason, expected),
        Err(error) => panic!("expected InvalidInput({expected:?}), got {error:?}"),
        Ok(_) => panic!("expected InvalidInput({expected:?}), got success"),
    }
}

#[test]
fn dense_scalar_empty_and_rank_boundary_layouts_execute() {
    let dense = Layout::c_contiguous([2]).expect("dense layout");
    let mut dense_parameter = [1.0_f32, 2.0];
    let dense_gradient = [0.25_f32, -0.5];
    let mut dense_velocity = [0.0_f32; 2];
    stateful_update::<f32, Sgd, 1>(
        ArrayViewMut::new(dense, &mut dense_parameter),
        ArrayView::new(dense, &dense_gradient),
        ArrayViewMut::new(dense, &mut dense_velocity),
        SgdParameters::new(0.1, 0.0).expect("dense parameters"),
    )
    .expect("dense update");
    assert_eq!(dense_velocity, dense_gradient);
    assert_eq!(dense_parameter, [0.975, 2.05]);

    let scalar_layout = Layout::c_contiguous([]).expect("scalar layout");
    let mut scalar_parameter = [2.0_f32];
    let scalar_gradient = [0.5_f32];
    let mut scalar_velocity = [0.25_f32];
    stateful_update::<f32, Sgd, 0>(
        ArrayViewMut::new(scalar_layout, &mut scalar_parameter),
        ArrayView::new(scalar_layout, &scalar_gradient),
        ArrayViewMut::new(scalar_layout, &mut scalar_velocity),
        SgdParameters::new(0.2, 0.5).expect("scalar parameters"),
    )
    .expect("scalar update");
    assert_eq!(scalar_velocity, [0.625]);
    assert_eq!(scalar_parameter, [1.875]);

    let empty_layout =
        Layout::c_contiguous([1, 1, 1, 0, 1, 1, 1, 1]).expect("rank-eight empty layout");
    let mut empty_parameter: [f32; 0] = [];
    let empty_gradient: [f32; 0] = [];
    let mut empty_velocity: [f32; 0] = [];
    stateful_update::<f32, Sgd, 8>(
        ArrayViewMut::new(empty_layout, &mut empty_parameter),
        ArrayView::new(empty_layout, &empty_gradient),
        ArrayViewMut::new(empty_layout, &mut empty_velocity),
        SgdParameters::new(0.1, 0.9).expect("empty parameters"),
    )
    .expect("empty update");
    assert_eq!(empty_parameter, []);
    assert_eq!(empty_velocity, []);
}

#[test]
fn non_finite_operands_follow_ieee_propagation() {
    let layout = Layout::c_contiguous([1]).expect("layout");
    let mut parameter = [1.0_f32];
    let gradient = [f32::NAN];
    let mut velocity = [0.0_f32];
    stateful_update::<f32, Sgd, 1>(
        ArrayViewMut::new(layout, &mut parameter),
        ArrayView::new(layout, &gradient),
        ArrayViewMut::new(layout, &mut velocity),
        SgdParameters::new(0.1, 0.9).expect("parameters"),
    )
    .expect("IEEE update");
    assert!(parameter[0].is_nan());
    assert!(velocity[0].is_nan());
}

#[test]
fn interleaved_injective_layout_updates_each_logical_element_once() {
    let layout = Layout::new([2, 3], [3, 2], 0);
    let mut parameter = [1.0_f32, 91.0, 2.0, 3.0, 4.0, 5.0, 92.0, 6.0];
    let gradient = [0.1_f32, 81.0, 0.2, 0.3, 0.4, 0.5, 82.0, 0.6];
    let mut velocity = [0.0_f32, 71.0, 0.0, 0.0, 0.0, 0.0, 72.0, 0.0];
    stateful_update::<f32, Sgd, 2>(
        ArrayViewMut::new(layout, &mut parameter),
        ArrayView::new(layout, &gradient),
        ArrayViewMut::new(layout, &mut velocity),
        SgdParameters::new(0.5, 0.0).expect("parameters"),
    )
    .expect("interleaved update");

    assert_eq!(parameter, [0.95, 91.0, 1.9, 2.85, 3.8, 4.75, 92.0, 5.7]);
    assert_eq!(velocity, [0.1, 71.0, 0.2, 0.3, 0.4, 0.5, 72.0, 0.6]);
}
