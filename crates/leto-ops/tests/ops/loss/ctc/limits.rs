//! Boundary and degenerate-extent contracts.

use super::fixtures::{array, close};
use eunomia::{Bf16, FloatElement, F16};
use leto::Storage;
use leto_ops::{
    ctc::{CtcError, CtcState},
    RealScalar,
};

fn boundaries<T: RealScalar>() {
    let input = array([2, 2, 2], vec![FloatElement::ln(T::from_f64(0.5)); 8]);
    let state = CtcState::forward(&input.view(), &[1], &[1, 2], &[0, 1], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    close(
        state.loss(),
        (2.0_f64.ln() + (4.0_f64 / 3.0).ln()) / 2.0,
        20,
    );
    let mut gradient = array([2, 2, 2], vec![T::from_count(4); 8]);
    state
        .backward_accumulate(T::from_count(2), &mut gradient.view_mut())
        .expect("invariant: analytical fixture satisfies the operation boundary");
    for (&actual, expected) in gradient.storage().as_slice().iter().zip([
        3.0,
        4.0,
        4.0 - 1.0 / 3.0,
        4.0 - 2.0 / 3.0,
        4.0,
        4.0,
        4.0 - 1.0 / 3.0,
        4.0 - 2.0 / 3.0,
    ]) {
        close(actual, expected, 24);
    }
    // Empty valid prefixes are legal even when the stored tensor has padding.
    let state = CtcState::forward(&input.view(), &[], &[0, 0], &[0, 0], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    assert_eq!(state.loss(), T::ZERO);
    let before = gradient.storage().as_slice().to_vec();
    state
        .backward_accumulate(T::ONE, &mut gradient.view_mut())
        .expect("invariant: analytical fixture satisfies the operation boundary");
    assert_eq!(gradient.storage().as_slice(), before);
    for (targets, lengths, target_lengths, failing_sample) in [
        (vec![1], [0, 0], [0, 1], 1),
        (vec![1, 1], [2, 2], [0, 2], 1),
    ] {
        let state = CtcState::forward(&input.view(), &targets, &lengths, &target_lengths, 0)
            .expect("invariant: analytical fixture satisfies the operation boundary");
        assert_eq!(state.loss(), T::INFINITY);
        assert!(
            matches!(state.backward_accumulate(T::ONE, &mut gradient.view_mut()),
            Err(CtcError::ImpossibleAlignment { sample }) if sample == failing_sample)
        );
        assert_eq!(gradient.storage().as_slice(), before);
    }
    let zeros = array([1, 1, 2], vec![-T::INFINITY; 2]);
    let impossible = CtcState::forward(&zeros.view(), &[], &[1], &[0], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    assert_eq!(impossible.loss(), T::INFINITY);
}

#[test]
fn ctc_empty_impossible_and_padded_samples() {
    boundaries::<f32>();
    boundaries::<f64>();
    boundaries::<F16>();
    boundaries::<Bf16>();
}

#[test]
fn ctc_zero_frame_extent_and_padding_values() {
    let empty = array([0, 1, 2], Vec::<f32>::new());
    assert_eq!(
        CtcState::forward(&empty.view(), &[], &[0], &[0], 0)
            .expect("invariant: analytical fixture satisfies the operation boundary")
            .loss(),
        0.0
    );
    assert_eq!(
        CtcState::forward(&empty.view(), &[1], &[0], &[1], 0)
            .expect("invariant: analytical fixture satisfies the operation boundary")
            .loss(),
        f32::INFINITY
    );
    let empty_batch = array([1, 0, 2], Vec::<f32>::new());
    assert!(matches!(
        CtcState::forward(&empty_batch.view(), &[], &[], &[], 0),
        Err(CtcError::EmptyDimension { .. })
    ));
    let padding = array([2, 1, 2], vec![-std::f32::consts::LN_2; 4]);
    let mut values = padding.storage().as_slice().to_vec();
    values[2] = f32::NAN;
    values[3] = f32::INFINITY;
    let padded = array([2, 1, 2], values);
    assert_eq!(
        CtcState::forward(&padded.view(), &[], &[1], &[0], 0)
            .expect("invariant: analytical fixture satisfies the operation boundary")
            .loss(),
        std::f32::consts::LN_2
    );
}
