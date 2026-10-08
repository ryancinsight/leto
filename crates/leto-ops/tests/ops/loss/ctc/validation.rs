//! Typed rejection surface: lengths, labels, storage, and destinations.

use super::fixtures::array;
use leto::{ArrayView, ArrayViewMut, Layout, Storage};
use leto_ops::ctc::{CtcError, CtcState};

#[test]
fn ctc_rejects_invalid_lengths_labels_and_numeric_inputs() {
    let input = array([1, 1, 2], vec![-std::f32::consts::LN_2; 2]);
    assert!(matches!(
        CtcState::forward(&input.view(), &[1], &[2], &[1], 0),
        Err(CtcError::InputLength {
            sample: 0,
            actual: 2,
            maximum: 1
        })
    ));
    assert!(matches!(
        CtcState::forward(&input.view(), &[], &[1], &[1], 0),
        Err(CtcError::TargetCount {
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        CtcState::forward(&input.view(), &[1], &[], &[1], 0),
        Err(CtcError::LengthCount { .. })
    ));
    for label in [0, 2] {
        assert!(matches!(
            CtcState::forward(&input.view(), &[label], &[1], &[1], 0),
            Err(CtcError::Label { .. })
        ));
    }
    assert!(matches!(
        CtcState::forward(&input.view(), &[], &[1], &[0], 2),
        Err(CtcError::Label { .. })
    ));
    assert!(matches!(
        CtcState::forward(&input.view(), &[], &[1], &[usize::MAX], 0),
        Err(CtcError::SizeOverflow { .. })
    ));
    for value in [f32::NAN, f32::INFINITY, 0.25] {
        let bad = array([1, 1, 2], vec![value, -std::f32::consts::LN_2]);
        assert!(matches!(
            CtcState::forward(&bad.view(), &[], &[1], &[0], 0),
            Err(CtcError::LogProbability { index: [0, 0, 0] })
        ));
    }
}

#[test]
fn ctc_backward_errors_preserve_destination() {
    let input = array([1, 1, 2], vec![-std::f32::consts::LN_2; 2]);
    let state = CtcState::forward(&input.view(), &[], &[1], &[0], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    let mut gradient = array([1, 1, 2], vec![7.0_f32; 2]);
    for seed in [f32::NAN, f32::INFINITY] {
        assert!(matches!(
            state.backward_accumulate(seed, &mut gradient.view_mut()),
            Err(CtcError::Arithmetic { .. })
        ));
        assert_eq!(gradient.storage().as_slice(), &[7.0; 2]);
    }
    let mut overflow = array([1, 1, 2], vec![-f32::MAX, 7.0]);
    assert!(matches!(
        state.backward_accumulate(f32::MAX, &mut overflow.view_mut()),
        Err(CtcError::Arithmetic { .. })
    ));
    assert_eq!(overflow.storage().as_slice(), &[-f32::MAX, 7.0]);
    let mut wrong = array([1, 2, 1], vec![7.0_f32; 2]);
    assert!(matches!(
        state.backward_accumulate(1.0, &mut wrong.view_mut()),
        Err(CtcError::GradientShape { .. })
    ));
    assert_eq!(wrong.storage().as_slice(), &[7.0; 2]);
}

#[test]
fn ctc_strided_offset_views_and_invalid_storage() {
    let values = [
        99.0_f32,
        -std::f32::consts::LN_2,
        99.0,
        -std::f32::consts::LN_2,
        99.0,
    ];
    let layout = Layout::try_new([1, 1, 2], [4, 4, -2], 3)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    let input = ArrayView::new(layout, &values);
    let state = CtcState::forward(&input, &[], &[1], &[0], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    assert_eq!(state.loss(), std::f32::consts::LN_2);
    let mut output = [11.0_f32; 5];
    state
        .backward_accumulate(2.0, &mut ArrayViewMut::new(layout, &mut output))
        .expect("invariant: analytical fixture satisfies the operation boundary");
    assert_eq!(output, [11.0, 11.0, 11.0, 9.0, 11.0]);
    let short = ArrayView::new(layout, &values[..2]);
    assert!(matches!(
        CtcState::forward(&short, &[], &[1], &[0], 0),
        Err(CtcError::Layout(_))
    ));
    let mut short_output = [13.0_f32; 2];
    assert!(matches!(
        state.backward_accumulate(1.0, &mut ArrayViewMut::new(layout, &mut short_output)),
        Err(CtcError::Layout(_))
    ));
    assert_eq!(short_output, [13.0; 2]);
    let overlap = Layout::try_new([1, 1, 2], [0, 0, 0], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    let mut overlapped_output = [13.0_f32];
    assert!(matches!(
        state.backward_accumulate(1.0, &mut ArrayViewMut::new(overlap, &mut overlapped_output)),
        Err(CtcError::AliasedGradient)
    ));
    assert_eq!(overlapped_output, [13.0]);
}
