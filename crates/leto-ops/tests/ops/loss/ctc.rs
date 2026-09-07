use eunomia::{Bf16, FloatElement, F16};
use leto::{Array, ArrayView, ArrayViewMut, Layout, Storage, VecStorage};
use leto_ops::{
    ctc::{CtcError, CtcState},
    RealScalar,
};

fn array<T, const N: usize>(shape: [usize; N], values: Vec<T>) -> Array<T, VecStorage<T>, N> {
    Array::new(
        Layout::c_contiguous(shape)
            .expect("invariant: analytical fixture satisfies the operation boundary"),
        VecStorage::new(values),
    )
    .expect("invariant: analytical fixture satisfies the operation boundary")
}

fn close<T: RealScalar>(actual: T, expected: f64, operations: usize) {
    // Unit roundoff u=epsilon/2. The caller counts the arithmetic and
    // elementary-function rounding sites along its longest dependency path.
    // gamma(k)=ku/(1-ku), scaled by 1+|reference|, also covers absolute error
    // near zero. Oracles enumerate paths independently in double precision.
    let half = T::ONE / (T::ONE + T::ONE);
    let mut epsilon = T::ONE;
    while T::ONE + epsilon * half > T::ONE {
        epsilon *= half;
    }
    let ku = epsilon.to_f64() * 0.5 * operations as f64;
    let bound = ku / (1.0 - ku) * (1.0 + expected.abs());
    assert!(ku < 1.0);
    assert!(
        (actual.to_f64() - expected).abs() <= bound,
        "actual {:?}, expected {expected}, bound {bound}",
        actual
    );
}

fn analytical<T: RealScalar>() {
    let half_log = FloatElement::ln(T::from_f64(0.5));
    let input = array([1, 1, 2], vec![half_log; 2]);
    for (targets, expected) in [(vec![], [-2.0, 0.0]), (vec![1], [0.0, -2.0])] {
        let state = CtcState::forward(&input.view(), &targets, &[1], &[targets.len()], 0)
            .expect("invariant: analytical fixture satisfies the operation boundary");
        assert_eq!(state.loss(), -half_log);
        let mut gradient = array([1, 1, 2], vec![T::from_usize(3); 2]);
        state
            .backward_accumulate(T::from_usize(2), &mut gradient.view_mut())
            .expect("invariant: analytical fixture satisfies the operation boundary");
        for (&value, expected) in gradient.storage().as_slice().iter().zip(expected) {
            assert_eq!(value, T::from_f64(3.0 + expected));
        }
    }
    let input = array([2, 1, 2], vec![half_log; 4]);
    let state = CtcState::forward(&input.view(), &[1], &[2], &[1], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    // Valid paths [blank,label], [label,blank], [label,label] each have
    // probability 1/4. Occupancies are [1/3,2/3] at both times.
    close(state.loss(), (4.0_f64 / 3.0).ln(), 16);
    let mut gradient = array([2, 1, 2], vec![T::ZERO; 4]);
    state
        .backward_accumulate(T::ONE, &mut gradient.view_mut())
        .expect("invariant: analytical fixture satisfies the operation boundary");
    for (&actual, expected) in gradient
        .storage()
        .as_slice()
        .iter()
        .zip([-1.0 / 3.0, -2.0 / 3.0].into_iter().cycle())
    {
        close(actual, expected, 24);
    }
}

#[test]
fn ctc_analytical_loss_and_seeded_gradient() {
    analytical::<f32>();
    analytical::<f64>();
    analytical::<F16>();
    analytical::<Bf16>();
}

fn normalization<T: RealScalar>() {
    // 60_000 is finite in every shipped real type. Only one sample emits,
    // with its log weight equal to the batch divisor, so both ratios are
    // exactly one even when the reciprocal divisor is subnormal.
    let batch = 60_000;
    let divisor = T::from_usize(batch);
    let mut values = vec![T::ZERO; batch];
    values[0] = -divisor;
    let input = array([1, batch, 1], values);
    let mut lengths = vec![0; batch];
    lengths[0] = 1;
    let state = CtcState::forward(&input.view(), &[], &lengths, &vec![0; batch], 0)
        .expect("invariant: one active blank path has a representable normalized loss");
    assert_eq!(state.loss(), T::ONE);
    let mut gradient = array([1, batch, 1], vec![T::ZERO; batch]);
    state
        .backward_accumulate(divisor, &mut gradient.view_mut())
        .expect("invariant: seeded normalized occupancy is exactly one");
    assert_eq!(gradient.storage().as_slice()[0], -T::ONE);
    assert!(gradient.storage().as_slice()[1..]
        .iter()
        .all(|&x| x == T::ZERO));
}

#[test]
fn ctc_normalization_preserves_representable_ratios() {
    normalization::<f32>();
    normalization::<f64>();
    normalization::<F16>();
    normalization::<Bf16>();
}

fn separated_paths<T: RealScalar>() {
    // Pick A with spacing above ln(3), so a single scalar loses path counts.
    // All extra-emission paths are below exp(-A); A>=512 makes their aggregate
    // contribution smaller than unit roundoff even in the widest shipped type.
    let mut magnitude = T::from_usize(512);
    while magnitude + T::ONE != magnitude {
        magnitude += magnitude;
    }
    for frames in [2, 3] {
        let input = array(
            [frames, 1, 2],
            [T::ZERO, -magnitude]
                .into_iter()
                .cycle()
                .take(frames * 2)
                .collect(),
        );
        let state = CtcState::forward(&input.view(), &[1], &[frames], &[1], 0)
            .expect("invariant: separated paths have finite log likelihood");
        let expected_loss = magnitude - FloatElement::ln(T::from_usize(frames));
        assert_eq!(state.loss(), expected_loss);
        let mut gradient = array([frames, 1, 2], vec![T::ZERO; frames * 2]);
        state
            .backward_accumulate(T::ONE, &mut gradient.view_mut())
            .expect("invariant: centered posterior is representable");
        for row in gradient.storage().as_slice().chunks_exact(2) {
            if frames == 2 {
                assert_eq!(row, &[-T::from_f64(0.5); 2]);
            }
            // At most two merges per frame, compensated sums, one exp and the
            // class sum: 64 rounding sites bound this three-frame dependency.
            close(row[0], -(frames as f64 - 1.0) / frames as f64, 64);
            close(row[1], -1.0 / frames as f64, 64);
        }
    }
}

#[test]
fn ctc_preserves_path_counts_below_log_weight_spacing() {
    separated_paths::<f32>();
    separated_paths::<f64>();
    separated_paths::<F16>();
    separated_paths::<Bf16>();
}

fn impossible_normalization<T: RealScalar>() {
    // 8192^2 exceeds the reciprocal range of F16 although each divisor and
    // the specified impossible loss remain representable.
    let batch = 8192;
    let input = array([0, batch, 2], Vec::<T>::new());
    let mut target_lengths = vec![0; batch];
    target_lengths[0] = batch;
    let state = CtcState::forward(
        &input.view(),
        &vec![1; batch],
        &vec![0; batch],
        &target_lengths,
        0,
    )
    .expect("invariant: impossible paths have infinite loss at every normalization");
    assert_eq!(state.loss(), T::INFINITY);
    let mut gradient = array([0, batch, 2], Vec::<T>::new());
    assert!(matches!(
        state.backward_accumulate(T::ONE, &mut gradient.view_mut()),
        Err(CtcError::ImpossibleAlignment { sample: 0 })
    ));
    assert_eq!(gradient.storage().as_slice(), &[]);
}

#[test]
fn ctc_impossible_loss_survives_small_normalization() {
    impossible_normalization::<f32>();
    impossible_normalization::<f64>();
    impossible_normalization::<F16>();
    impossible_normalization::<Bf16>();
}

fn enumerated<T: RealScalar>() {
    let frames = 3;
    let classes = 3;
    let logs: Vec<T> = [0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.5]
        .into_iter()
        .map(|p| FloatElement::ln(T::from_f64(p)))
        .collect();
    let input = array([frames, 1, classes], logs.clone());
    for target in [vec![], vec![1], vec![1, 1], vec![1, 2], vec![2, 1]] {
        let mut total = 0.0;
        let mut occupancy = vec![0.0; frames * classes];
        // Enumerate all 3^3 paths, collapse adjacent repeats then remove blanks.
        // This does not use the implementation's extended-target recurrence.
        for encoded in 0..27usize {
            let path = [encoded / 9, (encoded / 3) % 3, encoded % 3];
            let collapsed: Vec<_> = path
                .iter()
                .copied()
                .enumerate()
                .filter(|&(i, label)| label != 0 && (i == 0 || path[i - 1] != label))
                .map(|(_, label)| label)
                .collect();
            if collapsed != target {
                continue;
            }
            let probability: f64 = path
                .iter()
                .enumerate()
                .map(|(time, &label)| logs[time * classes + label].to_f64().exp())
                .product();
            total += probability;
            for (time, &label) in path.iter().enumerate() {
                occupancy[time * classes + label] += probability;
            }
        }
        let normalization = target.len().max(1) as f64;
        let state = CtcState::forward(&input.view(), &target, &[frames], &[target.len()], 0)
            .expect("invariant: analytical fixture satisfies the operation boundary");
        // At most two log-adds per frame, each subtract/exp/add/ln/add,
        // plus emission, terminal merge and normalization: <=40 rounding sites.
        close(state.loss(), -total.ln() / normalization, 40);
        let mut gradient = array([frames, 1, classes], vec![T::ONE; 9]);
        state
            .backward_accumulate(T::from_usize(2), &mut gradient.view_mut())
            .expect("invariant: analytical fixture satisfies the operation boundary");
        for (&actual, posterior) in gradient.storage().as_slice().iter().zip(occupancy) {
            // Forward and suffix recurrences, normalized posterior, class sum,
            // seed, normalization and additive destination: <=64 sites.
            close(actual, 1.0 - 2.0 * posterior / total / normalization, 64);
        }
    }
}

#[test]
fn ctc_matches_independently_enumerated_paths() {
    enumerated::<f32>();
    enumerated::<f64>();
    enumerated::<F16>();
    enumerated::<Bf16>();
}

fn boundaries<T: RealScalar>() {
    let input = array([2, 2, 2], vec![FloatElement::ln(T::from_f64(0.5)); 8]);
    let state = CtcState::forward(&input.view(), &[1], &[1, 2], &[0, 1], 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    close(
        state.loss(),
        (2.0_f64.ln() + (4.0_f64 / 3.0).ln()) / 2.0,
        20,
    );
    let mut gradient = array([2, 2, 2], vec![T::from_usize(4); 8]);
    state
        .backward_accumulate(T::from_usize(2), &mut gradient.view_mut())
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
