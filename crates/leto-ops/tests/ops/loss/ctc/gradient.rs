//! Central-difference differentiability proof.

use super::fixtures::{array, unit_roundoff};
use leto::Storage;
use leto_ops::{ctc::CtcState, RealScalar};

fn central_difference<T: RealScalar>() {
    // Two-sided differences prove `backward_accumulate` differentiates the
    // loss rather than a self-consistent occupancy: every analytical oracle
    // above asserts the same posterior formula the implementation uses, so
    // only numerical differentiation can catch a wrong-but-consistent one.
    // f16/bf16 are excluded: their rounding floor u/h exceeds the gradient
    // scale, so no step size separates truncation from rounding there.
    let shape = [3, 2, 3];
    let targets = vec![1, 2, 1];
    let input_lengths = vec![3, 2];
    let target_lengths = vec![2, 1];
    // Deterministic nonpositive lanes with no posterior symmetries to hide
    // behind; the floor keeps every raised perturbation inside the
    // nonpositive input contract. Sample 1, frame 2 is padding: the loss
    // never reads it, so both one-sided evaluations agree bitwise and the
    // quotient is exactly zero there.
    let lane =
        |t: usize, b: usize, c: usize| -(((t * 13 + b * 7 + c * 3 + 11) % 97) as f64 * 0.05 + 0.05);
    let mut base = Vec::with_capacity(shape.iter().product());
    for t in 0..shape[0] {
        for b in 0..shape[1] {
            for c in 0..shape[2] {
                base.push(lane(t, b, c));
            }
        }
    }
    let loss_of = |values: &[f64]| {
        let owned: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
        let input = array(shape, owned);
        CtcState::forward(&input.view(), &targets, &input_lengths, &target_lengths, 0)
            .expect("invariant: perturbed fixture stays inside the operation boundary")
            .loss()
            .to_f64()
    };
    // Central differences balance truncation h^2|L'''|/6 against the
    // rounding floor u/h at h=u^(1/3); the loss's log-sum-exp derivatives
    // are posterior cumulants of order one after mean reduction, so the
    // quotient error is a small multiple of u^(2/3). The factor 16 and the
    // (1+|gradient|) scale follow the file's gamma-bound convention; the
    // non-triviality assert below keeps the bound honest. Observed: peak
    // gradient 0.325 in both precisions; worst lane uses 4.3% of the bound
    // in f32 and 1.2% in f64, so the bound catches O(1) formula defects
    // with an order of magnitude of cross-libm margin.
    let unit = unit_roundoff::<T>();
    let step = unit.cbrt();
    let bound_scale = 16.0 * (step * step + unit / step);
    let input = array(shape, base.iter().map(|&v| T::from_f64(v)).collect());
    let state = CtcState::forward(&input.view(), &targets, &input_lengths, &target_lengths, 0)
        .expect("invariant: analytical fixture satisfies the operation boundary");
    assert!(state.loss().to_f64().is_finite());
    let mut gradient = array(shape, vec![T::ZERO; base.len()]);
    state
        .backward_accumulate(T::ONE, &mut gradient.view_mut())
        .expect("invariant: analytical fixture satisfies the operation boundary");
    let analytic = gradient.storage().as_slice();
    let peak = analytic
        .iter()
        .map(|v| v.to_f64().abs())
        .fold(0.0, f64::max);
    assert!(
        peak > 100.0 * bound_scale,
        "gradient scale {peak} must dwarf the quotient bound {bound_scale}"
    );
    for (index, &expected) in analytic.iter().enumerate() {
        let mut raised = base.clone();
        raised[index] += step;
        let mut lowered = base.clone();
        lowered[index] -= step;
        let numerical = (loss_of(&raised) - loss_of(&lowered)) / (2.0 * step);
        let expected = expected.to_f64();
        let bound = bound_scale * (1.0 + expected.abs());
        assert!(
            (numerical - expected).abs() <= bound,
            "lane {index}: numerical {numerical}, analytic {expected}, bound {bound}"
        );
    }
}

#[test]
fn ctc_gradient_matches_central_differences() {
    central_difference::<f32>();
    central_difference::<f64>();
}
