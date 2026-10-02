use super::{tail_norm_sq, ColumnNorms, PartialColumnNorms};
use crate::application::linalg::householder::{apply_left, reflector};
use crate::application::linalg::thresholds::machine_epsilon;
use crate::domain::real::RealScalar;

fn assert_encloses_exact<T: RealScalar>(
    bounds: super::super::norm_bounds::TailBounds<T>,
    exact: f64,
) {
    assert!(
        bounds.lower.to_f64() <= exact && exact <= bounds.upper.to_f64(),
        "exact squared tail {exact} outside [{:?}, {:?}] for {}",
        bounds.lower,
        bounds.upper,
        core::any::type_name::<T>()
    );
}

fn householder_downdate_pivot_certification<T: RealScalar>(expect_certified: bool) {
    let four = T::from_count(4);
    let two = T::from_count(2);
    let mut r = vec![T::ZERO, four, T::ZERO, T::ONE, T::ONE, two];
    let mut norms = PartialColumnNorms::new(&r, 3, 2);
    let reflection = reflector(&[T::ZERO, T::ONE]).expect("nonzero reflector tail");
    let mut scratch = Vec::new();

    apply_left(&reflection.0, &mut r, 3, 0, 1, 3, &mut scratch);
    norms.remove_row(&r, 3, 2, 0, Some(&reflection.0));

    assert_eq!(r[1], T::ZERO.sub(T::ONE));
    assert_eq!(r[4], T::ZERO.sub(four));
    assert_eq!(r[2], T::ZERO.sub(two));
    assert_eq!(r[5], T::ZERO);
    assert_eq!(norms.current_squared[1], T::from_count(16));
    assert_eq!(norms.reference_squared[1], T::from_count(17));
    assert_eq!(norms.current_squared[2], T::ZERO);
    assert_eq!(norms.reference_squared[2], T::ZERO);

    let exact_first_tail = r[4].to_f64() * r[4].to_f64();
    let exact_second_tail = r[5].to_f64() * r[5].to_f64();
    assert_encloses_exact(
        norms.bounds[1].expect("non-cancelling reflector downdate keeps a certificate"),
        exact_first_tail,
    );
    assert_encloses_exact(
        norms.bounds[2].expect("cancelling reflector downdate recomputes a certificate"),
        exact_second_tail,
    );
    // The reflector drift bound scales with the format's unit gap. Bf16's 2⁻⁷
    // gap widens the first column's interval to [0, 64.5], so it cannot
    // separate from the cancelled column and certification fails closed;
    // exact recomputation then selects the pivot.
    let certified = norms.certified_pivot(1, 3, 1);
    if expect_certified {
        assert_eq!(certified, Some(1), "{}", core::any::type_name::<T>());
    } else {
        assert_eq!(certified, None, "{}", core::any::type_name::<T>());
    }
}

fn cancellation_downdate_recomputes_exact_tail<T: RealScalar>() {
    // δ² = ε/4 is below the downdate's reliability threshold √(ε/2) in every
    // format, so removing the leading row forces an exact recompute.
    let delta = machine_epsilon::<T>().sqrt().div(T::from_count(2));
    let r = vec![
        T::ZERO,
        T::ONE,
        T::ZERO,
        T::ZERO,
        delta,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ONE,
    ];
    let mut norms = PartialColumnNorms::new(&r, 3, 3);

    norms.remove_row(&r, 3, 3, 0, None);

    let recomputed = tail_norm_sq(&r, 3, 3, 1, 1);
    assert_eq!(norms.current_squared[1], recomputed);
    assert_eq!(norms.reference_squared[1], recomputed);
    let exact_small_tail = delta.to_f64() * delta.to_f64();
    assert_encloses_exact(
        norms.bounds[1].expect("cancellation recompute creates a fresh certificate"),
        exact_small_tail,
    );
    assert_encloses_exact(
        norms.bounds[2].expect("the separated reference column keeps a certificate"),
        1.0,
    );
    assert_eq!(norms.certified_pivot(1, 3, 1), Some(2));
}

#[test]
fn reflector_downdates_enclose_exact_tails_and_certify_pivots() {
    use eunomia::{Bf16, F16};

    householder_downdate_pivot_certification::<f64>(true);
    householder_downdate_pivot_certification::<f32>(true);
    householder_downdate_pivot_certification::<F16>(true);
    householder_downdate_pivot_certification::<Bf16>(false);
}

#[test]
fn cancellation_downdates_recompute_and_enclose_exact_tails() {
    use eunomia::{Bf16, F16};

    cancellation_downdate_recomputes_exact_tail::<f64>();
    cancellation_downdate_recomputes_exact_tail::<f32>();
    cancellation_downdate_recomputes_exact_tail::<F16>();
    cancellation_downdate_recomputes_exact_tail::<Bf16>();
}
