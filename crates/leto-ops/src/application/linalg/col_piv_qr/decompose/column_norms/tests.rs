use super::{tail_norm_sq, ColumnNorms, PartialColumnNorms};
use crate::application::linalg::householder::{apply_left, reflector};
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

fn householder_downdate_preserves_certification<T: RealScalar>() {
    let four = T::from_usize(4);
    let two = T::from_usize(2);
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
    assert_eq!(norms.current_squared[1], T::from_usize(16));
    assert_eq!(norms.reference_squared[1], T::from_usize(17));
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
    assert_eq!(norms.certified_pivot(1, 3, 1), Some(1));
}

fn cancellation_downdate_recomputes_exact_tail<T: RealScalar>() {
    let delta = T::ONE.scale_binary(-12);
    let mut r = vec![
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

    householder_downdate_preserves_certification::<f64>();
    householder_downdate_preserves_certification::<f32>();
    householder_downdate_preserves_certification::<F16>();
    householder_downdate_preserves_certification::<Bf16>();
}

#[test]
fn cancellation_downdates_recompute_and_enclose_exact_tails() {
    use eunomia::{Bf16, F16};

    cancellation_downdate_recomputes_exact_tail::<f64>();
    cancellation_downdate_recomputes_exact_tail::<f32>();
    cancellation_downdate_recomputes_exact_tail::<F16>();
    cancellation_downdate_recomputes_exact_tail::<Bf16>();
}
