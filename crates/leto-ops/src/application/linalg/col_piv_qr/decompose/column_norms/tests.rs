use super::{tail_norm_sq, ColumnNorms};
use crate::application::linalg::thresholds::machine_epsilon;
use crate::domain::real::RealScalar;
use eunomia::{Bf16, F16};

fn downdate_keeps_the_estimate_and_the_reference<T: RealScalar>() {
    // Columns [·, 1, 2] and the removed row [·, 1]: the estimate 5 - 1 = 4 is
    // 4/5 of the reference, above every format's TOL3Z = sqrt(eps/2) < 0.1.
    let r = vec![T::ZERO, T::ONE, T::ZERO, T::from_count(2)];
    let mut norms = ColumnNorms::new(&r, 2, 2);

    norms.remove_row(&r, 2, 2, 0);

    assert_eq!(norms.current_squared[1], T::from_count(4));
    assert_eq!(norms.reference_squared[1], T::from_count(5));
}

fn cancellation_recomputes_the_exact_tail<T: RealScalar>() {
    // delta^2 = eps/4 is below the unit roundoff eps/2, so the squared norm of
    // [1, delta] rounds to 1. Removing the leading 1 estimates 0, which is
    // below TOL3Z of the reference in every format, so the tail is recomputed.
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
    let mut norms = ColumnNorms::new(&r, 3, 3);
    assert_eq!(norms.current_squared[1], T::ONE);

    norms.remove_row(&r, 3, 3, 0);

    let exact_tail = tail_norm_sq(&r, 3, 3, 1, 1);
    assert_eq!(exact_tail, delta.mul(delta));
    assert_eq!(norms.current_squared[1], exact_tail);
    assert_eq!(norms.reference_squared[1], exact_tail);
    // The unit column has nothing removed and keeps its estimate.
    assert_eq!(norms.current_squared[2], T::ONE);
    assert_eq!(norms.reference_squared[2], T::ONE);
}

fn pivot_is_the_first_maximum<T: RealScalar>() {
    let r = vec![T::ONE, T::from_count(3), T::from_count(3)];
    let norms = ColumnNorms::new(&r, 3, 1);

    assert_eq!(norms.largest_squared(), T::from_count(9));
    assert_eq!(norms.pivot(0), 1);
    assert_eq!(norms.pivot(2), 2);
}

#[test]
fn downdates_follow_dlaqp2_across_scalar_types() {
    downdate_keeps_the_estimate_and_the_reference::<f64>();
    downdate_keeps_the_estimate_and_the_reference::<f32>();
    downdate_keeps_the_estimate_and_the_reference::<F16>();
    downdate_keeps_the_estimate_and_the_reference::<Bf16>();

    cancellation_recomputes_the_exact_tail::<f64>();
    cancellation_recomputes_the_exact_tail::<f32>();
    cancellation_recomputes_the_exact_tail::<F16>();
    cancellation_recomputes_the_exact_tail::<Bf16>();
}

#[test]
fn pivot_breaks_ties_toward_the_lowest_column_across_scalar_types() {
    pivot_is_the_first_maximum::<f64>();
    pivot_is_the_first_maximum::<f32>();
    pivot_is_the_first_maximum::<F16>();
    pivot_is_the_first_maximum::<Bf16>();
}
