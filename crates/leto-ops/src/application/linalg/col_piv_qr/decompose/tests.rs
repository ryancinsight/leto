use super::column_norms::{tail_norm_sq, ColumnNorms, PartialColumnNorms, RecomputedColumnNorms};
use super::{factor, factor_with_norms};
use crate::application::linalg::thresholds::{machine_epsilon, safe_min};
use crate::domain::real::RealScalar;
use leto::Array2;

fn assert_matches_recomputed<T: RealScalar>(
    case: &str,
    rows: usize,
    columns: usize,
    values: &[f64],
) {
    let matrix = Array2::from_shape_vec(
        [rows, columns],
        values.iter().copied().map(T::from_f64).collect(),
    )
    .expect("invariant: fixture count matches its matrix shape");
    let downdated = factor(&matrix.view()).expect("finite fixture must factor");
    let recomputed = factor_with_norms::<T, RecomputedColumnNorms>(&matrix.view())
        .expect("finite fixture must factor");
    assert_eq!(
        downdated.perm,
        recomputed.perm,
        "{case} permutation for {}",
        core::any::type_name::<T>()
    );
    assert_eq!(
        downdated.q,
        recomputed.q,
        "{case} Q for {}",
        core::any::type_name::<T>()
    );
    assert_eq!(
        downdated.r,
        recomputed.r,
        "{case} R for {}",
        core::any::type_name::<T>()
    );
    assert_eq!(
        downdated.rank,
        recomputed.rank,
        "{case} rank for {}",
        core::any::type_name::<T>()
    );
}

fn existing_contract_fixtures_match_recomputed<T: RealScalar>() {
    assert_matches_recomputed::<T>(
        "reconstruction",
        4,
        3,
        &[
            4.0, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
        ],
    );
    assert_matches_recomputed::<T>(
        "least squares",
        4,
        2,
        &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
    );
    assert_matches_recomputed::<T>(
        "rank deficiency",
        4,
        3,
        &[1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0],
    );
}

fn near_tied_pivots_after_downdate_follow_native_rounding<T: RealScalar>() {
    // Let ε be the spacing above one and δ² = 3ε/4. At 2, the spacing is
    // 2ε, so 2 + δ² rounds to 2; at 1, δ² exceeds the ε/2 midpoint, so
    // 1 + δ² rounds to 1 + ε. The initial norms tie, while exact tails do
    // not. With unit roundoff u = ε/2 ≤ 1/256, rounding sqrt and then its
    // square bounds δ² between (3/4)(1−u)^3 ε and (3/4)(1+u)^3 ε, inside
    // (ε/2, ε) for every supported binary format.
    let delta = machine_epsilon::<T>().mul(T::from_f64(0.75)).sqrt();
    let matrix = Array2::from_shape_vec(
        [4, 3],
        vec![
            T::from_usize(2),
            T::ONE,
            T::ONE,
            T::ZERO,
            T::ONE,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ONE,
            T::ZERO,
            T::ZERO,
            delta,
        ],
    )
    .expect("invariant: fixture count matches its matrix shape");
    let downdated = factor(&matrix.view()).expect("finite fixture must factor");
    let recomputed = factor_with_norms::<T, RecomputedColumnNorms>(&matrix.view())
        .expect("finite fixture must factor");

    assert_eq!(downdated.perm, recomputed.perm);
    assert_eq!(downdated.perm, [0, 2, 1]);
    assert_eq!(downdated.rank, 3);
    assert_eq!(recomputed.rank, 3);
    assert_eq!(downdated.q, recomputed.q);
    assert_eq!(downdated.r, recomputed.r);
}

fn cancellation_boundary_matches_recomputed<T: RealScalar>() {
    let delta = machine_epsilon::<T>().sqrt().div(T::from_usize(2));
    let matrix = Array2::from_shape_vec(
        [3, 3],
        vec![
            T::from_usize(2),
            T::ONE,
            T::ONE,
            T::ZERO,
            delta,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            delta.mul(T::from_usize(2)),
        ],
    )
    .expect("invariant: fixture count matches its matrix shape");
    let downdated = factor(&matrix.view()).expect("finite fixture must factor");
    let recomputed = factor_with_norms::<T, RecomputedColumnNorms>(&matrix.view())
        .expect("finite fixture must factor");

    assert_eq!(downdated.perm, [0, 2, 1]);
    assert_eq!(downdated.perm, recomputed.perm);
    assert_eq!(downdated.rank, recomputed.rank);
    assert_eq!(downdated.q, recomputed.q);
    assert_eq!(downdated.r, recomputed.r);
}

fn exponent_boundary_fixtures_match_recomputed<T: RealScalar>() {
    let sigma = safe_min::<T>().mul(machine_epsilon::<T>());
    let small = sigma.sqrt();
    let matrix = Array2::from_shape_vec(
        [5, 3],
        vec![
            T::from_usize(4),
            T::ONE,
            T::ONE,
            small,
            T::ONE,
            T::ONE,
            T::ZERO,
            small,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            small,
            sigma,
            T::ZERO,
            sigma,
        ],
    )
    .expect("invariant: fixture count matches its matrix shape");
    let downdated = factor(&matrix.view()).expect("finite fixture must factor");
    let recomputed = factor_with_norms::<T, RecomputedColumnNorms>(&matrix.view())
        .expect("finite fixture must factor");

    assert_eq!(downdated.perm, recomputed.perm);
    assert_eq!(downdated.rank, recomputed.rank);
    assert_eq!(downdated.q, recomputed.q);
    assert_eq!(downdated.r, recomputed.r);
}

#[test]
fn downdated_pivots_preserve_existing_contract_across_scalar_types() {
    use eunomia::{Bf16, F16};

    existing_contract_fixtures_match_recomputed::<f64>();
    existing_contract_fixtures_match_recomputed::<f32>();
    existing_contract_fixtures_match_recomputed::<F16>();
    existing_contract_fixtures_match_recomputed::<Bf16>();

    cancellation_boundary_matches_recomputed::<f64>();
    cancellation_boundary_matches_recomputed::<f32>();
    cancellation_boundary_matches_recomputed::<F16>();
    cancellation_boundary_matches_recomputed::<Bf16>();

    exponent_boundary_fixtures_match_recomputed::<f64>();
    exponent_boundary_fixtures_match_recomputed::<f32>();
    exponent_boundary_fixtures_match_recomputed::<F16>();
    exponent_boundary_fixtures_match_recomputed::<Bf16>();

    near_tied_pivots_after_downdate_follow_native_rounding::<f64>();
    near_tied_pivots_after_downdate_follow_native_rounding::<f32>();
    near_tied_pivots_after_downdate_follow_native_rounding::<F16>();
    near_tied_pivots_after_downdate_follow_native_rounding::<Bf16>();
}

#[test]
fn partial_norm_downdates_recompute_cancellation_boundary() {
    let delta = f64::EPSILON.sqrt() / 2.0;
    let r = vec![
        2.0,
        1.0,
        1.0, // Removed row.
        0.0,
        delta,
        0.0, // First remaining tail.
        0.0,
        0.0,
        2.0 * delta,
    ];
    let mut partial = PartialColumnNorms::new(&r, 3, 3);

    partial.remove_row(&r, 3, 3, 0, None);

    for column in 1..3 {
        assert_eq!(
            partial.current_squared[column],
            tail_norm_sq(&r, 3, 3, column, 1),
            "boundary trailing squared norm for column {column}"
        );
    }
}

#[test]
fn separated_pivot_is_certified_after_a_householder_update() {
    use crate::application::linalg::householder::{apply_left, reflector};

    let mut r = vec![
        4.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0,
    ];
    let mut norms = PartialColumnNorms::new(&r, 3, 4);
    let reflection = reflector(&[4.0_f64, 1.0, 0.0, 0.0]).expect("nonzero reflector tail");
    let mut scratch = Vec::new();
    apply_left(&reflection.0, &mut r, 3, 0, 1, 3, &mut scratch);

    norms.remove_row(&r, 3, 4, 0, Some(&reflection.0));
    norms.assert_bounds_cover_exact_keys(&r, 3, 4, 1);

    assert_eq!(norms.certified_pivot(1, 3, 3), Some(1));
}

fn squared_norm_order_survives_a_rounded_root_tie<T: RealScalar>() {
    let delta = machine_epsilon::<T>().sqrt();
    let matrix = Array2::from_shape_vec([2, 2], vec![T::ONE, T::ONE, T::ZERO, delta])
        .expect("invariant: fixture count matches its matrix shape");

    assert_eq!(
        factor(&matrix.view())
            .expect("finite fixture must factor")
            .perm,
        [1, 0],
        "squared column norms remain distinguishable for {}",
        core::any::type_name::<T>()
    );
}

#[test]
fn pivot_order_uses_squared_norms_across_scalar_types() {
    use eunomia::{Bf16, F16};

    squared_norm_order_survives_a_rounded_root_tie::<f64>();
    squared_norm_order_survives_a_rounded_root_tie::<f32>();
    squared_norm_order_survives_a_rounded_root_tie::<F16>();
    squared_norm_order_survives_a_rounded_root_tie::<Bf16>();
}
