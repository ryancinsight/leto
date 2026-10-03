//! Tests for column-pivoted (rank-revealing) QR `A P = Q R`.
//!
//! The contract asserted on every factorization (`assert_contract`) is the one
//! of ADR 0035: `A P = Q R` within the a-priori backward error, `Q`
//! orthonormal within its accumulated reflector error, `R` upper triangular,
//! and each pivot dominating the exact tail norm of every later column up to
//! the derived slack `τ` (`backward_error::col_piv_qr_pivot_slack`). The order
//! of columns whose tail norms differ by less than `τ` is not asserted; an
//! exact pivot sequence is asserted only where the norms are separated by more
//! than `τ`. The bounds are asserted where informative, as elsewhere.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::a_posteriori;
use super::backward_error::{self, informative};
use super::format::{epsilon, Format};
use eunomia::{Bf16, F16};
use leto::{Array, Array2, Storage};
use leto_ops::{col_piv_qr, solve_least_squares, Xorshift64};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

#[track_caller]
fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

fn matrix_of<T: Format>(rows: usize, columns: usize, values: &[f64]) -> Array2<T> {
    assert_eq!(values.len(), rows * columns);
    Array2::from_shape_vec(
        [rows, columns],
        values.iter().map(|&value| T::from_f64(value)).collect(),
    )
    .unwrap()
}

fn image<T: Format>(array: &Array2<T>) -> Vec<f64> {
    array
        .storage()
        .as_slice()
        .iter()
        .map(|value| value.to_f64())
        .collect()
}

/// Asserts the pivoted-QR contract on `matrix` and returns the permutation
/// and the rank.
#[track_caller]
fn assert_contract<T: Format>(matrix: &Array2<T>) -> (Vec<usize>, usize) {
    let [rows, columns] = matrix.shape();
    let name = core::any::type_name::<T>();
    let eps = epsilon::<T>();
    let values = image(matrix);
    let decomposition = col_piv_qr(&matrix.view()).unwrap();
    let (q, r) = (image(&decomposition.q()), image(&decomposition.r()));
    let permutation = decomposition.permutation();
    let rank = decomposition.rank();

    let mut sorted_permutation = permutation.to_vec();
    sorted_permutation.sort_unstable();
    assert_eq!(
        sorted_permutation,
        (0..columns).collect::<Vec<_>>(),
        "{name}"
    );
    // The sub-diagonal is zeroed explicitly (`decompose.rs`); the reflectors
    // leave rounding there, which the residual and dominance bounds absorb.
    for row in 0..rows {
        for column in 0..columns.min(row) {
            assert_eq!(r[row * columns + column], 0.0, "{name}: R[{row},{column}]");
        }
    }

    // ‖A·P − Q·R‖_F ≤ η‖A‖_F, with Q·R evaluated as Q·I·(Rᵀ)ᵀ in f64.
    if let Some(eta) = informative(backward_error::col_piv_qr(rows, columns, eps)) {
        let mut permuted = vec![0.0; rows * columns];
        let mut r_transposed = vec![0.0; rows * columns];
        let mut identity = vec![0.0; rows * rows];
        for row in 0..rows {
            identity[row * rows + row] = 1.0;
            for column in 0..columns {
                permuted[row * columns + column] = values[row * columns + permutation[column]];
                r_transposed[column * rows + row] = r[row * columns + column];
            }
        }
        let residual =
            a_posteriori::residual(&permuted, &q, &identity, &r_transposed, rows, columns, rows);
        let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(
            residual <= eta * norm,
            "{name}: ‖A·P − Q·R‖_F {residual:e} exceeds derived bound {:e}",
            eta * norm
        );
    }

    // ‖QᵀQ − I‖_F ≤ 2e + e², e = √rows·((1 + η_h)^steps − 1): ADR 0035,
    // orthonormality bound.
    let reflector_error = backward_error::householder(rows, eps);
    let steps = rows.min(columns) as f64;
    let q_error = (rows as f64).sqrt() * (steps * reflector_error.ln_1p()).exp_m1();
    if let Some(bound) = informative(2.0 * q_error + q_error * q_error) {
        let defect = a_posteriori::gram_defect(&q, rows, rows);
        assert!(
            defect <= bound,
            "{name}: ‖QᵀQ − I‖_F {defect:e} exceeds derived bound {bound:e}"
        );
    }

    // Every pivot dominates the exact tail of every later column up to τ; the
    // f64 evaluation of both sums adds γ_{rows+2}.
    if let Some(slack) = pivot_slack(rows, columns, eps) {
        let g = backward_error::gamma(rows as f64 + 2.0, f64::EPSILON);
        for k in 0..rank {
            let pivot = r[k * columns + k].powi(2);
            for j in (k + 1)..columns {
                let tail: f64 = (k..rows).map(|i| r[i * columns + j].powi(2)).sum();
                assert!(
                    tail * (1.0 - g) <= pivot * (1.0 + g) * (1.0 + slack),
                    "{name}: step {k}, column {j}: tail {tail:e} exceeds pivot {pivot:e}·(1 + {slack:e})"
                );
            }
        }
    }
    (permutation.to_vec(), rank)
}

fn pivot_slack(rows: usize, columns: usize, eps: f64) -> Option<f64> {
    informative(backward_error::col_piv_qr_pivot_slack(rows, columns, eps))
}

fn random_matrix<T: Format>(rows: usize, columns: usize, seed: u64) -> Array2<T> {
    let mut rng = Xorshift64::new(seed);
    let values: Vec<f64> = (0..rows * columns)
        .map(|_| 2.0 * rng.next_unit_f64() - 1.0)
        .collect();
    matrix_of(rows, columns, &values)
}

#[test]
fn col_piv_qr_reconstructs_a_p() {
    let values = [
        4.0, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
    ];
    let a = Array2::from_shape_vec([4, 3], values.to_vec()).unwrap();
    let f = col_piv_qr(&a.view()).unwrap();
    assert_eq!(f.q().shape(), [4, 4]);
    assert_eq!(f.r().shape(), [4, 3]);

    fn check<T: Format>(values: &[f64]) {
        let (_, rank) = assert_contract(&matrix_of::<T>(4, 3, values));
        assert_eq!(rank, 3, "{}", core::any::type_name::<T>());
    }
    check::<f64>(&values);
    check::<f32>(&values);
    check::<F16>(&values);
    check::<Bf16>(&values);
}

#[test]
fn col_piv_qr_least_squares_matches_qr_and_normal_equations() {
    // Overdetermined, full column rank.
    let (m, n) = (4, 2);
    let a_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let b_vals = vec![6.0, 5.0, 7.0, 10.0];
    let a = Array2::from_shape_vec([m, n], a_vals).unwrap();
    let b = Array::from_shape_vec([m], b_vals).unwrap();

    let x = col_piv_qr(&a.view())
        .unwrap()
        .solve_least_squares(&b.view())
        .unwrap();

    // Same least-squares problem as the plain QR solver.
    let x_qr = solve_least_squares(&a.view(), &b.view()).unwrap();
    assert_close_slice(x.storage().as_slice(), x_qr.storage().as_slice());

    // Hand-computed normal-equations solution:
    // AᵀA = [[4,10],[10,30]], Aᵀb = [28,77]
    // det(AᵀA) = 120-100 = 20
    // (AᵀA)⁻¹ = (1/20)[[30,-10],[-10,4]]
    // x = (AᵀA)⁻¹Aᵀb = (1/20)[30*28-10*77, -10*28+4*77] = (1/20)[840-770, -280+308]
    //   = (1/20)[70, 28] = [3.5, 1.4]
    assert_close_slice(x.storage().as_slice(), &[3.5, 1.4]);
}

#[test]
fn col_piv_qr_reveals_rank_deficiency() {
    // Column 2 = column 0 + column 1 ⇒ rank 2 (of 3). The tail left after two
    // steps is rounding noise of order ε‖A‖, below the 1e-12 relative rank
    // threshold in f64 only, so the rank is asserted there.
    let (m, n) = (4, 3);
    let values = vec![1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0];
    let a = Array2::from_shape_vec([m, n], values.clone()).unwrap();
    let f = col_piv_qr(&a.view()).unwrap();
    let (_, rank) = assert_contract(&a);
    assert_eq!(rank, 2);
    assert_eq!(f.rank(), 2);
    assert_contract(&matrix_of::<f32>(m, n, &values));
    assert_contract(&matrix_of::<F16>(m, n, &values));
    assert_contract(&matrix_of::<Bf16>(m, n, &values));
    // Rank-deficient least squares is rejected (not silently wrong).
    let b = Array::from_shape_vec([m], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(f.solve_least_squares(&b.view()).is_err());
}

/// Scaled Hadamard columns: orthogonal with squared norms `4·s²` for
/// `s = 1, 4, 2, 8`, so every step's tails stay separated by a factor of 4.
#[test]
fn col_piv_qr_orders_well_separated_norms_exactly() {
    const SCALES: [f64; 4] = [1.0, 4.0, 2.0, 8.0];
    const HADAMARD: [[f64; 4]; 4] = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    let values: Vec<f64> = HADAMARD
        .iter()
        .flat_map(|row| row.iter().zip(SCALES).map(|(h, s)| h * s))
        .collect();

    fn check<T: Format>(values: &[f64]) {
        let (permutation, rank) = assert_contract(&matrix_of::<T>(4, 4, values));
        assert_eq!(rank, 4, "{}", core::any::type_name::<T>());
        // The exact order is forced where τ < 3 < the gap ratio 4 − 1.
        if pivot_slack(4, 4, epsilon::<T>()).is_some_and(|slack| slack < 3.0) {
            assert_eq!(permutation, [3, 1, 2, 0], "{}", core::any::type_name::<T>());
        }
    }
    check::<f64>(&values);
    check::<f32>(&values);
    check::<F16>(&values);
    check::<Bf16>(&values);
}

/// Cancellation past `TOL3Z` forces a recompute that only the pivot reveals.
///
/// With `δ² = ε/4 < u`, the squared norm of `A = [1, δ, 0]` rounds to 1;
/// removing its leading 1 downdates it to 0 although its exact tail is `δ²`.
/// The competitor `B = [0, δ/2, δ/2]` loses nothing in the first step and keeps
/// its exact key `δ²/2`. After the recompute `A` (key `δ²`) outranks `B`
/// (`δ²/2`) by a factor of 2, far above the `O(u)` error of both keys; on the
/// stale key 0 the pivot would be `B` and the permutation `[0, 1, 2]`.
#[test]
fn col_piv_qr_recomputes_a_cancelled_partial_norm() {
    fn check<T: Format>() {
        let delta = epsilon::<T>().sqrt() / 2.0;
        let values = [
            2.0,
            0.0,
            1.0,
            0.0,
            delta / 2.0,
            delta,
            0.0,
            delta / 2.0,
            0.0,
        ];
        let (permutation, rank) = assert_contract(&matrix_of::<T>(3, 3, &values));
        assert_eq!(permutation, [0, 2, 1], "{}", core::any::type_name::<T>());
        assert_eq!(rank, 3, "{}", core::any::type_name::<T>());
    }
    check::<f64>();
    check::<f32>();
    check::<F16>();
    check::<Bf16>();
}

/// `TOL3Z = √u` (`u = ε/2`, `f64`), not `ε`, `u`, 0, or the root of the ratio.
///
/// No reflector fires (every pivot column is already on `e₁`), so `R` is the
/// permuted `A` and only the cached keys round. Column 2 is `[1, d, 0]`,
/// `d = 1e-7`: its key `1 + d²` rounds to `1 + 45·ε` (`d²` is 45.04 ulps at 1),
/// and removing the 1 at step 0 leaves the stale estimate `45ε = 9.992e-15`
/// against the exact tail `d² = 1e-14`. The ratio to the reference, 9.992e-15,
/// is below `√u = 1.05e-8` but above `ε`, `u` and 0, and its root 1.0e-7 is
/// above `√u`, so only `TOL3Z = √u` compared with the unrooted ratio
/// recomputes. Column 1 is `[0, b, 0]` with `b² = 9.996e-15` between the stale
/// and the exact key. Recomputed, column 2 is the step-1 pivot by the relative
/// gap `d²/b² − 1 = 4.0e-4`, above `τ = 7.0e-6` (3×3); on the stale key the
/// pivot is column 1 and the dominance clause fails. The reference equals the
/// current key at step 0, so this fixture does not test the denominator.
#[test]
fn col_piv_qr_recomputes_below_the_root_of_the_unit_roundoff() {
    let (d, b) = (1.0e-7_f64, 9.996e-15_f64.sqrt());
    let gap = (d * d) / (b * b) - 1.0;
    assert!(pivot_slack(3, 3, f64::EPSILON).is_some_and(|slack| slack < gap));
    let values = [2.0, 0.0, 1.0, 0.0, b, d, 0.0, 0.0, 0.0];
    let (permutation, rank) = assert_contract(&matrix_of::<f64>(3, 3, &values));
    assert_eq!(permutation, [0, 2, 1]);
    assert_eq!(rank, 2);
}

/// The recompute test divides by the reference key `vn2²`, not the current
/// key `vn1²` (`dlaqp2.f` lines 236-249).
///
/// No reflector fires, as above. Column 2 is `[1, a, d, 0]` with `a² = 1e-5`,
/// `d = 1.0074e-6` (`d² = 1.0149e-12`); its reference is `1 + a² + d²`,
/// rounded by up to `ε` absolutely (two additions). Step 0 removes the 1: the estimate
/// `a² + d²` over the reference is 1e-5, above `√u = 1.05e-8`, so it is kept
/// as the current key while the reference stays near 1. Step 1 removes `a`:
/// the estimate is `d²` plus the carried rounding, `d²·(1 − 4.47e-5)`. Over the
/// reference it is 1.0e-12 and recomputes to `d²`; over the current key 1e-5 it
/// would be 1.0e-7 and keep the stale value. Column 3 is `[0, 0, b, 0]` with
/// `b² = d²·(1 − 2.5e-5)` exact, between the stale and the exact key, so the
/// step-2 pivot is column 2 by the gap `d²/b² − 1 = 2.5e-5`, above
/// `τ = 1.07e-5` (4×4), and column 3 on the stale key, which fails dominance.
/// The threshold mutants of the fixture above also keep the stale key here.
#[test]
fn col_piv_qr_measures_the_downdate_against_the_reference_norm() {
    let (a, d) = (1.0e-5_f64.sqrt(), 1.0074e-6_f64);
    let b = d * (1.0 - 2.5e-5_f64).sqrt();
    let gap = (d * d) / (b * b) - 1.0;
    assert!(pivot_slack(4, 4, f64::EPSILON).is_some_and(|slack| slack < gap));
    let values = [
        4.0, 0.0, 1.0, 0.0, 0.0, 2.0, a, 0.0, 0.0, 0.0, d, b, 0.0, 0.0, 0.0, 0.0,
    ];
    let (permutation, rank) = assert_contract(&matrix_of::<f64>(4, 4, &values));
    assert_eq!(permutation, [0, 1, 2, 3]);
    assert_eq!(rank, 3);
}

/// Tails that differ by less than `τ` (`3ε/4` and `ε` relative, ADR 0035):
/// the order is unspecified, the factorization contract is not.
#[test]
fn col_piv_qr_near_tied_tails_preserve_the_factorization() {
    fn check<T: Format>() {
        // At 2 the spacing is 2ε and 3ε/4 stays below the midpoint; at 1 it
        // exceeds the ε/2 midpoint. The downdated squared norms tie at 1
        // although the exact tails are 1 and 1 + ε.
        let delta = (0.75 * epsilon::<T>()).sqrt();
        let values = [2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta];
        let (_, rank) = assert_contract(&matrix_of::<T>(4, 3, &values));
        assert_eq!(rank, 3, "{}", core::any::type_name::<T>());

        // Squared norms 1 and 1 + ε: a gap of ε.
        let delta = epsilon::<T>().sqrt();
        let (_, rank) = assert_contract(&matrix_of::<T>(2, 2, &[1.0, 1.0, 0.0, delta]));
        assert_eq!(rank, 2, "{}", core::any::type_name::<T>());
    }
    check::<f64>();
    check::<f32>();
    check::<F16>();
    check::<Bf16>();
}

#[test]
fn col_piv_qr_satisfies_the_contract_on_seeded_matrices() {
    for (seed, (rows, columns)) in [(5, 5), (8, 5), (5, 8), (16, 16), (33, 17)]
        .into_iter()
        .enumerate()
    {
        let seed = 0x5EED_0000 + seed as u64;
        let (_, rank) = assert_contract(&random_matrix::<f64>(rows, columns, seed));
        assert_eq!(rank, rows.min(columns), "f64 {rows}x{columns}");
    }
    let (_, rank) = assert_contract(&random_matrix::<f32>(8, 5, 0x5EED_00FF));
    assert_eq!(rank, 5, "f32 8x5");
}
