//! Skew-symmetric ties (`LETO-BF16-SKEW-FRANCIS-STALL-2026-09-25`): the
//! deflation stage for a stalled block and the typed non-convergence.
//!
//! The item's recorded superdiagonal (three significant figures) does not
//! reproduce the stall; the cases below come from rerunning its scan — Bf16
//! skew-symmetric tridiagonals, orders 3–8, off-diagonal magnitudes
//! log-uniform over `[10⁻⁴, 1]` (the `graded_scan` skew family), seed
//! `0x5EED_BF16`, 90,000 matrices — where 6 exhausted the iteration cap
//! before this stage and 3 still do (the unresolvable ties).

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::certified_blocks;
use crate::ops::a_posteriori::BlockEigenvalue;
use crate::ops::backward_error::{self, informative};
use crate::ops::format::epsilon;
use eunomia::{Bf16, F16};
use leto::{Array2, LetoError};
use leto_ops::{eigenvalues, schur, RealScalar};

/// The skew-symmetric tridiagonal with superdiagonal `sup` in `T`, its exact
/// `f64` image, and the image's spectrum from the `f64` Schur form.
fn skew<T: RealScalar>(sup: &[f64]) -> (Array2<T>, Vec<f64>, Vec<(f64, f64)>) {
    let n = sup.len() + 1;
    let mut values = vec![T::ZERO; n * n];
    for (i, &s) in sup.iter().enumerate() {
        values[i * n + i + 1] = T::from_f64(s);
        values[(i + 1) * n + i] = T::from_f64(-s);
    }
    let image: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();
    let reference = schur(
        &Array2::from_shape_vec([n, n], image.clone())
            .unwrap()
            .view(),
    )
    .unwrap()
    .eigenvalues()
    .iter()
    .map(|z| (z.re, z.im))
    .collect();
    (
        Array2::from_shape_vec([n, n], values).unwrap(),
        image,
        reference,
    )
}

fn nearest(reference: &[(f64, f64)], re: f64, im: f64) -> f64 {
    reference
        .iter()
        .map(|(r, i)| (re - r).hypot(im - i))
        .fold(f64::INFINITY, f64::min)
}

/// `schur` and `eigenvalues` of the skew tridiagonal `sup` in `T`, checked
/// against the spectrum of its image (`κ = 1`, the matrix is normal).
///
/// - A posteriori, every format: each eigenvalue of `T̂` within `‖E‖₂`
///   certified from the returned factors, plus the `f64` reference's
///   `η(ε₆₄)·‖Â‖_F`; `eigenvalues` (the eigenvalues-only window omits only
///   entries off every diagonal block, `francis.rs`) to the same
///   certificate plus `u = ε/2` per component for its rounding into `T`.
/// - A priori, where `backward_error::francis(n, ε(T)) < 1` (f64, f32): each
///   within `η(ε(T))·‖Â‖_F` plus the reference's error. This is what ties
///   the result to `ε(T)`; in F16 and Bf16 the derived `η` exceeds `1`
///   (`LETO-LOW-PRECISION-FACTOR-ORACLE-2026-09-25`) and only the
///   certificate applies there — the deflation threshold itself is pinned
///   per format by `francis/deflation.rs`'s unit test.
fn check<T: RealScalar>(sup: &[f64]) {
    let (matrix, image, reference) = skew::<T>(sup);
    let n = sup.len() + 1;
    let norm = image.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rho = backward_error::francis(n, f64::EPSILON) * norm;
    let label = std::any::type_name::<T>();
    let result = schur(&matrix.view()).unwrap_or_else(|error| panic!("{label} {sup:?}: {error}"));
    let (blocks, radius) = certified_blocks(&image, &result);
    let mut largest_error = 0.0_f64;
    for BlockEigenvalue { re, im, error } in blocks {
        largest_error = largest_error.max(error);
        let distance = nearest(&reference, re, im);
        assert!(
            distance <= radius + rho + error,
            "{label} {sup:?}: λ(T̂) {re}+{im}i is {distance:e} from the spectrum"
        );
    }
    let values =
        eigenvalues(&matrix.view()).unwrap_or_else(|error| panic!("{label} {sup:?}: {error}"));
    let eta = informative(backward_error::francis(n, epsilon::<T>()));
    for z in values {
        let (re, im) = (z.re.to_f64(), z.im.to_f64());
        let distance = nearest(&reference, re, im);
        let rounding = epsilon::<T>() / 2.0 * (re.abs() + im.abs());
        assert!(
            distance <= radius + rho + largest_error + rounding,
            "{label} {sup:?}: eigenvalues() {re}+{im}i is {distance:e} from the spectrum"
        );
        if let Some(eta) = eta {
            assert!(
                distance <= eta * norm + rho + rounding,
                "{label} {sup:?}: eigenvalues() {re}+{im}i beyond η·‖Â‖_F"
            );
        }
    }
}

/// Three scan matrices (orders 4, 8, 6) that exhausted the cap in Bf16:
/// after one step each holds a block `[[0, a], [−a, 0]]`, a coupling
/// `δ ≤ ulp·2a`, and another `[[0, a], [−a, 0]]`, whose clusters
/// `±i(√(a² + δ²/4) ± δ/2)` the Wilkinson shift `±ia` sits exactly between,
/// and Bf16 rounding undoes every exceptional shift (period 2). Once both
/// exceptional shifts have failed the neighbourhood test deflates `δ`. The
/// same inputs
/// in the other formats converge by the shifts alone.
#[test]
fn skew_ties_converge_once_the_exceptional_shifts_fail() {
    let cases: [&[f64]; 3] = [
        &[
            -0.018_066_406_25,
            -0.000_138_282_775_878_906_25,
            -0.018_310_546_875,
        ],
        &[
            -0.808_593_75,
            0.023_559_570_312_5,
            -0.002_380_371_093_75,
            0.004_913_330_078_125,
            0.035_888_671_875,
            0.000_169_754_028_320_312_5,
            -0.036_376_953_125,
        ],
        &[
            -0.000_169_754_028_320_312_5,
            0.001_647_949_218_75,
            0.154_296_875,
            -0.000_125_885_009_765_625,
            0.000_150_680_541_992_187_5,
        ],
    ];
    for sup in cases {
        check::<f64>(sup);
        check::<f32>(sup);
        check::<F16>(sup);
        check::<Bf16>(sup);
    }
}

/// The stage at order 67, where `dlahqr`'s budget `30·max(10, n) = 2010`
/// exceeds the former fixed cap of 2000: the order-4 tie above, an exact
/// zero coupling, then order-2 skew blocks split by zeros. The certificate
/// also checks the Schur vectors stayed orthonormal: waiting out the budget
/// before this stage left Bf16's `‖QᵀQ − I‖ > 1` here.
#[test]
fn skew_tie_converges_at_order_67() {
    let mut sup = vec![
        -0.018_066_406_25,
        -0.000_138_282_775_878_906_25,
        -0.018_310_546_875,
        0.0,
    ];
    while sup.len() < 66 {
        sup.extend([0.5, 0.0]);
    }
    sup.truncate(66);
    check::<f64>(&sup);
    check::<f32>(&sup);
    check::<F16>(&sup);
    check::<Bf16>(&sup);
}

/// A tie the neighbourhood test does not reach, and the reason it is Bf16
/// only: at the cap the iterate holds `δ = 2⁻⁹` between corners
/// `a = 0.064453125 = 33·2⁻⁹` over a diagonal of rounding residue, so
/// `δ/(2a + residue)` is `1/66 = 1.939·2⁻⁷`, `1.9375·ε` once rounded to
/// Bf16's eight significant bits — above the `ε` the test needs. The
/// iteration reports it as a typed non-convergence, never a partial Schur
/// form: `max_iters` is `dlahqr`'s budget `30·max(10, 5) = 300`.
#[test]
fn unresolvable_bf16_skew_tie_reports_its_residual() {
    let (matrix, _, _) = skew::<Bf16>(&[
        -0.064_453_125,
        0.002_151_489_257_812_5,
        -0.064_453_125,
        -0.001_113_891_601_562_5,
    ]);
    let ulp = epsilon::<Bf16>();
    for result in [
        schur(&matrix.view()).map(|_| ()),
        eigenvalues(&matrix.view()).map(|_| ()),
    ] {
        let Err(LetoError::ConvergenceError {
            max_iters,
            residual,
            tol,
        }) = result
        else {
            panic!("expected ConvergenceError, got {result:?}");
        };
        assert_eq!(max_iters, 300);
        assert_eq!(tol, ulp);
        assert_eq!(residual, 1.9375 * ulp);
    }
}
