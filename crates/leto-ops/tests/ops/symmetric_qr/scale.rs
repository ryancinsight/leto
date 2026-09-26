//! Dynamic range, the full per-exponent scale sweep, non-finite rejection,
//! and the safe-range-specific regressions (finding
//! `LETO-DENSE-SCALE-RANGE-2026-09-24`, item A) for the symmetric
//! tridiagonal-QL eigensolver. Split out of [`super`] to keep each file
//! near the 500-line target; shares its fixtures ([`super::fixtures`]).

use super::super::format::{epsilon, Format};
use super::fixtures::{assert_spectrum, backward_bound, certified, round_into, with_spectrum};
use eunomia::{Bf16, F16};
use leto::{Array2, LetoError, Storage};
use leto_ops::{
    symmetric_eigen_jacobi_with_tolerance, symmetric_eigen_qr, RealScalar, SymmetricEigenWorkspace,
};

fn check_dynamic_range<T: RealScalar>() {
    // Eigenvalues across twelve decades: the bound is normwise, so the small
    // ones are resolved to n²·ε·‖A‖_F absolutely, not relatively.
    let spectrum = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0];
    assert_spectrum::<T>(&with_spectrum(&spectrum, 23), &spectrum);
}

/// `s·[[2,1,1],[1,2,1],[1,1,2]]` (eigenvalues `s·{1, 1, 4}`) at every
/// power-of-two scale `T` represents: a normal-range scale must decompose
/// correctly; a scale whose entries or eigenvalues leave the normal range
/// must decompose correctly or fail with a typed error, never a wrong `Ok`.
fn check_scale_sweep<T: Format>() {
    let base = [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0];
    let n = 3;
    // Relative bounds, in units of s: the a-posteriori radius of the returned
    // pairs on `base` (the entries 2s and s are exact in every format), and
    // the a-priori backward bound of the unscaled matrix where informative.
    let a_priori = backward_bound::<T>(&base, n);
    // From seven binades into the subnormals (Bf16 has seven below its
    // smallest normal, the fewest of the four) to the largest scale whose
    // entries 2s stay finite; 4s then overflows at the top, which must be a
    // typed error.
    let (low, high) = (T::MIN_EXPONENT - 7, T::MAX_EXPONENT);
    for exponent in low..high {
        let scale = T::ONE.scale_binary(exponent);
        let values: Vec<T> = base.iter().map(|&v| T::from_f64(v).mul(scale)).collect();
        let matrix = Array2::from_shape_vec([n, n], values).unwrap();
        // Normal range: 2s at most the largest value's exponent less the
        // headroom for λ = 4s (two binades), s at least the smallest normal.
        let normal = exponent >= T::MIN_EXPONENT && exponent + 2 <= T::MAX_EXPONENT;
        match symmetric_eigen_qr(&matrix.view()) {
            Ok(eigen) => {
                let s = scale.to_f64();
                let unit = leto_ops::SymmetricEigenDecomposition {
                    eigenvalues: eigen.eigenvalues.iter().map(|v| v.to_f64() / s).collect(),
                    eigenvectors: Array2::from_shape_vec(
                        [n, n],
                        eigen
                            .eigenvectors
                            .storage()
                            .as_slice()
                            .iter()
                            .map(|v| v.to_f64())
                            .collect(),
                    )
                    .unwrap(),
                };
                let radius = certified(&base, &unit);
                for (value, expected) in eigen.eigenvalues.iter().zip([1.0, 1.0, 4.0]) {
                    let error = (value.to_f64() / s - expected).abs();
                    // The entries are powers of two and exact even when
                    // subnormal; scaling is exact, so the only extra error
                    // is rounding each computed eigenvalue back onto the
                    // subnormal grid, whose spacing is 2^MIN_EXPONENT·ε: half
                    // of it, relative to s = 2^exponent.
                    let lost = if exponent < T::MIN_EXPONENT {
                        0.5 * 2.0_f64.powi(T::MIN_EXPONENT - exponent) * epsilon::<T>()
                    } else {
                        0.0
                    };
                    for relative in [Some(radius), a_priori].into_iter().flatten() {
                        assert!(
                            error <= relative + lost,
                            "2^{exponent}: {:e} vs {expected}·s",
                            value.to_f64()
                        );
                    }
                }
            }
            Err(error) => {
                assert!(!normal, "2^{exponent}: normal-range scale failed: {error}");
                assert!(
                    matches!(
                        error,
                        LetoError::Overflow { .. } | LetoError::ConvergenceError { .. }
                    ),
                    "2^{exponent}: untyped failure {error:?}"
                );
            }
        }
    }
}

fn check_non_finite_input<T: RealScalar>() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let values: Vec<T> = [1.0, 0.0, bad, 1.0]
            .iter()
            .map(|&v| T::from_f64(v))
            .collect();
        let matrix = Array2::from_shape_vec([2, 2], values).unwrap();
        let mut workspace = SymmetricEigenWorkspace::new();
        match workspace.decompose(&matrix.view()) {
            Err(LetoError::InvalidInput(reason)) => assert!(reason.contains("(1, 0)"), "{reason}"),
            other => panic!("{bad}: expected InvalidInput, got {other:?}"),
        }
        assert_eq!(workspace.order(), 0);
    }
}

/// Finding `LETO-DENSE-SCALE-RANGE-2026-09-24`/A: a diagonal matrix whose
/// norm (its largest entry) sits far outside the LAPACK safe range, so the
/// fallback bring-to-`[1,4)` scale still applies, and can still lose the far
/// smaller entry to underflow (`f64 diag(1e300, 1e-300)` scales by roughly
/// `2⁻⁹⁹⁶`, sending `1e-300 · 2⁻⁹⁹⁶` to `0`). The module documentation states
/// this loss is bounded by the factorization's own backward error, never
/// exact — this test is the falsifiable form of that claim: the result must
/// land within the `n²·ε·‖A‖_F` envelope, which a normwise bound
/// against a `1e300`-scale norm satisfies trivially for a `1e-300`-scale
/// discrepancy, rather than the wrong, unbounded `[0, 1e300]` a defect
/// elsewhere in the pipeline (a NaN, an unrelated overflow) could also
/// produce.
fn check_far_out_of_range_entry_loss_stays_within_bound<T: RealScalar>(large: f64, small: f64) {
    let values = [large, 0.0, 0.0, small];
    let (matrix, image) = round_into::<T>(&values, 2);
    let eigen = symmetric_eigen_qr(&matrix.view()).unwrap();
    let mut expected = [image[0], image[3]];
    expected.sort_by(f64::total_cmp);
    let mut computed: Vec<f64> = eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    computed.sort_by(f64::total_cmp);
    for bound in [
        Some(certified(&image, &eigen)),
        backward_bound::<T>(&image, 2),
    ]
    .into_iter()
    .flatten()
    {
        for (value, expected) in computed.iter().zip(&expected) {
            assert!(
                (value - expected).abs() <= bound,
                "large={large} small={small}: {value} vs {expected}, bound {bound:e}"
            );
        }
    }
}

#[test]
fn symmetric_eigen_qr_far_out_of_range_entry_loss_stays_within_the_backward_bound() {
    check_far_out_of_range_entry_loss_stays_within_bound::<f64>(1e300, 1e-300);
    check_far_out_of_range_entry_loss_stays_within_bound::<f32>(1e38, 1e-38);
    check_far_out_of_range_entry_loss_stays_within_bound::<Bf16>(1e38, 1e-38);
    // F16: `32768 = 2¹⁵` exceeds the QL gate's degree-2 upper end
    // `√65504 ≈ 256`, so this also exercises the minimal move (the reported
    // `diag(32768, 1.0009765625) → 1.0`).
    check_far_out_of_range_entry_loss_stays_within_bound::<F16>(32768.0, 1.0009765625);
}

/// Finding A's other half: an input whose norm the safe-range gate classifies
/// as needing **no** balancing is factored on the caller's exact values, so
/// its eigenvalues match a same-input Jacobi decomposition (which never
/// balances) far more tightly than the normwise backward bound alone would
/// guarantee — both algorithms see the identical, unscaled entries.
fn check_in_range_norm_matches_unscaled_jacobi<T: RealScalar>() {
    // diag(1, 1e-6): norm 1 is inside every shipped format's range for both
    // the QL solver's degree-2 gate and Jacobi's degree-1 gate (F16's
    // narrowest is QL's `[0.25, 256)`); 1e-6 is representable (as a
    // subnormal in F16) without underflowing to zero.
    let values = [1.0_f64, 0.0, 0.0, 1e-6];
    let (matrix, image) = round_into::<T>(&values, 2);
    let qr_eigen = symmetric_eigen_qr(&matrix.view()).unwrap();
    let jacobi_eigen =
        symmetric_eigen_jacobi_with_tolerance(&matrix.view(), T::from_f64(epsilon::<T>())).unwrap();
    let mut qr: Vec<f64> = qr_eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    let mut jacobi: Vec<f64> = jacobi_eigen
        .eigenvalues
        .iter()
        .map(|v| v.to_f64())
        .collect();
    qr.sort_by(f64::total_cmp);
    jacobi.sort_by(f64::total_cmp);
    let mut expected = [image[0], image[3]];
    expected.sort_by(f64::total_cmp);
    // Both are exact for a diagonal input (no rotation changes an
    // already-diagonal matrix), so all three agree bit for bit.
    assert_eq!(qr, expected);
    assert_eq!(jacobi, expected);
}

#[test]
fn symmetric_eigen_qr_in_range_norm_matches_unscaled_jacobi() {
    check_in_range_norm_matches_unscaled_jacobi::<f64>();
    check_in_range_norm_matches_unscaled_jacobi::<f32>();
    check_in_range_norm_matches_unscaled_jacobi::<F16>();
    check_in_range_norm_matches_unscaled_jacobi::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_spans_a_wide_dynamic_range_for_every_scalar() {
    check_dynamic_range::<f64>();
    check_dynamic_range::<f32>();
    check_dynamic_range::<F16>();
    check_dynamic_range::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_is_correct_or_typed_across_each_exponent_range() {
    check_scale_sweep::<f64>();
    check_scale_sweep::<f32>();
    check_scale_sweep::<F16>();
    check_scale_sweep::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_rejects_non_finite_input_for_every_scalar() {
    check_non_finite_input::<f64>();
    check_non_finite_input::<f32>();
    check_non_finite_input::<F16>();
    check_non_finite_input::<Bf16>();
}
