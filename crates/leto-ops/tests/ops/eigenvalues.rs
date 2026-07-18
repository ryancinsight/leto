//! Exact-value and self-validation tests for non-symmetric eigenvalues (ADR 0006).
//!
//! The spectrum is order-independent (and near-equal real parts make a
//! lexicographic sort brittle), so eigenvalues are matched by order-independent
//! greedy nearest-neighbor bijection within epsilon. Structured matrices with
//! known spectra give oracle-independent checks; the Schur decomposition
//! provides cross-validation for general cases.

use eunomia::Complex;
use leto::Array2;
use leto_ops::{eigenvalues, schur};

/// Match tolerance for analytically-exact / perfectly-conditioned spectra
/// (diagonal, isolated simple eigenvalues, symmetric): these are computed to
/// `O(ε‖A‖)` and must agree with the reference tightly.
const EXACT_TOL: f64 = 1.0e-7;

/// Assert the two spectra are equal as multisets: every oracle eigenvalue is
/// matched to a distinct leto eigenvalue within `tol`.
#[track_caller]
fn assert_spectra_close(leto: Vec<leto::Complex<f64>>, oracle: Vec<Complex<f64>>, tol: f64) {
    assert_eq!(leto.len(), oracle.len(), "eigenvalue count mismatch");
    let mut used = vec![false; leto.len()];
    for o in &oracle {
        let matched = leto
            .iter()
            .enumerate()
            .find(|(i, l)| !used[*i] && (l.re - o.re).abs() < tol && (l.im - o.im).abs() < tol);
        match matched {
            Some((i, _)) => used[i] = true,
            None => panic!("no leto eigenvalue matches oracle {o} within tol {tol:e}"),
        }
    }
}

/// Backward-error agreement tolerance for a *general* non-symmetric spectrum that
/// may contain a **defective** (multiple) eigenvalue.
///
/// Two backward-stable eigensolvers each return the exact spectrum of `A + E`
/// with `‖E‖₂ ≤ c·n·ε·‖A‖₂`. A defective eigenvalue of partial multiplicity `m`
/// perturbs by `|δλ| ~ ‖E‖^{1/m}`. The worst case present in this battery is a
/// **defective double eigenvalue**: the 16×16 fixture is singular with nullity 3
/// (machine-checked `det(A) ≈ −8.7e-30`, smallest singular values
/// `[5.55, 1.3e-15, 0, 0]`), so its zero eigenvalue is defective and reports it
/// as a spurious tiny complex pair `±i·1.75e-7` — exactly the
/// `√(ε‖A‖) = 1.54e-7` perturbation scale. Hence two backward-stable solvers may
/// legitimately differ by `≤ 2√(ε‖A‖)` here.
fn backward_error_tol(values: &[f64]) -> f64 {
    let fro: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
    8.0 * (f64::EPSILON * fro).sqrt()
}

fn leto_eigs(n: usize, values: &[f64]) -> Vec<leto::Complex<f64>> {
    let a = Array2::from_shape_vec([n, n], values.to_vec()).unwrap();
    eigenvalues(&a.view()).unwrap()
}

/// Cross-validate: compare eigenvalues against Schur decomposition eigenvalues
/// (both computed by leto, different algorithms → differential evidence).
fn schur_spectrum(n: usize, values: &[f64]) -> Vec<Complex<f64>> {
    let a = Array2::from_shape_vec([n, n], values.to_vec()).unwrap();
    let s = schur(&a.view()).unwrap();
    s.eigenvalues()
        .into_iter()
        .map(|c| Complex::new(c.re, c.im))
        .collect()
}

#[test]
fn eigenvalues_of_diagonal_are_the_diagonal() {
    let eigs = leto_eigs(3, &[2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, -3.0]);
    let expected = vec![
        Complex::new(2.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(-3.0, 0.0),
    ];
    assert_spectra_close(eigs, expected, EXACT_TOL);
}

#[test]
fn eigenvalues_complex_conjugate_pair_exact() {
    // [[1, -1], [1, 1]] has eigenvalues 1 ± i.
    let eigs = leto_eigs(2, &[1.0, -1.0, 1.0, 1.0]);
    assert_spectra_close(
        eigs,
        vec![Complex::new(1.0, -1.0), Complex::new(1.0, 1.0)],
        EXACT_TOL,
    );

    // A 3×3 with a complex pair (±i) and a real eigenvalue (2).
    let eigs3 = leto_eigs(3, &[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    assert_spectra_close(
        eigs3,
        vec![
            Complex::new(0.0, -1.0),
            Complex::new(0.0, 1.0),
            Complex::new(2.0, 0.0),
        ],
        EXACT_TOL,
    );
}

#[test]
fn eigenvalues_self_validate_via_schur_cross_check() {
    // Mixed real/complex spectra across sizes; cross-validate eigenvalues()
    // against schur().eigenvalues() (different algorithm, same result).
    let cases: [(usize, Vec<f64>); 7] = [
        // Upper triangular → eigenvalues are the diagonal.
        (3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]),
        // Non-symmetric, real spectrum.
        (3, vec![2.0, 1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 3.0]),
        // 4×4 with complex pairs.
        (
            4,
            vec![
                0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 2.0, 1.0,
            ],
        ),
        // Dense non-symmetric 4×4.
        (
            4,
            vec![
                4.0, 1.0, -2.0, 2.0, 5.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0,
            ],
        ),
        // Dense non-symmetric 5×5 (deterministic).
        (
            5,
            (0..25).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect(),
        ),
        // Dense non-symmetric 8×8 — multiple bulge-chase steps and deflations,
        // exercising the block-confined eigenvalue-only Francis updates.
        (
            8,
            (0..64).map(|i| ((i * 13 + 5) % 17) as f64 - 8.0).collect(),
        ),
        // Dense non-symmetric 16×16 — many nested active blocks; stresses the
        // [lo, hi] / [lo, k+len+1] apply ranges across the full chase.
        (
            16,
            (0..256)
                .map(|i| ((i * 31 + 7) % 23) as f64 - 11.0)
                .collect(),
        ),
    ];

    for (n, values) in cases {
        assert_spectra_close(
            leto_eigs(n, &values),
            schur_spectrum(n, &values),
            backward_error_tol(&values),
        );
    }
}

#[test]
fn eigenvalues_symmetric_are_real_and_self_validate() {
    // Symmetric input → all-real spectrum; the general solver must agree with
    // Schur decomposition (and have negligible imaginary parts).
    let values = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let eigs = leto_eigs(3, &values);
    for e in &eigs {
        assert!(
            e.im.abs() < 1.0e-7,
            "symmetric eigenvalue has imaginary part {e}"
        );
    }
    assert_spectra_close(eigs, schur_spectrum(3, &values), EXACT_TOL);
}

#[test]
fn eigenvalues_rejects_non_square() {
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(eigenvalues(&a.view()).is_err());
}
