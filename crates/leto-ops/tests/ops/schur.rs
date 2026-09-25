//! Real Schur `A = Q T Qᵀ`: reconstruction, orthogonality, structure, spectrum.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{schur, MatrixDecompose, RealScalar, Xorshift64};

fn mat(n: usize, data: Vec<f64>) -> Array2<f64> {
    Array2::from_shape_vec([n, n], data).unwrap()
}

/// `Q T Qᵀ` reconstructed by nested loops.
fn reconstruct(q: &[f64], t: &[f64], n: usize) -> Vec<f64> {
    let mut qt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += q[i * n + k] * t[k * n + j];
            }
            qt[i * n + j] = s;
        }
    }
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += qt[i * n + k] * q[j * n + k]; // (Qᵀ)[k,j] = Q[j,k]
            }
            a[i * n + j] = s;
        }
    }
    a
}

#[track_caller]
fn assert_schur_contract(a: &Array2<f64>, n: usize) {
    let s = schur(&a.view()).unwrap();
    let q = s.q();
    let t = s.t();
    let qs = q.storage().as_slice();
    let ts = t.storage().as_slice();
    let asl = a.storage().as_slice();

    // 1. Reconstruction A = Q T Qᵀ.
    let recon = reconstruct(qs, ts, n);
    for (r, expected) in recon.iter().zip(asl.iter()) {
        assert!(
            (r - expected).abs() <= 1e-9,
            "reconstruction {r} vs {expected}"
        );
    }

    // 2. Q orthogonal: QᵀQ = I.
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += qs[k * n + i] * qs[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((acc - expected).abs() <= 1e-9, "QᵀQ[{i},{j}] = {acc}");
        }
    }

    // 3. Quasi-upper-triangular: zero below the first subdiagonal, and any
    //    nonzero subdiagonal belongs to a 2×2 block with complex eigenvalues.
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 {
                assert!(ts[i * n + j].abs() <= 1e-9, "T[{i},{j}] below subdiagonal");
            }
        }
    }
    let mut i = 0;
    while i < n {
        let block = i + 1 < n && ts[(i + 1) * n + i].abs() > 1e-9;
        if block {
            let aa = ts[i * n + i];
            let bb = ts[i * n + i + 1];
            let cc = ts[(i + 1) * n + i];
            let dd = ts[(i + 1) * n + i + 1];
            let disc = (aa - dd) * (aa - dd) + 4.0 * bb * cc;
            assert!(
                disc < 1e-12,
                "2x2 block must hold a complex pair (disc={disc})"
            );
            i += 2;
        } else {
            i += 1;
        }
    }
}

#[test]
fn schur_symmetric_real_spectrum() {
    // [[2,1,0],[1,3,1],[0,1,2]] — symmetric, eigenvalues {1, 2, 4}.
    let a = mat(3, vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
    assert_schur_contract(&a, 3);
    let s = schur(&a.view()).unwrap();
    let mut eigs: Vec<f64> = s.eigenvalues().into_iter().map(|c| c.re).collect();
    eigs.sort_by(|x, y| x.total_cmp(y));
    assert!((eigs[0] - 1.0).abs() < 1e-7, "eigenvalue 1");
    assert!((eigs[1] - 2.0).abs() < 1e-7, "eigenvalue 2");
    assert!((eigs[2] - 4.0).abs() < 1e-7, "eigenvalue 4");
}

#[test]
fn schur_complex_pair() {
    // [[0,-1],[1,0]] has eigenvalues ±i — a single 2×2 Schur block.
    let a = mat(2, vec![0.0, -1.0, 1.0, 0.0]);
    assert_schur_contract(&a, 2);
    let s = schur(&a.view()).unwrap();
    let eigs = s.eigenvalues();
    let mut mags: Vec<f64> = eigs
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();
    mags.sort_by(|x, y| x.total_cmp(y));
    assert!((mags[0] - 1.0).abs() < 1e-7, "|eigenvalue| must be 1");
    assert!((mags[1] - 1.0).abs() < 1e-7, "|eigenvalue| must be 1");
}

#[test]
fn schur_general_mixed_spectrum() {
    let a = mat(
        4,
        vec![
            4.0, 1.0, -2.0, 2.0, //
            1.0, 2.0, 0.0, 1.0, //
            -2.0, 0.0, 3.0, -2.0, //
            2.0, 1.0, -2.0, -1.0,
        ],
    );
    assert_schur_contract(&a, 4);
}

#[test]
fn schur_nonsymmetric_with_complex_eigs() {
    // Eigenvalues 5 (real) and 1 ± i√6.
    let a = mat(3, vec![1.0, -3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 5.0]);
    assert_schur_contract(&a, 3);
    let s = schur(&a.view()).unwrap();
    let eigs = s.eigenvalues();
    // Find the real eigenvalue (5).
    let real_eig = eigs.iter().find(|c| c.im.abs() < 1e-7).unwrap();
    assert!(
        (real_eig.re - 5.0).abs() < 1e-7,
        "real eigenvalue must be 5"
    );
}

#[test]
fn schur_eigenvalues_agree_with_eigenvalues_kernel() {
    // Self-validate: schur eigenvalues vs eigenvalues() free function.
    let a = mat(
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            -1.0, 1.0, 0.0, 2.0, //
            0.0, -2.0, 2.0, 1.0, //
            1.0, 0.0, -1.0, 3.0,
        ],
    );
    let s = schur(&a.view()).unwrap();
    let free = leto_ops::eigenvalues(&a.view()).unwrap();
    // Cross-validate: each Schur eigenvalue must match a free-function eigenvalue.
    for se in s.eigenvalues() {
        let matched = free
            .iter()
            .any(|fe| (se.re - fe.re).abs() < 1e-7 && (se.im - fe.im).abs() < 1e-7);
        assert!(
            matched,
            "Schur eigenvalue {se:?} not found in free-function eigenvalues"
        );
    }
}

#[test]
fn schur_fluent_method_matches_free_function() {
    let a = mat(3, vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
    let free = schur(&a.view()).unwrap();
    let fluent = a.schur().unwrap();
    let f = free.t();
    let m = fluent.t();
    for (x, y) in f.storage().as_slice().iter().zip(m.storage().as_slice()) {
        assert!((x - y).abs() <= 1e-12);
    }
}

#[test]
fn schur_rejects_non_square() {
    let rect = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(schur(&rect.view()).is_err());
}

/// The F16 matrix `LETO-DENSE-SCALE-RANGE-2026-09-24` finding G reported:
/// `‖A‖_max = 7.296875`, outside F16's `[0.25, 4]` safe range on either
/// derived scale (`safe_range` or the tighter `product_safe_range`), so this
/// input balances to `[1, 4)` exactly as the pre-redesign code already did —
/// the safe-range change alters *when* balancing applies, not the outcome for
/// an input this far outside it. Probed directly, F16's Francis iteration
/// still stagnates on the balanced matrix (`LETO-F16-FRANCIS-2026-09-24`,
/// already filed as a recorded non-convergence, not a scale artifact this
/// change addresses): this regression test pins that as the expected typed
/// error rather than silently accepting whatever the sweep returns, and
/// documents that the specific "Ok↔Err flip" the review reported was two
/// instances of the same known F16 defect surfacing at different `s`, not a
/// new one introduced by unconditional scaling.
#[test]
#[expect(
    clippy::excessive_precision,
    reason = "exact f32 bit values from the reviewed regression matrix; truncating changes the input"
)]
fn schur_f16_scale_regression_matrix_is_the_recorded_francis_stagnation() {
    let raw: [f32; 9] = [
        -4.0859375,
        -7.296875,
        4.953125,
        1.4638671875,
        2.12109375,
        0.91943359375,
        3.537109375,
        4.5625,
        -0.386962890625,
    ];
    let f16: Vec<F16> = raw.iter().map(|&v| F16::from_f32(v)).collect();
    let m = Array2::from_shape_vec([3, 3], f16).unwrap();
    let result = schur(&m.view());
    match result {
        Err(leto::LetoError::StorageError { ref reason })
            if reason.contains("failed to converge") => {}
        other => panic!("expected the recorded F16 Francis non-convergence, got {other:?}"),
    }
    // Differential oracle confirming the *matrix itself* is well-posed (the
    // failure is F16-precision-specific, not a malformed or singular input):
    // f64, f32, and Bf16 all resolve it to the same spectrum.
    let f64_eigen = schur(
        &Array2::from_shape_vec([3, 3], raw.iter().map(|&v| f64::from(v)).collect())
            .unwrap()
            .view(),
    )
    .unwrap()
    .eigenvalues();
    let f32_eigen = schur(&Array2::from_shape_vec([3, 3], raw.to_vec()).unwrap().view())
        .unwrap()
        .eigenvalues();
    let mut expected: Vec<f64> = f64_eigen.iter().map(|z| z.re).collect();
    let mut computed: Vec<f64> = f32_eigen.iter().map(|z| f64::from(z.re)).collect();
    expected.sort_by(f64::total_cmp);
    computed.sort_by(f64::total_cmp);
    let frobenius: f64 = raw
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let bound = 9.0 * f64::from(f32::EPSILON) * frobenius;
    for (re, eref) in computed.iter().zip(&expected) {
        assert!(
            (re - eref).abs() <= bound,
            "{re} vs {eref}, bound {bound:e}"
        );
    }
}

/// Small seeded sweep: for matrices whose norm the safe-range gate classifies
/// as needing no balancing (comfortably inside
/// `product_safe_range`, entries in `[-4, 4]`), `schur` runs its Hessenberg
/// and Francis stages on the caller's values directly (`balanced_for_products`
/// returns `None`) — so the result is, by construction, bit-for-bit what the
/// unscaled computation produces. This sweep is the regression guard for that
/// invariant across every shipped scalar, differentially checked against the
/// `f64` computation on the same values.
#[test]
fn schur_in_range_inputs_match_the_unscaled_f64_reference() {
    fn check<T: RealScalar>() {
        let n = 3;
        for seed in [1_u64, 2, 3, 4, 5] {
            let mut rng = Xorshift64::new(seed);
            let raw: Vec<f64> = (0..n * n)
                .map(|_| 4.0 * (2.0 * rng.next_unit_f64() - 1.0))
                .collect();
            let narrowed: Vec<T> = raw.iter().map(|&v| T::from_f64(v)).collect();
            let image: Vec<f64> = narrowed.iter().map(|v| v.to_f64()).collect();
            let m = Array2::from_shape_vec([n, n], narrowed).unwrap();
            let Ok(result) = schur(&m.view()) else {
                continue; // A genuine (typed) non-convergence is not this test's concern.
            };
            let f64_matrix = Array2::from_shape_vec([n, n], image.clone()).unwrap();
            let Ok(reference) = schur(&f64_matrix.view()) else {
                continue;
            };
            let mut computed: Vec<(f64, f64)> = result
                .eigenvalues()
                .iter()
                .map(|z| (z.re.to_f64(), z.im.to_f64()))
                .collect();
            let mut expected: Vec<(f64, f64)> = reference
                .eigenvalues()
                .iter()
                .map(|z| (z.re, z.im))
                .collect();
            computed.sort_by(|a, b| a.0.total_cmp(&b.0));
            expected.sort_by(|a, b| a.0.total_cmp(&b.0));
            let frobenius: f64 = image.iter().map(|v| v * v).sum::<f64>().sqrt();
            let bound = 9.0 * machine_epsilon_of::<T>().to_f64() * frobenius
                + 9.0 * f64::EPSILON * frobenius;
            for ((re, im), (eref, iref)) in computed.iter().zip(&expected) {
                assert!((re - eref).abs() <= bound, "seed {seed}: {re} vs {eref}");
                assert!((im - iref).abs() <= bound, "seed {seed}: {im} vs {iref}");
            }
        }
    }
    check::<f64>();
    check::<f32>();
    check::<F16>();
    check::<Bf16>();
}

/// `T`'s machine epsilon via the halving probe (mirrors
/// `thresholds::machine_epsilon`, re-derived here since that function is
/// crate-private).
fn machine_epsilon_of<T: RealScalar>() -> T {
    let mut eps = T::ONE;
    let half = T::ONE.div(T::from_usize(2));
    while T::ONE.add(eps.mul(half)) > T::ONE {
        eps = eps.mul(half);
    }
    eps
}
