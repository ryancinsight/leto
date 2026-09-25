//! Real Schur `A = Q T Qᵀ`: reconstruction, orthogonality, structure, spectrum.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::backward_error;
use super::format::epsilon;
use super::spectral_condition::bauer_fike_factor;
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

/// The matrix `LETO-DENSE-SCALE-RANGE-2026-09-24` finding G reported (exact
/// in F16; `Bf16` rounds it). It stalled the unscaled Francis step in F16
/// before the kernels formed their products scale-safely; it now converges
/// in every format, checked against the Bauer–Fike bound
/// `|λ̂ − λ| ≤ κ·(δ + η·‖Â‖_F) + ρ + ε(T)·|λ|`: `κ ≥ κ₂(V)` from the
/// matrix's own left and right eigenvectors in `f64`, computed independently
/// of `schur` (`spectral_condition`); `δ = ‖Â − A‖_F` the rounding of the
/// input into `T`; `η·‖Â‖_F` the derived Francis backward error
/// (`backward_error::francis`); `ρ = κ·η(ε₆₄)·‖A‖_F` the `f64` reference's own error;
/// and `ε(T)·|λ|` the rounding of each eigenvalue onto `T`'s grid.
#[test]
#[expect(
    clippy::excessive_precision,
    reason = "exact f32 bit values from the reviewed regression matrix; truncating changes the input"
)]
fn schur_scale_regression_matrix_converges_in_every_format() {
    const RAW: [f32; 9] = [
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
    let exact: [f64; 9] = RAW.map(f64::from);
    let frobenius = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let reference = schur(
        &Array2::from_shape_vec([3, 3], exact.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap()
    .eigenvalues();
    let mut eigenvalues = [0.0; 3];
    for (slot, z) in eigenvalues.iter_mut().zip(&reference) {
        assert_eq!(z.im, 0.0, "the regression matrix has a real spectrum");
        *slot = z.re;
    }
    eigenvalues.sort_by(f64::total_cmp);
    let kappa = bauer_fike_factor(&exact, &eigenvalues.map(|re| (re, 0.0)));
    let reference_error = kappa * backward_error::francis(3, f64::EPSILON) * frobenius(&exact);

    fn check<T: RealScalar>(exact: &[f64; 9], eigenvalues: &[f64; 3], kappa: f64, rho: f64) {
        let narrowed: Vec<T> = exact.iter().map(|&v| T::from_f64(v)).collect();
        let image: Vec<f64> = narrowed.iter().map(|v| v.to_f64()).collect();
        let delta = image
            .iter()
            .zip(exact)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let eps = epsilon::<T>();
        let norm = image.iter().map(|v| v * v).sum::<f64>().sqrt();
        let result = schur(&Array2::from_shape_vec([3, 3], narrowed).unwrap().view())
            .unwrap_or_else(|error| panic!("schur must converge: {error}"));
        let mut computed: Vec<(f64, f64)> = result
            .eigenvalues()
            .iter()
            .map(|z| (z.re.to_f64(), z.im.to_f64()))
            .collect();
        computed.sort_by(|a, b| a.0.total_cmp(&b.0));
        for ((re, im), expected) in computed.into_iter().zip(eigenvalues) {
            let bound = kappa * (delta + backward_error::francis(3, eps) * norm)
                + rho
                + eps * expected.abs();
            assert!(
                (re - expected).abs() <= bound && im.abs() <= bound,
                "{re}+{im}i vs {expected}, bound {bound:e} (κ {kappa})"
            );
        }
    }
    check::<f64>(&exact, &eigenvalues, kappa, reference_error);
    check::<f32>(&exact, &eigenvalues, kappa, reference_error);
    check::<F16>(&exact, &eigenvalues, kappa, reference_error);
    check::<Bf16>(&exact, &eigenvalues, kappa, reference_error);
}

/// Small seeded sweep of random nonsymmetric 3×3 matrices, entries in
/// `[-4, 4]`, differentially checked against the `f64` computation on the
/// same values. By Bauer–Fike each computed spectrum is within
/// `κ·η·‖Â‖_F` of the exact spectrum of `Â`, `κ ≥ κ₂(V)` computed from
/// `Â`'s own left and right eigenvectors in `f64` (`spectral_condition`,
/// complex eigenvalues included) and `η` the derived Francis backward error
/// (`backward_error::francis`); the two computations are therefore within
/// `κ·(η(ε(T)) + η(ε₆₄))·‖Â‖_F` of each other.
#[test]
fn schur_matches_the_f64_reference_within_the_bauer_fike_bound() {
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
            let result = schur(&m.view()).unwrap_or_else(|error| panic!("seed {seed}: {error}"));
            let f64_matrix = Array2::from_shape_vec([n, n], image.clone()).unwrap();
            let reference = schur(&f64_matrix.view()).unwrap();
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
            let order =
                |a: &(f64, f64), b: &(f64, f64)| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1));
            computed.sort_by(order);
            expected.sort_by(order);
            let frobenius: f64 = image.iter().map(|v| v * v).sum::<f64>().sqrt();
            let exact: [f64; 9] = image.clone().try_into().unwrap();
            let spectrum: [(f64, f64); 3] = expected.clone().try_into().unwrap();
            let kappa = bauer_fike_factor(&exact, &spectrum);
            let bound = kappa
                * (backward_error::francis(n, epsilon::<T>())
                    + backward_error::francis(n, f64::EPSILON))
                * frobenius;
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

/// The clustered symmetric family at the Francis gate's lower end: `2ᵉ·Q·
/// diag(1, 1, 1.001, 1.001, 1.002, 1.002, 1.003, 0.5)·Qᵀ`, `n = 8`, which the
/// minimal move lands exactly on the gate's lower end for `e ≤ −485`. The
/// 2×2 standardization formed its rotation's norm from unscaled products
/// there, which underflowed; the rotation was not orthogonal and `schur`
/// returned `1.0163` for `1.002`. Symmetric input has `κ = 1`, so Weyl bounds
/// every sorted eigenvalue within `η·‖Â‖_F` (`backward_error::francis`) of
/// the exact spectrum of the `f64` matrix — its own rounding from the exact
/// `Q` is below `γ_{n}·‖Â‖_F` and added.
#[test]
fn schur_is_accurate_on_clusters_at_the_gate_edge() {
    let n = 8;
    let spectrum = [0.5, 1.0, 1.0, 1.001, 1.001, 1.002, 1.002, 1.003];
    let mut rng = Xorshift64::new(0xC1C1_0008);
    for trial in 0..24 {
        // A random orthogonal Q (Gram–Schmidt, twice).
        let mut q = vec![0.0; n * n];
        for j in 0..n {
            let mut v: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_unit_f64() - 1.0).collect();
            for _ in 0..2 {
                for k in 0..j {
                    let d: f64 = (0..n).map(|i| v[i] * q[i * n + k]).sum();
                    for i in 0..n {
                        v[i] -= d * q[i * n + k];
                    }
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for i in 0..n {
                q[i * n + j] = v[i] / norm;
            }
        }
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v: f64 = (0..n)
                    .map(|k| q[i * n + k] * spectrum[k] * q[j * n + k])
                    .sum();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let frobenius = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Q's own rounding: ‖QQᵀ − I‖ and the products forming `a`.
        let construction = backward_error::gamma(3.0 * n as f64, f64::EPSILON) * 2.0 * frobenius;
        let bound = backward_error::francis(n, f64::EPSILON) * frobenius + construction;
        for exponent in [-600, -500, -490, -486, -485, -484, -480, -478, 0, 500] {
            let scaled: Vec<f64> = a.iter().map(|v| v * 2.0_f64.powi(exponent)).collect();
            let matrix = Array2::from_shape_vec([n, n], scaled).unwrap();
            for result in [
                schur(&matrix.view()).map(|decomposition| decomposition.eigenvalues()),
                leto_ops::eigenvalues(&matrix.view()),
            ] {
                let values = result.unwrap_or_else(|error| panic!("{trial} 2^{exponent}: {error}"));
                let mut real: Vec<f64> = values
                    .iter()
                    .map(|z| {
                        assert!(z.im.abs() * 2.0_f64.powi(-exponent) <= bound);
                        z.re * 2.0_f64.powi(-exponent)
                    })
                    .collect();
                real.sort_by(f64::total_cmp);
                for (value, exact) in real.iter().zip(spectrum) {
                    assert!(
                        (value - exact).abs() <= bound,
                        "trial {trial} 2^{exponent}: {value} vs {exact}, bound {bound:e}"
                    );
                }
            }
        }
    }
}

/// LAPACK `dlahqr`'s Ahues–Tisseur test keeps a subdiagonal the ulp-relative
/// pre-check alone would drop. In `[[5, 0, 0], [0, 1, 1], [0, c, d]]` with
/// `c = 10⁻¹⁷`, `d = 10⁻²⁰` (the Hessenberg reduction applies no reflector:
/// the first column's tail is zero), `|c| ≤ ulp·(1 + d)` passes the pre-check, but
/// `ba·(ab/s) = c/(1 + 1) ≈ 5·10⁻¹⁸` exceeds `ulp·(bb·(aa/s)) ≈ 10⁻³⁶`, so
/// the 2×2 block survives to `dlanv2`, whose real branch has `z = 1` exactly
/// (`p = ½`, `z = ¼` rounds from `¼ + bc`) and writes
/// `d′ = fl(d − (1/1)·c)` — the small eigenvalue `≈ −10⁻¹⁷`, negative.
/// Dropping `c` instead would return `d = +10⁻²⁰`.
#[test]
fn ahues_tisseur_keeps_a_subdiagonal_that_sets_a_small_eigenvalue() {
    let (c, d) = (1e-17, 1e-20);
    let a = mat(3, vec![5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, c, d]);
    let expected = [d - c, 1.0, 5.0];
    for spectrum in [
        schur(&a.view()).unwrap().eigenvalues(),
        leto_ops::eigenvalues(&a.view()).unwrap(),
    ] {
        let mut real: Vec<f64> = spectrum
            .iter()
            .map(|z| {
                assert_eq!(z.im, 0.0);
                z.re
            })
            .collect();
        real.sort_by(f64::total_cmp);
        assert_eq!(real, expected);
    }
}

/// The ulp-relative pre-check keeps a subdiagonal the Ahues–Tisseur test
/// alone would drop: in `[[1, 10⁻³⁰, 0], [10⁻³, ½, 0], [0, 0, 5]]`,
/// `ba·(ab/s) ≈ 2·10⁻³³` passes Ahues–Tisseur, but `10⁻³ > ulp·1.5` fails
/// the pre-check. Dropping `10⁻³` would barely move the eigenvalues but would
/// break the Schur factorization by `10⁻³`; `A = Q̂·T·Q̂ᵀ` must instead hold
/// within the derived bound: `A + E = Q̃·T·Q̃ᵀ`, `‖E‖_F ≤ η‖A‖_F`
/// (`backward_error::francis`), each row of `Q̂` within `η_Q` of `Q̃`
/// (`backward_error::francis_vectors`), plus the `f64` evaluation's
/// `γ_{2n}·‖|Q̂||T||Q̂|ᵀ‖_F`.
#[test]
fn the_ulp_precheck_keeps_a_subdiagonal_the_schur_form_needs() {
    let n = 3;
    let values = vec![1.0, 1e-30, 0.0, 1e-3, 0.5, 0.0, 0.0, 0.0, 5.0];
    let a = mat(n, values.clone());
    let s = schur(&a.view()).unwrap();
    let (q, t) = (s.q(), s.t());
    let (q, t) = (q.storage().as_slice(), t.storage().as_slice());
    let recon = reconstruct(q, t, n);
    let residual = recon
        .iter()
        .zip(&values)
        .map(|(r, v)| (r - v).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let eps = f64::EPSILON;
    let (eta, eta_q) = (
        backward_error::francis(n, eps),
        backward_error::francis_vectors(n, eps),
    );
    let root_n = (n as f64).sqrt();
    let rounding = backward_error::gamma(2.0 * n as f64, eps) * norm * n as f64;
    let bound = eta * norm
        + (2.0 * root_n * eta_q + n as f64 * eta_q * eta_q) * (1.0 + eta) * norm
        + rounding;
    assert!(residual <= bound, "‖A − QTQᵀ‖ {residual:e} > {bound:e}");
}
