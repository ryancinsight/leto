//! Real Schur `A = Q T Qᵀ`: reconstruction, orthogonality, structure, spectrum.

use leto::{Array2, Storage};
use leto_ops::{schur, MatrixDecompose};

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
    let mut mags: Vec<f64> = eigs.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect();
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
    assert!((real_eig.re - 5.0).abs() < 1e-7, "real eigenvalue must be 5");
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
        let matched = free.iter().any(|fe| {
            (se.re - fe.re).abs() < 1e-7 && (se.im - fe.im).abs() < 1e-7
        });
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
