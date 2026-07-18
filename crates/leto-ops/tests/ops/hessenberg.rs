//! Tests for the Householder Hessenberg reduction (ADR 0006).
//!
//! Hessenberg `H` is unique only up to reflector signs, so we verify the
//! convention-independent *contract* — `A = Q H Qᵀ`, `Q` orthogonal, `H` upper
//! Hessenberg — plus orthogonal-similarity invariants (trace, Frobenius norm).

use leto::{Array2, Storage};
use leto_ops::{hessenberg, norm_l2, trace, MatrixProduct};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn reconstruct(q: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
    let qh = q.matmul(h).unwrap();
    let qt = q.transpose([1, 0]).unwrap();
    qh.matmul(&qt).unwrap()
}

fn assert_orthogonal(q: &Array2<f64>, n: usize) {
    let qt = q.transpose([1, 0]).unwrap();
    let gram = qt.matmul(q).unwrap();
    let g = gram.storage().as_slice();
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_close(g[i * n + j], expected);
        }
    }
}

#[test]
fn hessenberg_reconstructs_and_is_upper_hessenberg() {
    let n = 4;
    let values = vec![
        4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0,
    ];
    // Make it genuinely non-symmetric.
    let values = {
        let mut v = values;
        v[1] = 5.0; // a[0][1] ≠ a[1][0]
        v
    };
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let decomp = hessenberg(&a.view()).unwrap();
    let h = decomp.h();
    let q = decomp.q();

    // Q orthogonal.
    assert_orthogonal(q, n);

    // H upper Hessenberg: zero below the first subdiagonal.
    let hs = h.storage().as_slice();
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 {
                assert_close(hs[i * n + j], 0.0);
            }
        }
    }

    // Reconstruction A = Q H Qᵀ.
    let reconstructed = reconstruct(q, h);
    assert_eq!(reconstructed.storage().as_slice().len(), values.len());
    for (actual, expected) in reconstructed.storage().as_slice().iter().zip(values.iter()) {
        assert_close(*actual, *expected);
    }

    // Orthogonal-similarity invariants (oracle-independent).
    assert_close(trace(&h.view()).unwrap(), trace(&a.view()).unwrap());
    assert_close(norm_l2(&h.view()).unwrap(), norm_l2(&a.view()).unwrap());
}

#[test]
fn hessenberg_of_symmetric_is_tridiagonal() {
    let n = 4;
    let values = vec![
        4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0,
    ];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let decomp = hessenberg(&a.view()).unwrap();
    let hs = decomp.h().storage().as_slice();

    // Symmetric input → Hessenberg form is tridiagonal (also zero above the
    // first superdiagonal).
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 || j > i + 1 {
                assert_close(hs[i * n + j], 0.0);
            }
        }
    }
    let reconstructed = reconstruct(decomp.q(), decomp.h());
    for (actual, expected) in reconstructed.storage().as_slice().iter().zip(values.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn hessenberg_rejects_non_square() {
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(hessenberg(&a.view()).is_err());
}
