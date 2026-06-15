//! Tests for Golub–Kahan bidiagonalization (ADR 0006 SVD-prerequisite track).
//!
//! `B` is unique only up to reflector signs, so we verify the
//! convention-independent contract — `A = U B Vᵀ`, `U`/`V` orthogonal, `B` upper
//! bidiagonal — plus singular-value preservation (the property that makes
//! bidiagonalization the SVD's first phase), tied to both leto's own
//! `singular_values` and nalgebra's SVD.

use leto::{Array2, Storage};
use leto_ops::{bidiagonalize, singular_values, MatrixProduct};
use nalgebra::DMatrix;

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn assert_orthogonal(q: &Array2<f64>, n: usize) {
    let gram = q.transpose([1, 0]).unwrap().matmul(q).unwrap();
    let g = gram.storage().as_slice();
    for i in 0..n {
        for j in 0..n {
            assert_close(g[i * n + j], if i == j { 1.0 } else { 0.0 });
        }
    }
}

fn run_case(m: usize, n: usize, values: Vec<f64>) {
    let a = Array2::from_shape_vec([m, n], values.clone()).unwrap();
    let bd = bidiagonalize(&a.view()).unwrap();
    let (u, b, v) = (bd.u(), bd.b(), bd.v());

    assert_eq!(u.shape(), [m, m]);
    assert_eq!(b.shape(), [m, n]);
    assert_eq!(v.shape(), [n, n]);

    // U, V orthogonal.
    assert_orthogonal(u, m);
    assert_orthogonal(v, n);

    // B upper bidiagonal: nonzero only on the diagonal and first superdiagonal.
    let bs = b.storage().as_slice();
    for i in 0..m {
        for j in 0..n {
            if j < i || j > i + 1 {
                assert_close(bs[i * n + j], 0.0);
            }
        }
    }

    // Reconstruction A = U B Vᵀ.
    let recon = u
        .matmul(b)
        .unwrap()
        .matmul(&v.transpose([1, 0]).unwrap())
        .unwrap();
    for (actual, expected) in recon.storage().as_slice().iter().zip(values.iter()) {
        assert_close(*actual, *expected);
    }

    // Singular-value preservation: σ(B) == σ(A) (leto), and == nalgebra's σ(A).
    let sv_b = singular_values(&b.view()).unwrap();
    let sv_a = singular_values(&a.view()).unwrap();
    assert_eq!(sv_b.len(), sv_a.len());
    for (lb, la) in sv_b.iter().zip(sv_a.iter()) {
        assert_close(*lb, *la);
    }
    let na_sv = DMatrix::from_row_slice(m, n, &values)
        .svd(false, false)
        .singular_values
        .as_slice()
        .to_vec();
    let mut sv_b_sorted = sv_b.clone();
    let mut na_sorted = na_sv;
    sv_b_sorted.sort_by(|x, y| y.total_cmp(x));
    na_sorted.sort_by(|x, y| y.total_cmp(x));
    for (lb, na) in sv_b_sorted.iter().zip(na_sorted.iter()) {
        assert_close(*lb, *na);
    }
}

#[test]
fn bidiagonalize_tall() {
    run_case(
        4,
        3,
        vec![
            4.0, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
        ],
    );
}

#[test]
fn bidiagonalize_square() {
    run_case(3, 3, vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0]);
}

#[test]
fn bidiagonalize_rectangular_full_rank() {
    run_case(5, 2, vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 3.0, 1.0]);
}

#[test]
fn bidiagonalize_rejects_wide() {
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(bidiagonalize(&a.view()).is_err());
}
