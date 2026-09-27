#![allow(clippy::unwrap_used, reason = "test scope")]

use super::factor::factor_numeric;
use crate::application::sparse::lu_symbolic::factor_symbolic;
use crate::application::sparse::{CooMatrix, CscMatrix};
use leto::{Array1, LetoError};

/// Build an f64 CSC square matrix from triples.
fn make_csc(n: usize, triplets: &[(usize, usize, f64)]) -> CscMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for &(r, c, v) in triplets {
        coo.push(r, c, v);
    }
    coo.to_csc()
}

/// `‖A x - b‖∞` for a dense `A` reconstructed from the CSC test matrix.
/// Uses `CscMatrix::to_dense` (existing API) since `CscMatrix::get` is
/// not part of the public surface.
fn residual_inf(a: &CscMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let dense = a.to_dense();
    let [nrows, ncols] = dense.shape();
    let mut r = 0.0_f64;
    for (i, &bi) in b.iter().enumerate().take(nrows) {
        let mut sum = 0.0;
        for (j, &xj) in x.iter().enumerate().take(ncols) {
            let v = dense.get([i, j]).copied().unwrap_or(0.0);
            sum += v * xj;
        }
        let dx = (sum - bi).abs();
        if dx > r {
            r = dx;
        }
    }
    r
}

#[test]
fn factor_poisson_1d_laplacian_n16_roundtrip() {
    // Tridiagonal Poisson-Laplacian: A[i,i] = 2, A[i,i-1] = A[i,i+1] = -1
    // (rows 0 and n-1 are Dirichlet-only: only A[0,0] and A[n-1,n-1]).
    let n = 16usize;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0));
        if i > 0 {
            triplets.push((i, i - 1, -1.0));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
        }
    }
    let b: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let csc = make_csc(n, &triplets);
    let symbolic = factor_symbolic(&csc);
    let lu = factor_numeric(&csc, &symbolic, 1e-12).expect("factor");
    let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
    let x = lu.solve(&b_arr.view()).expect("solve");
    let residual = residual_inf(&csc, x.as_slice().unwrap(), &b);
    assert!(residual < 1e-10, "residual = {residual}");
}

#[test]
fn factor_banded_5_diagonal_n32() {
    let n = 32usize;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    // 5-diagonal: main = 6, ±1 = -1, ±2 = -1
    for i in 0..n {
        triplets.push((i, i, 6.0));
        if i >= 1 {
            triplets.push((i, i - 1, -1.0));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
        }
        if i >= 2 {
            triplets.push((i, i - 2, -1.0));
        }
        if i + 2 < n {
            triplets.push((i, i + 2, -1.0));
        }
    }
    // x_known = [1.0, 0.5, 0.25, ...];
    let x_known: Vec<f64> = (0..n).map(|i| 1.0_f64 / (i as f64 + 1.0)).collect();
    // b = A * x_known (closed form via dense reconstruction)
    let mut b = vec![0.0_f64; n];
    for &(r, c, v) in &triplets {
        b[r] += v * x_known[c];
    }
    let csc = make_csc(n, &triplets);
    let symbolic = factor_symbolic(&csc);
    let lu = factor_numeric(&csc, &symbolic, 1e-12).expect("factor");
    let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
    let x = lu.solve(&b_arr.view()).expect("solve");
    let residual = residual_inf(&csc, x.as_slice().unwrap(), &b);
    assert!(residual < 1e-10, "residual = {residual}");
}

#[test]
fn factor_random_sparse_n64_diff_dense() {
    // Differential test against the existing dense LU oracle.
    // Use a known fixed seed for determinism.
    use crate::application::linalg::lu::lu_decompose;
    use crate::application::sparse::csr_to_dense;
    let n = 64usize;
    // Pre-generated pattern: 5 bands with diagonals 0,1,7,15,31. Deterministic.
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 1.7_f64 + (i as f64) * 0.01));
        let offsets = [1usize, 7, 15, 31];
        for &o in &offsets {
            if i + o < n {
                triplets.push((i, i + o, -0.3_f64 + (i as f64) * 0.001));
                triplets.push((i + o, i, 0.4_f64 - (i as f64) * 0.001));
            }
        }
    }
    let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();

    // Oracle via dense LU
    let csr = {
        let mut coo = CooMatrix::new(n, n);
        for &(r, c, v) in &triplets {
            coo.push(r, c, v);
        }
        coo.to_csr()
    };
    let dense = csr_to_dense(&csr);
    let lu_oracle = lu_decompose(&dense.view()).expect("dense oracle");
    let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
    let x_dense = lu_oracle.solve(&b_arr.view()).expect("dense solve");

    // Sparse LU
    let csc = {
        let mut coo = CooMatrix::new(n, n);
        for &(r, c, v) in &triplets {
            coo.push(r, c, v);
        }
        coo.to_csc()
    };
    let symbolic = factor_symbolic(&csc);
    let lu = factor_numeric(&csc, &symbolic, 1e-12).expect("sparse factor");
    let x_sparse = lu.solve(&b_arr.view()).expect("sparse solve");

    let mut max_diff = 0.0_f64;
    for i in 0..n {
        let d = (x_sparse[i] - x_dense[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(max_diff < 1e-8, "max_diff = {max_diff}");
}

#[test]
fn singular_matrix_yields_storage_error() {
    // Row 0 is zero → singular at step 0.
    let n = 3usize;
    let triplets: &[(usize, usize, f64)] = &[(1, 1, 1.0), (1, 2, 2.0), (2, 1, 3.0), (2, 2, 4.0)];
    let b = vec![1.0_f64, 2.0, 3.0];
    let csc = make_csc(n, triplets);
    let symbolic = factor_symbolic(&csc);
    let err = factor_numeric(&csc, &symbolic, 1e-12).expect_err("should be singular");
    match &err {
        LetoError::StorageError { reason } => {
            assert!(reason.contains("singular"), "unexpected: {reason}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    // b unused on the singular path; suppress dead-code lint.
    let _ = b;
}

#[test]
fn factor_f32_generic() {
    let n = 4usize;
    let mut triplets: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 4.0_f32));
        if i + 1 < n {
            triplets.push((i, i + 1, 1.0_f32));
            triplets.push((i + 1, i, 1.0_f32));
        }
    }
    let b: Vec<f32> = (1..=n).map(|k| k as f32).collect();
    let mut coo = CooMatrix::new(n, n);
    for &(r, c, v) in &triplets {
        coo.push(r, c, v);
    }
    let csc = coo.to_csc();
    let symbolic = factor_symbolic(&csc);
    let lu = factor_numeric(&csc, &symbolic, 1e-6).expect("f32 factor");
    let b_arr = Array1::from_shape_vec([n], b).expect("b shape");
    let x = lu.solve(&b_arr.view()).expect("solve");
    // Sanity: the f32 instantiation compiles and runs; one component
    // assertion confirms the solve returns a finite result.
    assert!(x[0].is_finite(), "f32 solve produced finite x[0]");
}
