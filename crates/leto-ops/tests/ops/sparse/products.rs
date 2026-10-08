//! Sparse products, complex values and rejection paths.

use super::*;

#[test]
fn spmv_matches_closed_form_and_overwrites_output() {
    let a = CsrMatrix::from_parts(
        vec![2.0f64, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .unwrap();
    let x_base = Array::from_shape_vec([4], vec![9.0f64, 1.0, 2.0, 3.0]).unwrap();
    let x = x_base
        .slice_with::<1>(&[SliceArg::range(Some(3), Some(0), -1)])
        .unwrap();
    let mut y = vec![99.0; 3];

    spmv_into(&a, &x, &mut y).unwrap();
    let allocated = spmv(&a, &x).unwrap();

    assert_eq!(y, &[5.0, 6.0, 4.0]);
    assert_eq!(allocated.storage().as_slice(), &[5.0, 6.0, 4.0]);
}

#[test]
fn spmm_matches_closed_form_with_strided_dense_rhs() {
    let a =
        CsrMatrix::from_parts(vec![2.0f64, -1.0, 3.0], vec![0, 2, 1], vec![0, 2, 3], 2, 3).unwrap();
    let b_base = Array2::from_shape_vec([3, 2], vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = b_base
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let mut c = vec![99.0; 4];

    spmm_into(&a, &b, &mut c).unwrap();
    let allocated = spmm(&a, &b).unwrap();

    assert_eq!(c, &[-2.0, -3.0, 12.0, 9.0]);
    assert_eq!(allocated.storage().as_slice(), &[-2.0, -3.0, 12.0, 9.0]);
}

#[test]
fn spgemm_matches_closed_form_and_sorted_csr_rows() {
    let a =
        CsrMatrix::from_parts(vec![2.0f64, -1.0, 3.0], vec![0, 2, 1], vec![0, 2, 3], 2, 3).unwrap();
    let b = CsrMatrix::from_parts(
        vec![4.0f64, 5.0, 6.0, 7.0],
        vec![1, 0, 0, 1],
        vec![0, 1, 2, 4],
        3,
        2,
    )
    .unwrap();

    let c = spgemm(&a, &b).unwrap();

    assert_eq!(c.shape(), (2, 2));
    assert_eq!(c.row_ptr(), &[0, 2, 3]);
    assert_eq!(c.col_indices(), &[0, 1, 0]);
    assert_eq!(c.values(), &[-6.0, 1.0, 15.0]);
    assert_eq!(c.to_dense().storage().as_slice(), &[-6.0, 1.0, 15.0, 0.0]);
}

#[test]
fn spgemm_drops_exact_zero_cancellation() {
    let a = CsrMatrix::from_parts(vec![1.0f64, 1.0], vec![0, 1], vec![0, 2], 1, 2).unwrap();
    let b =
        CsrMatrix::from_parts(vec![3.0f64, -3.0, 2.0], vec![0, 0, 1], vec![0, 1, 3], 2, 2).unwrap();

    let c = spgemm(&a, &b).unwrap();

    assert_eq!(c.shape(), (1, 2));
    assert_eq!(c.row_ptr(), &[0, 1]);
    assert_eq!(c.col_indices(), &[1]);
    assert_eq!(c.values(), &[2.0]);
}

#[test]
fn sparse_products_reject_shape_mismatch() {
    let a = CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![0, 1], 1, 1).unwrap();
    let bad_x = Array::from_shape_vec([2], vec![1.0f64, 2.0]).unwrap();
    let bad_b = Array2::from_shape_vec([2, 1], vec![1.0f64, 2.0]).unwrap();
    let good_b = Array2::from_shape_vec([1, 1], vec![1.0f64]).unwrap();

    assert!(spmv(&a, &bad_x.view()).is_err());
    assert!(spmm(&a, &bad_b.view()).is_err());
    assert!(spmm_into(&a, &good_b.view(), &mut [0.0; 2]).is_err());
    assert!(spgemm(&a, &CsrMatrix::zeros(2, 1)).is_err());
}

#[test]
fn csr_from_parts_rejects_invalid_structure() {
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![0], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![], vec![0, 1], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![1], vec![0, 1], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![1, 1], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64, 2.0], vec![0, 0], vec![0, 2], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64, 2.0], vec![1, 0], vec![0, 2], 1, 2).is_err());
}

#[test]
fn empty_width_dense_matrix_has_empty_csr_storage() {
    let dense = Array2::<f64>::from_shape_vec([2, 0], vec![]).unwrap();

    let csr = CsrMatrix::from_dense(&dense.view());

    assert_eq!(csr.shape(), (2, 0));
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.density(), 0.0);
    assert_eq!(csr.as_parts().2, &[0, 0, 0]);
    assert_eq!(csr.to_dense().storage().as_slice(), &[] as &[f64]);
}

// ── Complex scalars ──────────────────────────────────────────────────────────
//
// Frequency-domain consumers (boundary-element Helmholtz operators) assemble
// complex-valued sparse operators. These exercise the canonical containers at
// `Complex64` rather than a complex-only parallel type; the oracle is the
// closed-form complex arithmetic, computed independently of the kernel.

#[test]
fn coo_accumulates_duplicate_complex_entries() {
    // Element assembly adds overlapping contributions into the same slot:
    // (1 + 2i) + (0.5 - i) = (1.5 + i).
    let mut coo = CooMatrix::<Complex64>::new(2, 2);
    coo.push(0, 0, Complex64::new(1.0, 2.0));
    coo.push(0, 0, Complex64::new(0.5, -1.0));
    coo.push(1, 1, Complex64::new(3.0, -4.0));

    let csr = coo.to_csr();

    assert_eq!(csr.shape(), (2, 2));
    let a00 = csr.get(0, 0).expect("assembled diagonal entry");
    assert_complex_close(a00, Complex64::new(1.5, 1.0), "accumulated (0,0)");
    let a11 = csr.get(1, 1).expect("assembled diagonal entry");
    assert_complex_close(a11, Complex64::new(3.0, -4.0), "single-contribution (1,1)");
    assert!(csr.get(0, 1).is_none(), "structurally absent entry");
}

#[test]
fn complex_spmv_matches_closed_form() {
    // A = [[1+i, 2], [0, -3i]],  x = [1-i, 2i]
    // y0 = (1+i)(1-i) + 2(2i) = 2 + 4i
    // y1 = (-3i)(2i)          = 6
    let mut coo = CooMatrix::<Complex64>::new(2, 2);
    coo.push(0, 0, Complex64::new(1.0, 1.0));
    coo.push(0, 1, Complex64::new(2.0, 0.0));
    coo.push(1, 1, Complex64::new(0.0, -3.0));
    let a = coo.to_csr();

    let x = Array1::from_shape_vec(
        [2],
        vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 2.0)],
    )
    .expect("rhs shape");

    let y = spmv(&a, &x.view()).expect("complex spmv");

    assert_complex_close(y[0], Complex64::new(2.0, 4.0), "y0");
    assert_complex_close(y[1], Complex64::new(6.0, 0.0), "y1");
}

#[test]
fn complex_spgemm_matches_closed_form() {
    // A = [[i, 0], [0, 1]],  B = [[2, 0], [0, i]]
    // A·B = [[2i, 0], [0, i]]
    let mut a_coo = CooMatrix::<Complex64>::new(2, 2);
    a_coo.push(0, 0, Complex64::new(0.0, 1.0));
    a_coo.push(1, 1, Complex64::new(1.0, 0.0));
    let mut b_coo = CooMatrix::<Complex64>::new(2, 2);
    b_coo.push(0, 0, Complex64::new(2.0, 0.0));
    b_coo.push(1, 1, Complex64::new(0.0, 1.0));

    let c = spgemm(&a_coo.to_csr(), &b_coo.to_csr()).expect("complex spgemm");

    assert_complex_close(
        c.get(0, 0).expect("product entry"),
        Complex64::new(0.0, 2.0),
        "c00",
    );
    assert_complex_close(
        c.get(1, 1).expect("product entry"),
        Complex64::new(0.0, 1.0),
        "c11",
    );
}

#[test]
fn complex_values_mut_rescales_in_place_without_touching_pattern() {
    // Assembled operators are rescaled in place (frequency sweeps reuse one
    // pattern); the sparsity structure must survive untouched.
    let mut coo = CooMatrix::<Complex64>::new(2, 2);
    coo.push(0, 0, Complex64::new(1.0, 0.0));
    coo.push(1, 0, Complex64::new(0.0, 1.0));
    let mut csr = coo.to_csr();
    let pattern_before = (csr.row_ptr().to_vec(), csr.col_indices().to_vec());

    // Multiply every entry by i: 1 -> i, i -> -1.
    for value in csr.values_mut() {
        *value *= Complex64::new(0.0, 1.0);
    }

    assert_eq!(
        (csr.row_ptr().to_vec(), csr.col_indices().to_vec()),
        pattern_before,
        "in-place value update must not disturb the pattern"
    );
    assert_complex_close(
        csr.get(0, 0).expect("entry"),
        Complex64::new(0.0, 1.0),
        "rescaled (0,0)",
    );
    assert_complex_close(
        csr.get(1, 0).expect("entry"),
        Complex64::new(-1.0, 0.0),
        "rescaled (1,0)",
    );
}

fn assert_complex_close(actual: Complex64, expected: Complex64, context: &str) {
    const EPS: f64 = 1e-12;
    assert!(
        (actual.re - expected.re).abs() <= EPS && (actual.im - expected.im).abs() <= EPS,
        "{context}: actual {}{:+}i, expected {}{:+}i",
        actual.re,
        actual.im,
        expected.re,
        expected.im
    );
}
