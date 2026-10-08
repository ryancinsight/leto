use eunomia::Complex64;
use leto::Array1;
use leto_ops::{spgemm, spmv, CooMatrix};

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
