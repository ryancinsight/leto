use leto::{Array1, LetoError};
use leto_ops::{spmv_view_into, CsrView};

use super::trace;

// Exactly 2^53. `powi` has unspecified precision and is unsuitable for a
// bitwise oracle; the literal rounds exactly in binary64.
const MAGNITUDE: f64 = 9_007_199_254_740_992.0;

#[test]
fn canonicalization_changes_a_stored_order_product() {
    let _trace = trace();
    let _case = tracing::info_span!("canonicalization", scalar = "f64").entered();
    let (values, columns, offsets, x) = {
        let _phase = tracing::info_span!("arrange").entered();
        // At 2^53 the spacing is 2. Ties-to-even rounds 2^53 + 1 to 2^53.
        (
            [MAGNITUDE, 1.0, -MAGNITUDE],
            [1, 0, 1],
            [0, 3],
            Array1::from_shape_vec([2], vec![1.0, 1.0]).expect("RHS shape"),
        )
    };
    let (rejection, stored_sum, canonical_sum) = {
        let _phase = tracing::info_span!("act").entered();
        let rejection = CsrView::from_slices(&values, &columns, &offsets, 1, 2)
            .expect_err("the current view requires strictly increasing columns");
        // x is identically one.
        let stored_sum = values.iter().fold(0.0, |sum, value| sum + value);
        // Coalescing column 1 gives 0; the remaining column 0 contributes 1.
        let canonical_values = [1.0, 0.0];
        let canonical_columns = [0, 1];
        let canonical_offsets = [0, 2];
        let canonical = CsrView::from_slices(
            &canonical_values,
            &canonical_columns,
            &canonical_offsets,
            1,
            2,
        )
        .expect("canonical CSR");
        let mut output = [f64::NAN];
        spmv_view_into(&canonical, &x.view(), &mut output).expect("canonical product");
        (rejection, stored_sum, output[0])
    };
    let _phase = tracing::info_span!("assert").entered();
    assert!(matches!(rejection, LetoError::StorageError { reason }
        if reason.contains("strictly increasing")));
    assert_eq!(stored_sum.to_bits(), 0);
    assert_eq!(canonical_sum.to_bits(), 1.0_f64.to_bits());
}

#[test]
fn sorting_without_coalescing_also_changes_rounding() {
    let _trace = trace();
    let _case = tracing::info_span!("sorting", scalar = "f64").entered();
    let (values, columns, offsets) = {
        let _phase = tracing::info_span!("arrange").entered();
        ([MAGNITUDE, 1.0, -MAGNITUDE], [1, 1, 0], [0, 3])
    };
    let (rejection, stored, sorted) = {
        let _phase = tracing::info_span!("act").entered();
        let rejection = CsrView::from_slices(&values, &columns, &offsets, 1, 2)
            .expect_err("the current view rejects stored-order duplicate columns");
        let stored = values.iter().fold(0.0, |sum, value| sum + value);
        // Stable column sorting moves -2^53 first, without merging duplicates.
        let sorted = [values[2], values[0], values[1]]
            .iter()
            .fold(0.0_f64, |sum, value| sum + value);
        (rejection, stored, sorted)
    };
    let _phase = tracing::info_span!("assert").entered();
    assert!(matches!(rejection, LetoError::StorageError { reason }
        if reason.contains("strictly increasing")));
    assert_eq!(stored.to_bits(), 0);
    assert_eq!(sorted.to_bits(), 1.0_f64.to_bits());
}
