use leto::{Array1, Array2, LetoError, Storage};
use leto_ops::{spmm_view_into, spmv_view_into, CsrProduct, CsrView, ScalarStrategy, StoredOrder};

use super::trace;

#[test]
fn absolute_row_offsets_cannot_be_reused_as_a_standalone_subview() {
    let _trace = trace();
    let _case = tracing::info_span!("row_partition").entered();
    let (values, columns, offsets, x) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [2.0, 3.0, 5.0],
            [0, 0, 1],
            [0, 1, 3],
            Array1::from_shape_vec([2], vec![7.0, 11.0]).expect("RHS shape"),
        )
    };
    let (rejection, result) = {
        let _phase = tracing::info_span!("act").entered();
        let rejection = CsrView::from_slices(&values[1..], &columns[1..], &offsets[1..], 1, 2)
            .expect_err("standalone CSR must start at offset zero");
        // Only the rebased offsets work with today's constructor. The proposal
        // partitions the full validated descriptor, avoiding this new array.
        let rebased = [0, 2];
        let row = CsrView::from_slices(&values[1..], &columns[1..], &rebased, 1, 2)
            .expect("canonical rebased row");
        let mut output = [f64::NAN];
        spmv_view_into(&row, &x.view(), &mut output).expect("row product");
        (rejection, output[0])
    };
    let _phase = tracing::info_span!("assert").entered();
    assert!(matches!(rejection, LetoError::StorageError { reason }
        if reason.contains("start at 0")));
    assert_eq!(result, 76.0); // 3*7 + 5*11.
}

#[test]
fn empty_rows_and_shared_dimensions_overwrite_every_output() {
    let _trace = trace();
    let _case = tracing::info_span!("empty_dimensions").entered();
    for rows in 0..=3 {
        for shared in 0..=3 {
            for width in 0..=3 {
                let (offsets, b, x, mut output, mut vector) = {
                    let _phase = tracing::info_span!("arrange", rows, shared, width).entered();
                    (
                        vec![0; rows + 1],
                        Array2::from_shape_vec([shared, width], vec![7.0; shared * width])
                            .expect("dense shape"),
                        Array1::from_shape_vec([shared], vec![11.0; shared]).expect("vector shape"),
                        vec![f64::NAN; rows * width],
                        vec![f64::NAN; rows],
                    )
                };
                {
                    let _phase = tracing::info_span!("act").entered();
                    let empty = CsrView::from_slices(&[], &[], &offsets, rows, shared)
                        .expect("valid empty CSR");
                    spmm_view_into(&empty, &b.view(), &mut output).expect("empty matrix product");
                    spmv_view_into(&empty, &x.view(), &mut vector).expect("empty vector product");
                    let signed_offsets = vec![0i64; rows + 1];
                    let stored = CsrView::builder(&[], &[], &signed_offsets, rows, shared)
                        .with_order::<StoredOrder>()
                        .build()
                        .expect("empty signed CSR");
                    let mut prepared_output = vec![f64::NAN; rows * width];
                    CsrProduct::<_, _, _, ScalarStrategy>::prepare(
                        stored,
                        b.storage().as_slice(),
                        width,
                    )
                    .expect("empty prepared product")
                    .execute(&mut prepared_output)
                    .expect("complete empty output");
                    assert!(prepared_output.iter().all(|value| value.to_bits() == 0));
                }
                let _phase = tracing::info_span!("assert").entered();
                assert_eq!(output.len(), rows * width);
                assert_eq!(vector.len(), rows);
                assert!(output
                    .iter()
                    .chain(&vector)
                    .all(|value| value.to_bits() == 0));
            }
        }
    }
}

#[test]
fn output_shape_errors_precede_mutation() {
    let _trace = trace();
    let _case = tracing::info_span!("output_preflight").entered();
    let (values, columns, offsets, b, x, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [3.0],
            [0],
            [0, 1, 1],
            Array2::from_shape_vec([1, 1], vec![7.0]).expect("dense shape"),
            Array1::from_shape_vec([1], vec![11.0]).expect("vector shape"),
            [f64::from_bits(0x7ff8_0000_0000_0042)],
        )
    };
    let errors = {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::from_slices(&values, &columns, &offsets, 2, 1).expect("valid CSR");
        [
            spmm_view_into(&view, &b.view(), &mut output).expect_err("short matrix output"),
            spmv_view_into(&view, &x.view(), &mut output).expect_err("short vector output"),
        ]
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(output[0].to_bits(), 0x7ff8_0000_0000_0042);
    for error in errors {
        assert!(matches!(error, LetoError::ShapeMismatch { lhs, rhs }
            if lhs == [1] && rhs == [2]));
    }
}
