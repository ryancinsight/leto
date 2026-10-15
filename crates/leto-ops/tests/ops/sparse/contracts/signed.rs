use leto::LetoError;
use leto_ops::{CsrView, StoredOrder};

use super::trace;

#[test]
fn signed_stored_views_alias_all_original_parts() {
    let _trace = trace();
    let (values, columns, offsets) = {
        let _phase = tracing::info_span!("arrange").entered();
        ([3.0, 5.0, 7.0], [1i64, 0, 1], [0i64, 3, 3])
    };
    let (view, canonical_error) = {
        let _phase = tracing::info_span!("act").entered();
        let canonical_error = CsrView::builder(&values, &columns, &offsets, 2, 2)
            .build()
            .expect_err("duplicate and decreasing columns");
        let view = CsrView::builder(&values, &columns, &offsets, 2, 2)
            .with_order::<StoredOrder>()
            .build()
            .expect("stored CSR");
        (view, canonical_error)
    };
    let _phase = tracing::info_span!("assert").entered();
    let (v, c, r) = view.as_parts();
    assert_eq!(v.as_ptr(), values.as_ptr());
    assert_eq!(c.as_ptr(), columns.as_ptr());
    assert_eq!(r.as_ptr(), offsets.as_ptr());
    assert_eq!(view.shape(), (2, 2));
    assert_eq!(view.nnz(), 3);
    assert!(matches!(canonical_error, LetoError::StorageError { reason }
        if reason.contains("strictly increasing")));
}

#[test]
fn every_signed_index_is_checked_even_for_empty_products() {
    let _trace = trace();
    let _phase = tracing::info_span!("arrange").entered();
    // Width-independent rejection: the large column is either not representable
    // or is representable but outside k=1. The offset is likewise never valid.
    let cases = [
        (vec![-1], vec![0, 1], 1, 1),
        (vec![i64::MAX], vec![0, 1], 1, 1),
        (vec![0], vec![0, -1], 1, 1),
        (vec![0], vec![0, i64::MAX], 1, 1),
        (vec![0], vec![0, 2], 1, 1),
        (vec![0], vec![1, 1], 1, 1),
        (vec![0], vec![0, 1, 0, 1], 3, 1),
        (vec![0], vec![0, 1], 1, 0),
        (vec![], vec![-1], 0, 0),
        (vec![], vec![0], usize::MAX, 0),
    ];
    drop(_phase);
    for (columns, offsets, rows, shared) in cases {
        let values = vec![3.0; columns.len()];
        let result = {
            let _phase = tracing::info_span!("act", rows, shared).entered();
            CsrView::builder(&values, &columns, &offsets, rows, shared)
                .with_order::<StoredOrder>()
                .build()
        };
        let _phase = tracing::info_span!("assert").entered();
        assert!(matches!(result, Err(LetoError::StorageError { .. })));
    }
    // A representable large column is legal when k includes it; narrowing
    // targets must reject the identical original i64 without reinterpretation.
    let large = i64::from(u32::MAX) + 1;
    let values = [1.0];
    let columns = [large];
    let offsets = [0i64, 1];
    let result = CsrView::builder(&values, &columns, &offsets, 1, usize::MAX).build();
    assert_eq!(result.is_ok(), usize::try_from(large).is_ok());
}
