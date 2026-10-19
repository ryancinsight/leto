use leto_ops::{CsrProduct, CsrView, ScalarStrategy, StoredOrder};
use tracing_subscriber::fmt::format::FmtSpan;

#[test]
fn signed_stored_product_borrows_and_executes_without_allocation() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    );
    let (values, columns, offsets, rhs, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [3i64, 5, 7],
            [1i64, 0, 1],
            [0i64, 3, 3],
            [11i64, 13],
            [-1i64; 2],
        )
    };
    let (result, allocations, reallocations) = {
        let _phase = tracing::info_span!("act").entered();
        super::measured(|| {
            let view = CsrView::builder(&values, &columns, &offsets, 2, 2)
                .with_order::<StoredOrder>()
                .build()?;
            CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 1)?.execute(&mut output)
        })
    };
    let _phase = tracing::info_span!("assert").entered();
    result.expect("valid complete product");
    assert_eq!(output, [185, 0]); // 3*13 + 5*11 + 7*13 = 185.
    assert_eq!((allocations, reallocations), (0, 0));
}
