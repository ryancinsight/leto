use leto_ops::{CsrProduct, CsrView, ScalarStrategy, StoredOrder};

use super::trace;

#[test]
fn scoped_workers_write_disjoint_rows_in_any_order() {
    let _trace = trace();
    let (values, columns, offsets, rhs, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [2i64, 3, 5],
            [0i64, 0, 1],
            [0i64, 1, 1, 3],
            [7i64, 11, 13, 17],
            [-1i64; 6],
        )
    };
    let result = {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::builder(&values, &columns, &offsets, 3, 2)
            .with_order::<StoredOrder>()
            .build()
            .expect("stored CSR");
        let product =
            CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 2).expect("prepared");
        // SAFETY: scoped threads join before return/unwind, run each row once,
        // and have no access to the bound output or borrowed inputs otherwise.
        unsafe {
            product
                .bind_output(&mut output)
                .expect("bound")
                .dispatch_with(|rows, task| {
                    std::thread::scope(|scope| {
                        for row in (0..rows).rev() {
                            scope.spawn(move || task.run(row));
                        }
                    });
                    rows
                })
        }
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(result, 3);
    assert_eq!(output, [14, 22, 0, 0, 86, 118]);
}

#[test]
fn partial_dispatch_failure_and_unwind_join_before_borrow_returns() {
    let _trace = trace();
    let (view, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            CsrView::from_slices(&[3i64, 5], &[0, 0], &[0, 1, 2], 2, 1).expect("CSR"),
            [-1i64; 2],
        )
    };
    let (result, panic) = {
        let _phase = tracing::info_span!("act").entered();
        let product =
            CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &[7], 1).expect("prepared");
        // SAFETY: only row 1 runs; error allows a subset, and no work escapes.
        let result = unsafe {
            product
                .bind_output(&mut output)
                .expect("bound")
                .dispatch_with(|_, task| {
                    task.run(1);
                    Err::<(), _>("admission stopped")
                })
        };
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the scoped worker runs row 0 once and is joined even when
            // the dispatching thread panics. No task survives the scope.
            unsafe {
                product
                    .bind_output(&mut output)
                    .expect("bound")
                    .dispatch_with(|_, task| {
                        std::thread::scope(|scope| {
                            scope.spawn(move || task.run(0));
                            panic!("dispatch unwind witness");
                        });
                    });
            }
        }));
        (result, panic)
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(result, Err("admission stopped"));
    let payload = panic.expect_err("dispatcher must propagate its panic after joining");
    let message = payload
        .downcast_ref::<&str>()
        .expect("the dispatch witness carries a string literal");
    assert_eq!(*message, "dispatch unwind witness");
    assert_eq!(output, [21, 35]);
    output.fill(11); // The exclusive borrow is usable again after both scopes.
    assert_eq!(output, [11, 11]);
}
