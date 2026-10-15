use super::trace;
use leto_ops::{CsrProduct, CsrView, ScalarStrategy, StoredOrder};
use proptest::prelude::*;

type StoredFixture = (usize, usize, Vec<Vec<(usize, i64)>>, Vec<i64>);

fn stored_fixture() -> impl Strategy<Value = StoredFixture> {
    // Shrinking removes rows/entries, reduces dimensions and moves values and
    // columns toward zero. At most eight products of magnitude 64 per output
    // keep the independent integer oracle exact without overflow.
    (1usize..5, 0usize..5, 0usize..5).prop_flat_map(|(shared, width, rows)| {
        (
            Just(shared),
            Just(width),
            prop::collection::vec(prop::collection::vec((0..shared, -8i64..=8), 0..9), rows),
            prop::collection::vec(-8i64..=8, shared * width),
        )
    })
}

proptest! {
    #[test]
    fn stored_products_match_independent_columns((shared, width, rows, rhs) in stored_fixture()) {
        let _trace = trace();
        let _case = tracing::info_span!("stored_property", shared, width, rows = rows.len()).entered();
        let (values, columns, offsets, expected, mut output) = {
            let _phase = tracing::info_span!("arrange").entered();
            let mut values = Vec::new(); let mut columns = Vec::new(); let mut offsets = vec![0i64];
            let mut expected = Vec::new();
            for row in &rows {
                for &(column, value) in row {
                    columns.push(i64::try_from(column).expect("small column"));
                    values.push(value);
                }
                offsets.push(i64::try_from(values.len()).expect("small fixture"));
                for column in 0..width {
                    expected.push(row.iter().map(|&(index, value)| value * rhs[index * width + column]).sum::<i64>());
                }
            }
            (values, columns, offsets, expected, vec![i64::MIN; rows.len() * width])
        };
        {
            let _phase = tracing::info_span!("act").entered();
            let view = CsrView::builder(&values, &columns, &offsets, rows.len(), shared)
                .with_order::<StoredOrder>().build().expect("valid generated CSR");
            let product = CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, width)
                .expect("prepared");
            // SAFETY: reverse serial order visits every in-range row exactly
            // once, completes synchronously and retains no task copy.
            unsafe { product.bind_output(&mut output).expect("bound").dispatch_with(|count, task| {
                for row in (0..count).rev() { task.run(row); }
            }); }
        }
        let _phase = tracing::info_span!("assert").entered();
        prop_assert_eq!(output, expected);
    }
}
