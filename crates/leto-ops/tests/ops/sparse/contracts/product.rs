use eunomia::{Bf16, Complex, NumericElement, F16};
use leto::LetoError;
use leto_ops::{CsrProduct, CsrView, ScalarStrategy, StoredOrder};

use super::trace;

fn closed_form<T: NumericElement>() {
    let _case = tracing::info_span!("scalar_family", scalar = std::any::type_name::<T>()).entered();
    let (values, columns, offsets, rhs, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [T::ONE, T::ONE, T::ONE],
            [1i64, 0, 1],
            [0i64, 3, 3],
            [T::ONE, T::ZERO, T::ONE, T::ONE],
            [T::ONE; 4],
        )
    };
    {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::builder(&values, &columns, &offsets, 2, 2)
            .with_order::<StoredOrder>()
            .build()
            .expect("stored CSR");
        CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 2)
            .expect("prepared shape")
            .execute(&mut output)
            .expect("complete output");
    }
    let _phase = tracing::info_span!("assert").entered();
    let two = T::ONE + T::ONE;
    assert_eq!(output, [two + T::ONE, two, T::ZERO, T::ZERO]);
}

#[test]
fn minimal_shared_scalar_bound_covers_all_coeus_families() {
    let _trace = trace();
    closed_form::<u8>();
    closed_form::<u16>();
    closed_form::<u32>();
    closed_form::<u64>();
    closed_form::<i8>();
    closed_form::<i16>();
    closed_form::<i32>();
    closed_form::<i64>();
    closed_form::<f32>();
    closed_form::<f64>();
    closed_form::<F16>();
    closed_form::<Bf16>();
    closed_form::<Complex<f32>>();
    closed_form::<Complex<f64>>();
    closed_form::<Complex<F16>>();
    closed_form::<Complex<Bf16>>();
}

#[test]
fn scalar_product_retains_stored_order_and_separate_rounding() {
    let _trace = trace();
    let (values, rhs, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            [-1.0, 1.0 + f64::EPSILON],
            [1.0; 17]
                .into_iter()
                .chain([1.0 - f64::EPSILON; 17])
                .collect::<Vec<_>>(),
            [f64::NAN; 34],
        )
    };
    {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::builder(&values, &[0i64, 1], &[0i64, 2, 2], 2, 2)
            .with_order::<StoredOrder>()
            .build()
            .expect("stored CSR");
        CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 17)
            .expect("prepared")
            .execute(&mut output)
            .expect("output");
    }
    {
        let _phase = tracing::info_span!("assert").entered();
        assert!(output.iter().all(|value| value.to_bits() == 0));
    }
    let magnitude = 9_007_199_254_740_992.0;
    let values = [magnitude, 1.0, -magnitude];
    let view = CsrView::builder(&values, &[1i64, 0, 1], &[0i64, 3], 1, 2)
        .with_order::<StoredOrder>()
        .build()
        .expect("stored order");
    let mut output = [f64::NAN];
    CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &[1.0, 1.0], 1)
        .expect("prepared")
        .execute(&mut output)
        .expect("output");
    assert_eq!(output[0].to_bits(), 0);
}

#[test]
fn preparation_and_binding_errors_leave_outputs_untouched() {
    let _trace = trace();
    let (view, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        (
            CsrView::from_slices(&[3.0], &[0], &[0, 1, 1], 2, 1).expect("view"),
            [f64::from_bits(0x7ff8_0000_0000_0042)],
        )
    };
    let (shape_error, binding_error, overflow) = {
        let _phase = tracing::info_span!("act").entered();
        let shape_error =
            CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &[], 1).expect_err("short RHS");
        let product =
            CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &[7.0], 1).expect("prepared");
        let binding_error = product.execute(&mut output).expect_err("short output");
        let overflow = CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &[], usize::MAX)
            .expect_err("output count overflow");
        (shape_error, binding_error, overflow)
    };
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(output[0].to_bits(), 0x7ff8_0000_0000_0042);
    assert!(matches!(shape_error, LetoError::ShapeMismatch { .. }));
    assert!(matches!(binding_error, LetoError::ShapeMismatch { .. }));
    assert!(matches!(overflow, LetoError::StorageError { .. }));
}
