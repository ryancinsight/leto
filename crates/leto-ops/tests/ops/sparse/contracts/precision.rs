use super::{rounding::BinaryWitness, trace};
use eunomia::{Bf16, Complex, F16};
use leto_ops::{CsrProduct, CsrView, ScalarStrategy, StoredOrder};

fn adjacent_products<T: BinaryWitness>() {
    let _case =
        tracing::info_span!("adjacent_values", scalar = std::any::type_name::<T>()).entered();
    let (values, rhs, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        let half = T::ONE / (T::ONE + T::ONE);
        // Above 1, spacing is e; below 1 it is e/2. The exact product
        // (1+e)(1-e/2)=1+e/2-e²/2 lies below the midpoint, so rounds to 1.
        (
            [T::ONE + T::SPACING],
            [T::ONE - T::SPACING * half],
            [T::NAN],
        )
    };
    {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::builder(&values, &[0i64], &[0i64, 1], 1, 1)
            .with_order::<StoredOrder>()
            .build()
            .expect("signed CSR");
        CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 1)
            .expect("prepared")
            .execute(&mut output)
            .expect("output");
    }
    let _phase = tracing::info_span!("assert").entered();
    assert_eq!(output[0].bits(), T::ONE.bits());
}

fn complex_product<T: leto_ops::RealScalar>() {
    let _case =
        tracing::info_span!("complex_arithmetic", scalar = std::any::type_name::<T>()).entered();
    let (values, rhs, expected, mut output) = {
        let _phase = tracing::info_span!("arrange").entered();
        let two = T::ONE + T::ONE;
        (
            [Complex::new(T::ZERO, T::ONE), Complex::new(T::ONE, T::ONE)],
            [Complex::new(T::ONE, -T::ONE), Complex::new(T::ZERO, two)],
            Complex::new(-T::ONE, two + T::ONE),
            [Complex::new(T::ONE, T::ONE)],
        )
    };
    {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::builder(&values, &[0i64, 1], &[0i64, 2], 1, 2)
            .with_order::<StoredOrder>()
            .build()
            .expect("signed CSR");
        CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 1)
            .expect("prepared")
            .execute(&mut output)
            .expect("output");
    }
    let _phase = tracing::info_span!("assert").entered();
    // i(1-i)+(1+i)2i = -1+3i; both components are exactly representable.
    assert_eq!(output[0], expected);
}

#[test]
fn adjacent_rounding_preserves_each_real_precision() {
    let _trace = trace();
    adjacent_products::<f32>();
    adjacent_products::<f64>();
    adjacent_products::<F16>();
    adjacent_products::<Bf16>();
}

#[test]
fn complex_products_preserve_both_components_at_each_precision() {
    let _trace = trace();
    complex_product::<f32>();
    complex_product::<f64>();
    complex_product::<F16>();
    complex_product::<Bf16>();
}
