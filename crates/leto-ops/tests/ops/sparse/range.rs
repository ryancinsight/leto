//! Native range recovery without replacing finite four-lane rounding.
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, SliceArg, Storage};
use leto_ops::{spmv, spmv_into, spmv_view_into, CsrMatrix, RealScalar};
use std::fmt::Debug;
use tracing_subscriber::fmt::format::FmtSpan;

fn with_trace(run: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .finish();
    tracing::subscriber::with_default(subscriber, run);
}

fn recovery<T: RealScalar + RealField + FloatElement + Debug>(maximum: T) {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let large = maximum * T::from_f64(0.75);
    assert!(large.is_finite(), "fixture uses the finite floating bound");
    // Each stored-order pair cancels exactly. Stride-four lanes instead add
    // two same-sign magnitudes > MAX/2 and overflow before recombination.
    let mut values = vec![large, -large, large, -large, large, -large, large, -large];
    values.extend_from_within(..);
    values.push(one);
    let columns = (0..8).chain(0..9).collect();
    let matrix = CsrMatrix::from_parts(values, columns, vec![0, 8, 17], 2, 9)
        .expect("sorted finite rows with distinct columns");
    let input = Array1::from_shape_vec([9], vec![one; 9]).expect("matching vector");
    let storage =
        Array1::from_shape_vec([18], (0..9).flat_map(|_| [one, T::from_f64(7.0)]).collect())
            .expect("stride source");
    let strided = storage
        .slice_with::<1>(&[SliceArg::range(None, None, 2)])
        .expect("logical input is nine ones");
    drop(arrange);
    let (owned, borrowed, allocated, gathered) = {
        let _act = tracing::info_span!("act").entered();
        let mut owned = [T::from_f64(99.0); 2];
        let mut borrowed = owned;
        let mut gathered = owned;
        spmv_into(&matrix, &input.view(), &mut owned).expect("owned product");
        spmv_view_into(&matrix.as_view(), &input.view(), &mut borrowed).expect("borrowed product");
        let allocated = spmv(&matrix, &input.view()).expect("allocated product");
        spmv_into(&matrix, &strided, &mut gathered).expect("strided product");
        (owned, borrowed, allocated, gathered)
    };
    let _assert = tracing::info_span!("assert", ?owned, ?borrowed, ?gathered).entered();
    let expected = [zero, one];
    assert_eq!(owned, expected);
    assert_eq!(borrowed, expected);
    assert_eq!(allocated.storage().as_slice(), &expected);
    assert_eq!(gathered, expected);
}

fn finite_lane_results<T: RealScalar + RealField + FloatElement + Debug>(maximum: T) {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let large = maximum * T::from_f64(0.75);
    assert!(large.is_finite(), "fixture uses the finite floating bound");
    let scale = one / (<T as RealField>::EPSILON * <T as RealField>::EPSILON);
    // Row one is exact zero in the four-lane order; the old prefix fold
    // overflows on the initial pair. Row two has exact sum four, while the
    // old fold loses three units. A range repair must preserve both wins.
    let values = vec![
        large, large, -large, -large, -large, -large, large, large, scale, one, -scale, one,
        -scale, one, scale, one,
    ];
    let matrix = CsrMatrix::from_parts(values, (0..8).chain(0..8).collect(), vec![0, 8, 16], 2, 8)
        .expect("finite cancellation rows");
    let input = Array1::from_shape_vec([8], vec![one; 8]).expect("matching vector");
    drop(arrange);
    let result = {
        let _act = tracing::info_span!("act").entered();
        spmv(&matrix, &input.view()).expect("four-lane products")
    };
    let _assert = tracing::info_span!("assert", ?result).entered();
    assert_eq!(result.storage().as_slice(), &[zero, T::from_f64(4.0)]);
}

fn genuine_nonfinite<T: RealScalar + RealField + FloatElement + Debug>(maximum: T) {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let one = <T as NumericElement>::ONE;
    let large = maximum * T::from_f64(0.75);
    assert!(large.is_finite(), "fixture uses the finite floating bound");
    let matrix = CsrMatrix::from_parts(vec![large; 8], (0..8).collect(), vec![0, 8], 1, 8)
        .expect("finite positive row with unrepresentable sum");
    let invalid = CsrMatrix::from_parts(vec![T::from_f64(f64::NAN)], vec![0], vec![0, 1], 1, 8)
        .expect("shape-valid NaN input");
    let input = Array1::from_shape_vec([8], vec![one; 8]).expect("matching vector");
    drop(arrange);
    let (overflow, nan) = {
        let _act = tracing::info_span!("act").entered();
        (
            spmv(&matrix, &input.view()).expect("valid shape"),
            spmv(&invalid, &input.view()).expect("valid shape"),
        )
    };
    let _assert = tracing::info_span!("assert", ?overflow, ?nan).entered();
    assert_eq!(overflow.storage().as_slice(), &[T::from_f64(f64::INFINITY)]);
    assert!(nan.storage().as_slice()[0].is_nan());
}

#[test]
fn spmv_range_recovery_single_precision() {
    with_trace(|| recovery::<f32>(f32::MAX));
}
#[test]
fn spmv_range_recovery_double_precision() {
    with_trace(|| recovery::<f64>(f64::MAX));
}
#[test]
fn spmv_finite_lane_wins_single_precision() {
    with_trace(|| finite_lane_results::<f32>(f32::MAX));
}
#[test]
fn spmv_finite_lane_wins_double_precision() {
    with_trace(|| finite_lane_results::<f64>(f64::MAX));
}
#[test]
fn spmv_nonfinite_propagation_single_precision() {
    with_trace(|| genuine_nonfinite::<f32>(f32::MAX));
}
#[test]
fn spmv_nonfinite_propagation_double_precision() {
    with_trace(|| genuine_nonfinite::<f64>(f64::MAX));
}

// (1+eps)*(1-eps) is exactly 1-eps^2 over the reals, but its native
// product rounds to one. The accepted multiply-then-add lane therefore
// returns zero for row one; a fused/widened calculation would keep -eps^2.
fn separate_product_rounding<T: RealScalar + RealField + FloatElement + Debug>() {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let epsilon = <T as RealField>::EPSILON;
    let value = one + epsilon;
    let mut values = vec![-one, zero, zero, zero, value, zero, zero, zero];
    values.extend_from_within(..);
    values[8] = -(one + one);
    let matrix = CsrMatrix::from_parts(values, (0..8).chain(0..8).collect(), vec![0, 8, 16], 2, 8)
        .expect("sorted native rounding controls");
    let mut input = vec![one; 8];
    input[4] = one - epsilon;
    let input = Array1::from_shape_vec([8], input).expect("matching input");
    drop(arrange);
    let result = {
        let _act = tracing::info_span!("act").entered();
        spmv(&matrix, &input.view()).expect("native product")
    };
    let _assert = tracing::info_span!("assert", ?result).entered();
    assert_eq!(result.storage().as_slice(), &[zero, -one]);
}

#[test]
fn spmv_separate_product_rounding_single_precision() {
    with_trace(separate_product_rounding::<f32>);
}
#[test]
fn spmv_separate_product_rounding_double_precision() {
    with_trace(separate_product_rounding::<f64>);
}
