//! CPU traversal delegates the scalar contract without changing layouts.
use super::*;
use leto_ops::{unary_map_into, DigammaOp};

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn digamma_transpose_and_output_layout_match_analytic_values() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let _act = tracing::info_span!("act").entered();
    let gamma = 0.577_215_664_901_532_9;
    let half = -gamma - 2.0 * core::f64::consts::LN_2;
    let input = Array::from_shape_vec([2, 3], vec![0.5, 1.0, 2.0, -0.5, -1.5, 3.0]).unwrap();
    let view = input.transpose([1, 0]).unwrap();
    let result = unary_map(DigammaOp, &view).unwrap();
    let expected = [
        half,
        half + 2.0,
        -gamma,
        half + 2.0 + 2.0 / 3.0,
        1.0 - gamma,
        1.5 - gamma,
    ];
    assert_eq!(result.shape(), [3, 2]);
    let _assert = tracing::info_span!("assert").entered();
    for (actual, expected) in result.storage().as_slice().iter().zip(expected) {
        assert!((actual - expected).abs() <= 128.0 * f64::EPSILON / 2.0 * (1.0 + expected.abs()));
    }
    let mut output = Array::from_shape_vec([2, 3], vec![99.0; 6]).unwrap();
    let mut output_transposed = output.transpose_mut([1, 0]).unwrap();
    unary_map_into(DigammaOp, &view, &mut output_transposed).unwrap();
    for (actual, x) in output
        .storage()
        .as_slice()
        .iter()
        .zip(input.storage().as_slice())
    {
        assert_eq!(*actual, eunomia::FloatElement::digamma(*x));
    }
}

#[test]
fn digamma_cpu_map_preserves_domain_results_and_empty_shape() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let _act = tracing::info_span!("act").entered();
    let input = Array::from_shape_vec(
        [8],
        vec![
            0.0_f64,
            -0.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e-300,
            1e300,
        ],
    )
    .unwrap();
    let output = unary_map(DigammaOp, &input.view()).unwrap();
    let values = output.storage().as_slice();
    assert_eq!(values[0], f64::NEG_INFINITY);
    assert_eq!(values[1], f64::INFINITY);
    assert!(values[2].is_nan());
    assert_eq!(values[3], f64::INFINITY);
    assert!(values[4].is_nan());
    assert!(values[5].is_nan());
    assert!((values[6] / -1e300 - 1.0).abs() <= 128.0 * f64::EPSILON);
    assert!((values[7] - 1e300_f64.ln()).abs() <= f64::EPSILON * values[7].abs());
    let empty = Array::from_shape_vec([0, 3], Vec::<f64>::new()).unwrap();
    let result = unary_map(DigammaOp, &empty.view()).unwrap();
    assert_eq!(result.shape(), [0, 3]);
    assert!(result.storage().as_slice().is_empty());
}
