//! Fixtures shared by the binding test modules.
//!
//! `pyo3::prepare_freethreaded_python` must run exactly once per process and
//! before any test touches the interpreter, which is why it lives here rather
//! than being repeated: nextest runs test binaries in parallel processes, but
//! the tests within one binary share an interpreter.

use numpy::PyArray2;
use pyo3::prelude::*;
use std::sync::Once;

static INIT_PYTHON: Once = Once::new();

pub(crate) fn prepare_python() {
    INIT_PYTHON.call_once(pyo3::prepare_freethreaded_python);
}

pub(crate) fn array2<'py>(py: Python<'py>, values: &[Vec<f32>]) -> Bound<'py, PyArray2<f32>> {
    PyArray2::from_vec2(py, values).expect("rectangular test array must construct")
}

pub(crate) fn array2_f64<'py>(py: Python<'py>, values: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_vec2(py, values).expect("rectangular test array must construct")
}

/// Assert element-wise agreement at a tolerance scaled to each element.
///
/// The bound is relative for values above unity and absolute below it, which
/// is what a fixed epsilon cannot express: these fixtures span determinants of
/// order 100 and singular values of order 0.01 in the same suite.
pub(crate) fn assert_close_slice_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() <= 1e-9 * a.abs().max(1.0),
            "actual {a} expected {e}"
        );
    }
}
