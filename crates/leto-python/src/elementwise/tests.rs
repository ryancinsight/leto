//! Value-semantic tests for the element-wise binding surface.

use crate::elementwise::{add_py, matmul_py, sum_dyn_py, sum_py};
use crate::support::{array2, prepare_python};
use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::ffi::c_str;
use pyo3::prelude::*;

#[test]
fn add_returns_numpy_array_with_value_semantics() {
    prepare_python();

    Python::with_gil(|py| {
        let lhs = array2(py, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let rhs = array2(py, &[vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]]);

        let output = add_py(py, lhs.readonly(), rhs.readonly()).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(
            output.readonly().as_slice().unwrap(),
            &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        );
    });
}

#[test]
fn matmul_returns_numpy_array_with_value_semantics() {
    prepare_python();

    Python::with_gil(|py| {
        let lhs = array2(py, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let rhs = array2(py, &[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);

        let output = matmul_py(py, lhs.readonly(), rhs.readonly()).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(
            output.readonly().as_slice().unwrap(),
            &[58.0, 64.0, 139.0, 154.0]
        );
    });
}

#[test]
fn sum_releases_boundary_and_returns_scalar_value() {
    prepare_python();

    Python::with_gil(|py| {
        let input = array2(py, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let total = sum_py(py, input.readonly()).unwrap();

        assert_eq!(total, 21.0);
    });
}

#[test]
fn matmul_rejects_shape_mismatch() {
    prepare_python();

    Python::with_gil(|py| {
        let lhs = array2(py, &[vec![1.0, 2.0, 3.0]]);
        let rhs = array2(py, &[vec![4.0, 5.0, 6.0]]);

        let result = matmul_py(py, lhs.readonly(), rhs.readonly());

        assert!(result.is_err());
    });
}

#[test]
fn sum_dyn_handles_multiple_ranks() {
    prepare_python();

    Python::with_gil(|py| {
        // Rank 1: arange(5) → 0+1+2+3+4 = 10.
        let r1 = PyArray1::from_vec(py, vec![0.0f32, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sum_dyn_py(py, r1.to_dyn().readonly()).unwrap(), 10.0);

        // Rank 2: [[1,2,3],[4,5,6]] → 21.
        let r2 = array2(py, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(sum_dyn_py(py, r2.to_dyn().readonly()).unwrap(), 21.0);

        // Rank 3: ones((2,2,2)) → 8.
        let r3 = py
            .eval(
                c_str!("__import__('numpy').ones((2, 2, 2), dtype='float32')"),
                None,
                None,
            )
            .unwrap()
            .extract::<PyReadonlyArrayDyn<'_, f32>>()
            .unwrap();
        assert_eq!(sum_dyn_py(py, r3).unwrap(), 8.0);
    });
}

#[test]
fn sum_dyn_rejects_non_contiguous() {
    prepare_python();

    Python::with_gil(|py| {
        let view = py
            .eval(
                c_str!("__import__('numpy').arange(6, dtype='float32').reshape(2, 3).T"),
                None,
                None,
            )
            .unwrap()
            .extract::<PyReadonlyArrayDyn<'_, f32>>()
            .unwrap();
        assert!(sum_dyn_py(py, view).is_err());
    });
}

#[test]
fn operations_reject_non_contiguous_numpy_inputs() {
    prepare_python();

    Python::with_gil(|py| {
        let view = py
            .eval(
                c_str!("__import__('numpy').arange(6, dtype='float32').reshape(2, 3).T"),
                None,
                None,
            )
            .unwrap()
            .extract::<PyReadonlyArray2<'_, f32>>()
            .unwrap();

        let result = sum_py(py, view);

        assert!(result.is_err());
    });
}
