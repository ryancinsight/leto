//! Python bindings for elementwise and basic array operations.

use crate::numpy_bridge::{
    array_into_numpy1, array_into_numpy2, array_into_numpy3, output_array, output_array_3d,
    require_contiguous_1d, require_contiguous_2d, require_contiguous_3d, view_from_numpy,
    view_from_numpy_1d, view_from_numpy_3d,
};
use leto::{ArrayD, LayoutDyn, SliceStorage};
use leto_ops::{add, batched_matmul, div, dot, matmul, mul, sub, sum};
use numpy::{
    PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn binary_py<'py, F>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
    op: F,
) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    F: FnOnce(
            &leto::ArrayView<'_, f32, 2>,
            &leto::ArrayView<'_, f32, 2>,
            &mut leto::ArrayViewMut<'_, f32, 2>,
        ) -> leto::Result<()>
        + Send,
{
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;

    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;
    let shape = [a.shape()[0], a.shape()[1]];
    let mut out_arr = output_array(shape)?;

    py.allow_threads(|| {
        op(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    array_into_numpy2(py, out_arr, shape)
}

#[pyfunction]
#[pyo3(name = "add")]
pub(crate) fn add_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, add)
}

#[pyfunction]
#[pyo3(name = "sub")]
pub(crate) fn sub_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, sub)
}

#[pyfunction]
#[pyo3(name = "mul")]
pub(crate) fn mul_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, mul)
}

#[pyfunction]
#[pyo3(name = "div")]
pub(crate) fn div_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, div)
}

#[pyfunction]
#[pyo3(name = "dot")]
pub(crate) fn dot_py(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f32>,
    b: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    require_contiguous_1d(&a, "a")?;
    require_contiguous_1d(&b, "b")?;
    let a_view = view_from_numpy_1d(&a)?;
    let b_view = view_from_numpy_1d(&b)?;
    py.allow_threads(|| dot(&a_view, &b_view).map_err(|e| PyValueError::new_err(e.to_string())))
}

#[pyfunction]
#[pyo3(name = "sum")]
pub(crate) fn sum_py(py: Python<'_>, a: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    Ok(py.allow_threads(|| sum(&a_view)))
}

#[pyfunction]
#[pyo3(name = "matmul")]
pub(crate) fn matmul_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;

    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;

    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a[1] != shape_b[0] {
        return Err(PyValueError::new_err(
            "Dimension mismatch for matrix multiplication",
        ));
    }

    let out_shape = [shape_a[0], shape_b[1]];
    let mut out_arr = output_array(out_shape)?;

    py.allow_threads(|| {
        matmul(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    array_into_numpy2(py, out_arr, out_shape)
}

/// Batched matrix multiply over 3D arrays `[batch, m, k] @ [batch, k, n] -> [batch, m, n]`.
/// A batch dimension of 1 on either operand broadcasts. Mirrors `numpy.matmul` on 3D inputs.
#[pyfunction]
#[pyo3(name = "batched_matmul")]
pub(crate) fn batched_matmul_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'_, f32>,
    b: PyReadonlyArray3<'_, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    require_contiguous_3d(&a, "a")?;
    require_contiguous_3d(&b, "b")?;
    let a_view = view_from_numpy_3d(&a)?;
    let b_view = view_from_numpy_3d(&b)?;

    let sa = a.shape();
    let sb = b.shape();
    if sa[2] != sb[1] {
        return Err(PyValueError::new_err(
            "Inner dimension mismatch for batched matmul",
        ));
    }
    let batch = sa[0].max(sb[0]);
    let out_shape = [batch, sa[1], sb[2]];
    let mut out_arr = output_array_3d(out_shape)?;

    py.allow_threads(|| {
        batched_matmul(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    array_into_numpy3(py, out_arr, out_shape)
}

const MAX_DYNAMIC_RANK: usize = 6;

fn sum_dynamic(shape: &[usize], slice: &[f32]) -> PyResult<f32> {
    let to_py = |e: leto::LetoError| PyValueError::new_err(e.to_string());
    let layout = LayoutDyn::c_contiguous(shape).map_err(to_py)?;
    let arr = ArrayD::new(layout, SliceStorage::new(slice)).map_err(to_py)?;
    let total = match arr.ndim() {
        1 => sum(&arr.into_dimensionality::<1>().map_err(to_py)?.view()),
        2 => sum(&arr.into_dimensionality::<2>().map_err(to_py)?.view()),
        3 => sum(&arr.into_dimensionality::<3>().map_err(to_py)?.view()),
        4 => sum(&arr.into_dimensionality::<4>().map_err(to_py)?.view()),
        5 => sum(&arr.into_dimensionality::<5>().map_err(to_py)?.view()),
        6 => sum(&arr.into_dimensionality::<6>().map_err(to_py)?.view()),
        n => {
            return Err(PyValueError::new_err(format!(
                "array rank {n} exceeds the supported dynamic-dispatch rank {MAX_DYNAMIC_RANK}"
            )))
        }
    };
    Ok(total)
}

#[pyfunction]
#[pyo3(name = "sum_dyn")]
pub(crate) fn sum_dyn_py(py: Python<'_>, a: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err("a must be C-contiguous"));
    }
    let shape = a.shape().to_vec();
    let slice = a.as_slice().map_err(|_| {
        PyValueError::new_err("Failed to extract contiguous slice from NumPy array")
    })?;
    py.allow_threads(|| sum_dynamic(&shape, slice))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_py, m)?)?;
    m.add_function(wrap_pyfunction!(sub_py, m)?)?;
    m.add_function(wrap_pyfunction!(mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(div_py, m)?)?;
    m.add_function(wrap_pyfunction!(dot_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_py, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(batched_matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_dyn_py, m)?)?;
    Ok(())
}
