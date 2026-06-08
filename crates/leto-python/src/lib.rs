use leto::{Array, ArrayView, ArrayViewMut, Layout, VecStorage};
use leto_ops::{add, div, matmul, mul, sub, sum};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Require C-contiguous layout check
fn require_contiguous_2d<T: numpy::Element>(
    input: &PyReadonlyArray2<'_, T>,
    name: &str,
) -> PyResult<()> {
    if input.is_c_contiguous() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn view_from_numpy<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray2<'_, T>,
) -> PyResult<ArrayView<'a, T, 2>> {
    let shape = arr.shape();
    let strides = arr.strides();

    let el_size = std::mem::size_of::<T>() as isize;
    if el_size == 0 {
        return Err(PyValueError::new_err(
            "Zero-sized element type not supported",
        ));
    }

    let mut el_strides = [0isize; 2];
    for i in 0..2 {
        if strides[i] % el_size != 0 {
            return Err(PyValueError::new_err("Non-element-aligned NumPy stride"));
        }
        el_strides[i] = strides[i] / el_size;
    }

    let shape_arr = [shape[0], shape[1]];
    let layout = Layout::new(shape_arr, el_strides, 0);

    let raw_slice = arr.as_slice().map_err(|_| {
        PyValueError::new_err("Failed to extract contiguous slice from NumPy array")
    })?;

    ArrayView::try_new(layout, raw_slice).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn output_array(shape: [usize; 2]) -> PyResult<Array<f32, VecStorage<f32>, 2>> {
    let layout = Layout::c_contiguous(shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let size = layout
        .checked_size()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Array::new(layout, VecStorage::fill(size, 0.0f32))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn array_into_numpy2<'py>(
    py: Python<'py>,
    array: Array<f32, VecStorage<f32>, 2>,
    shape: [usize; 2],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let py_arr1 = PyArray1::from_vec(py, array.into_vec());
    py_arr1.reshape(shape)
}

fn binary_py<'py, F>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
    op: F,
) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    F: FnOnce(
            &ArrayView<'_, f32, 2>,
            &ArrayView<'_, f32, 2>,
            &mut ArrayViewMut<'_, f32, 2>,
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
fn add_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, add)
}

#[pyfunction]
#[pyo3(name = "sub")]
fn sub_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, sub)
}

#[pyfunction]
#[pyo3(name = "mul")]
fn mul_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, mul)
}

#[pyfunction]
#[pyo3(name = "div")]
fn div_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    binary_py(py, a, b, div)
}

#[pyfunction]
#[pyo3(name = "sum")]
fn sum_py(py: Python<'_>, a: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    Ok(py.allow_threads(|| sum(&a_view)))
}

#[pyfunction]
#[pyo3(name = "matmul")]
fn matmul_py<'py>(
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

/// A Python module wrapping Leto strided array operations.
#[pymodule]
fn leto_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_py, m)?)?;
    m.add_function(wrap_pyfunction!(sub_py, m)?)?;
    m.add_function(wrap_pyfunction!(mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(div_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_py, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_py, m)?)?;
    Ok(())
}
