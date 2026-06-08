use leto::{Array, ArrayView, Layout, StorageMut, VecStorage};
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

// Map NumPy PyReadonlyArray2 to Leto ArrayView
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

    Ok(ArrayView::new(layout, raw_slice))
}

#[pyfunction]
#[pyo3(name = "add")]
fn add_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;

    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;

    let shape = a.shape();
    let out_storage = VecStorage::fill(shape[0] * shape[1], 0.0f32);
    let out_layout = Layout::c_contiguous([shape[0], shape[1]])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out_arr =
        Array::new(out_layout, out_storage).map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Release GIL around computation
    py.allow_threads(|| {
        add(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    // Convert owned Array back to PyArray2
    let vec = out_arr.storage_mut().as_mut_slice().to_vec();
    let py_arr1 = PyArray1::from_vec(py, vec);
    py_arr1.reshape([shape[0], shape[1]])
}

#[pyfunction]
#[pyo3(name = "sub")]
fn sub_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;

    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;

    let shape = a.shape();
    let out_storage = VecStorage::fill(shape[0] * shape[1], 0.0f32);
    let out_layout = Layout::c_contiguous([shape[0], shape[1]])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out_arr =
        Array::new(out_layout, out_storage).map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        sub(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let vec = out_arr.storage_mut().as_mut_slice().to_vec();
    let py_arr1 = PyArray1::from_vec(py, vec);
    py_arr1.reshape([shape[0], shape[1]])
}

#[pyfunction]
#[pyo3(name = "mul")]
fn mul_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;

    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;

    let shape = a.shape();
    let out_storage = VecStorage::fill(shape[0] * shape[1], 0.0f32);
    let out_layout = Layout::c_contiguous([shape[0], shape[1]])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out_arr =
        Array::new(out_layout, out_storage).map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        mul(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let vec = out_arr.storage_mut().as_mut_slice().to_vec();
    let py_arr1 = PyArray1::from_vec(py, vec);
    py_arr1.reshape([shape[0], shape[1]])
}

#[pyfunction]
#[pyo3(name = "div")]
fn div_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;

    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;

    let shape = a.shape();
    let out_storage = VecStorage::fill(shape[0] * shape[1], 0.0f32);
    let out_layout = Layout::c_contiguous([shape[0], shape[1]])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out_arr =
        Array::new(out_layout, out_storage).map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        div(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let vec = out_arr.storage_mut().as_mut_slice().to_vec();
    let py_arr1 = PyArray1::from_vec(py, vec);
    py_arr1.reshape([shape[0], shape[1]])
}

#[pyfunction]
#[pyo3(name = "sum")]
fn sum_py(a: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    Ok(sum(&a_view))
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

    let out_storage = VecStorage::fill(shape_a[0] * shape_b[1], 0.0f32);
    let out_layout = Layout::c_contiguous([shape_a[0], shape_b[1]])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out_arr =
        Array::new(out_layout, out_storage).map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        matmul(&a_view, &b_view, &mut out_arr.view_mut())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let vec = out_arr.storage_mut().as_mut_slice().to_vec();
    let py_arr1 = PyArray1::from_vec(py, vec);
    py_arr1.reshape([shape_a[0], shape_b[1]])
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
