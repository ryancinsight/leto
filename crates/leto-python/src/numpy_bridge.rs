//! Private NumPy ↔ Leto array conversion helpers.
//!
//! These are shared by all Python-binding modules and are `pub(crate)` so
//! they can be imported by sibling modules without exposing them to Python.

use leto::{Array, Array1, ArrayView, Layout, VecStorage};
use leto_ops::RealScalar;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn require_contiguous_1d<T: numpy::Element>(
    input: &PyReadonlyArray1<'_, T>,
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

pub(crate) fn require_contiguous_2d<T: numpy::Element>(
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

pub(crate) fn view_from_numpy_1d<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray1<'_, T>,
) -> PyResult<ArrayView<'a, T, 1>> {
    let shape = arr.shape();
    let strides = arr.strides();

    let el_size = std::mem::size_of::<T>() as isize;
    if el_size == 0 {
        return Err(PyValueError::new_err(
            "Zero-sized element type not supported",
        ));
    }

    if strides[0] % el_size != 0 {
        return Err(PyValueError::new_err("Non-element-aligned NumPy stride"));
    }
    let el_stride = strides[0] / el_size;

    let shape_arr = [shape[0]];
    let layout = Layout::new(shape_arr, [el_stride], 0);

    let raw_slice = arr.as_slice().map_err(|_| {
        PyValueError::new_err("Failed to extract contiguous slice from NumPy array")
    })?;

    ArrayView::try_new(layout, raw_slice).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn view_from_numpy<'a, T: numpy::Element>(
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

pub(crate) fn output_array<T: RealScalar + numpy::Element>(
    shape: [usize; 2],
) -> PyResult<Array<T, VecStorage<T>, 2>> {
    let layout = Layout::c_contiguous(shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let size = layout
        .checked_size()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Array::new(layout, VecStorage::fill(size, T::ZERO))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn array_into_numpy1<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    array: Array<T, VecStorage<T>, 1>,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    Ok(PyArray1::from_vec(py, array.into_vec()))
}

pub(crate) fn array_into_numpy2<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    array: Array<T, VecStorage<T>, 2>,
    shape: [usize; 2],
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let py_arr1 = PyArray1::from_vec(py, array.into_vec());
    py_arr1.reshape(shape)
}

pub(crate) fn require_contiguous_3d<T: numpy::Element>(
    input: &PyReadonlyArray3<'_, T>,
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

pub(crate) fn view_from_numpy_3d<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray3<'_, T>,
) -> PyResult<ArrayView<'a, T, 3>> {
    let shape = arr.shape();
    let strides = arr.strides();

    let el_size = std::mem::size_of::<T>() as isize;
    if el_size == 0 {
        return Err(PyValueError::new_err(
            "Zero-sized element type not supported",
        ));
    }

    let mut el_strides = [0isize; 3];
    for i in 0..3 {
        if strides[i] % el_size != 0 {
            return Err(PyValueError::new_err("Non-element-aligned NumPy stride"));
        }
        el_strides[i] = strides[i] / el_size;
    }

    let shape_arr = [shape[0], shape[1], shape[2]];
    let layout = Layout::new(shape_arr, el_strides, 0);

    let raw_slice = arr.as_slice().map_err(|_| {
        PyValueError::new_err("Failed to extract contiguous slice from NumPy array")
    })?;

    ArrayView::try_new(layout, raw_slice).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn output_array_3d<T: RealScalar + numpy::Element>(
    shape: [usize; 3],
) -> PyResult<Array<T, VecStorage<T>, 3>> {
    let layout = Layout::c_contiguous(shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let size = layout
        .checked_size()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Array::new(layout, VecStorage::fill(size, T::ZERO))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn array_into_numpy3<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    array: Array<T, VecStorage<T>, 3>,
    shape: [usize; 3],
) -> PyResult<Bound<'py, PyArray3<T>>> {
    let py_arr1 = PyArray1::from_vec(py, array.into_vec());
    py_arr1.reshape(shape)
}
