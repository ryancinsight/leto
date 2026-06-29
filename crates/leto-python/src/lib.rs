#![allow(clippy::type_complexity)]

use leto::{
    Array, Array1, ArrayD, ArrayView, ArrayViewMut, Layout, LayoutDyn, SliceStorage, VecStorage,
};
use leto_ops::{
    add, bidiagonalize, bunch_kaufman, cholesky_decompose, col_piv_qr, det, div, dot, eigenvalues,
    full_piv_lu, hessenberg, inv, kron, matexp, matmul, mul, norm_l1, norm_l2, norm_max,
    qr_decompose, schur, singular_values, solve, sub, sum, svd_decompose, symmetric_eigen_jacobi,
    trace, udu_decompose, RealScalar,
};
use numpy::{
    Complex64, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn require_contiguous_1d<T: numpy::Element>(
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

fn view_from_numpy_1d<'a, T: numpy::Element>(
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

fn output_array<T: RealScalar + numpy::Element>(
    shape: [usize; 2],
) -> PyResult<Array<T, VecStorage<T>, 2>> {
    let layout = Layout::c_contiguous(shape).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let size = layout
        .checked_size()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Array::new(layout, VecStorage::fill(size, T::ZERO))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn array_into_numpy1<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    array: Array<T, VecStorage<T>, 1>,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    Ok(PyArray1::from_vec(py, array.into_vec()))
}

fn array_into_numpy2<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    array: Array<T, VecStorage<T>, 2>,
    shape: [usize; 2],
) -> PyResult<Bound<'py, PyArray2<T>>> {
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
#[pyo3(name = "dot")]
fn dot_py(
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
fn sum_dyn_py(py: Python<'_>, a: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f32> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err("a must be C-contiguous"));
    }
    let shape = a.shape().to_vec();
    let slice = a.as_slice().map_err(|_| {
        PyValueError::new_err("Failed to extract contiguous slice from NumPy array")
    })?;
    py.allow_threads(|| sum_dynamic(&shape, slice))
}

#[pyfunction]
#[pyo3(name = "det")]
fn det_py(py: Python<'_>, a: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    py.allow_threads(|| det(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))
}

#[pyfunction]
#[pyo3(name = "inv")]
fn inv_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let out_arr =
        py.allow_threads(|| inv(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let shape = [out_arr.shape()[0], out_arr.shape()[1]];
    array_into_numpy2(py, out_arr, shape)
}

#[pyfunction]
#[pyo3(name = "solve")]
fn solve_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_1d(&b, "b")?;
    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy_1d(&b)?;
    let out_arr = py.allow_threads(|| {
        solve(&a_view, &b_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    array_into_numpy1(py, out_arr)
}

#[pyfunction]
#[pyo3(name = "cholesky")]
fn cholesky_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        cholesky_decompose(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let lower = decomp.lower().clone();
    let shape = [lower.shape()[0], lower.shape()[1]];
    array_into_numpy2(py, lower, shape)
}

#[pyfunction]
#[pyo3(name = "qr")]
fn qr_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        qr_decompose(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let q = decomp.q();
    let r = decomp.r();
    let q_shape = [q.shape()[0], q.shape()[1]];
    let r_shape = [r.shape()[0], r.shape()[1]];
    let py_q = array_into_numpy2(py, q, q_shape)?;
    let py_r = array_into_numpy2(py, r, r_shape)?;
    Ok((py_q, py_r))
}

#[pyfunction]
#[pyo3(name = "col_piv_qr")]
fn col_piv_qr_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp =
        py.allow_threads(|| col_piv_qr(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let q = decomp.q();
    let r = decomp.r();
    let perm = decomp
        .permutation()
        .iter()
        .map(|&x| x as u64)
        .collect::<Vec<_>>();
    let q_shape = [q.shape()[0], q.shape()[1]];
    let r_shape = [r.shape()[0], r.shape()[1]];
    let py_q = array_into_numpy2(py, q, q_shape)?;
    let py_r = array_into_numpy2(py, r, r_shape)?;
    let py_perm = PyArray1::from_vec(py, perm);
    Ok((py_q, py_r, py_perm))
}

#[pyfunction]
#[pyo3(name = "svd")]
fn svd_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        svd_decompose(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let u = decomp.left_singular_vectors;
    let s = Array1::from_shape_vec([decomp.singular_values.len()], decomp.singular_values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let vt_transposed = decomp
        .right_singular_vectors
        .transpose([1, 0])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vt = vt_transposed.to_contiguous();

    let u_shape = [u.shape()[0], u.shape()[1]];
    let vt_shape = [vt.shape()[0], vt.shape()[1]];

    let py_u = array_into_numpy2(py, u, u_shape)?;
    let py_s = array_into_numpy1(py, s)?;
    let py_vt = array_into_numpy2(py, vt, vt_shape)?;

    Ok((py_u, py_s, py_vt))
}

#[pyfunction]
#[pyo3(name = "symmetric_eigen")]
fn symmetric_eigen_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        symmetric_eigen_jacobi(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let w = Array1::from_shape_vec([decomp.eigenvalues.len()], decomp.eigenvalues)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v = decomp.eigenvectors;
    let v_shape = [v.shape()[0], v.shape()[1]];

    let py_w = array_into_numpy1(py, w)?;
    let py_v = array_into_numpy2(py, v, v_shape)?;

    Ok((py_w, py_v))
}

#[pyfunction]
#[pyo3(name = "singular_values")]
fn singular_values_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let s_vals = py.allow_threads(|| {
        singular_values(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let s = Array1::from_shape_vec([s_vals.len()], s_vals)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    array_into_numpy1(py, s)
}

#[pyfunction]
#[pyo3(name = "norm", signature = (a, ord=None))]
fn norm_py(py: Python<'_>, a: PyReadonlyArray2<'_, f64>, ord: Option<&str>) -> PyResult<f64> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    py.allow_threads(|| {
        let ord_str = ord.unwrap_or("fro");
        match ord_str {
            "1" | "l1" => norm_l1(&a_view).map_err(|e| PyValueError::new_err(e.to_string())),
            "2" | "fro" | "l2" => {
                norm_l2(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
            }
            "max" | "inf" => norm_max(&a_view).map_err(|e| PyValueError::new_err(e.to_string())),
            o => Err(PyValueError::new_err(format!(
                "Unsupported norm order: {o}"
            ))),
        }
    })
}

#[pyfunction]
#[pyo3(name = "schur")]
fn schur_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp =
        py.allow_threads(|| schur(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let q = decomp.q();
    let t = decomp.t();
    let q_shape = [q.shape()[0], q.shape()[1]];
    let t_shape = [t.shape()[0], t.shape()[1]];
    let py_q = array_into_numpy2(py, q, q_shape)?;
    let py_t = array_into_numpy2(py, t, t_shape)?;
    Ok((py_q, py_t))
}

#[pyfunction]
#[pyo3(name = "bunch_kaufman")]
fn bunch_kaufman_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        bunch_kaufman(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let l = decomp.l();
    let d = decomp.d();
    let perm = decomp
        .permutation()
        .iter()
        .map(|&x| x as u64)
        .collect::<Vec<_>>();
    let l_shape = [l.shape()[0], l.shape()[1]];
    let d_shape = [d.shape()[0], d.shape()[1]];
    let py_l = array_into_numpy2(py, l, l_shape)?;
    let py_d = array_into_numpy2(py, d, d_shape)?;
    let py_perm = PyArray1::from_vec(py, perm);
    Ok((py_l, py_d, py_perm))
}

#[pyfunction]
#[pyo3(name = "matexp")]
fn matexp_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let out_arr =
        py.allow_threads(|| matexp(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let shape = [out_arr.shape()[0], out_arr.shape()[1]];
    array_into_numpy2(py, out_arr, shape)
}

#[pyfunction]
#[pyo3(name = "kron")]
fn kron_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_2d(&b, "b")?;
    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy(&b)?;
    let out_arr = py.allow_threads(|| {
        kron(&a_view, &b_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let shape = [out_arr.shape()[0], out_arr.shape()[1]];
    array_into_numpy2(py, out_arr, shape)
}

#[pyfunction]
#[pyo3(name = "trace")]
fn trace_py(py: Python<'_>, a: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    py.allow_threads(|| trace(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))
}

/// Hessenberg reduction `A = Q H Qᵀ` with `H` upper-Hessenberg, `Q` orthogonal.
/// Returns `(Q, H)` (mirrors `scipy.linalg.hessenberg(a, calc_q=True)` ordered `(H, Q)`).
#[pyfunction]
#[pyo3(name = "hessenberg")]
fn hessenberg_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py
        .allow_threads(|| hessenberg(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let q = decomp.q().clone();
    let h = decomp.h().clone();
    let q_shape = [q.shape()[0], q.shape()[1]];
    let h_shape = [h.shape()[0], h.shape()[1]];
    let py_q = array_into_numpy2(py, q, q_shape)?;
    let py_h = array_into_numpy2(py, h, h_shape)?;
    Ok((py_q, py_h))
}

/// Eigenvalues of a general real matrix (Francis double-shift QR), returned as a
/// complex vector — mirrors `numpy.linalg.eigvals`. Order is not specified.
#[pyfunction]
#[pyo3(name = "eigenvalues")]
fn eigenvalues_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let vals = py
        .allow_threads(|| eigenvalues(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let out: Vec<Complex64> = vals.into_iter().map(|c| Complex64::new(c.re, c.im)).collect();
    Ok(PyArray1::from_vec(py, out))
}

/// Full-pivoting LU: `A[row_perm][:, col_perm] = L U` with `L` unit-lower, `U` upper.
/// Returns `(L, U, row_perm, col_perm)` (row/column permutations as index vectors).
#[pyfunction]
#[pyo3(name = "full_piv_lu")]
fn full_piv_lu_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py
        .allow_threads(|| full_piv_lu(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let l = decomp.l();
    let u = decomp.u();
    let row: Vec<u64> = decomp.row_permutation().iter().map(|&x| x as u64).collect();
    let col: Vec<u64> = decomp.col_permutation().iter().map(|&x| x as u64).collect();
    let l_shape = [l.shape()[0], l.shape()[1]];
    let u_shape = [u.shape()[0], u.shape()[1]];
    let py_l = array_into_numpy2(py, l, l_shape)?;
    let py_u = array_into_numpy2(py, u, u_shape)?;
    Ok((
        py_l,
        py_u,
        PyArray1::from_vec(py, row),
        PyArray1::from_vec(py, col),
    ))
}

/// Pivot-free `A = U D Uᵀ` factorization of a symmetric matrix.
/// Returns `(U, d)` with `U` upper-triangular and `d` the diagonal of `D`.
#[pyfunction]
#[pyo3(name = "udu")]
fn udu_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py
        .allow_threads(|| udu_decompose(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let u = decomp.u();
    let u_shape = [u.shape()[0], u.shape()[1]];
    let d = decomp.diagonal().to_vec();
    let py_u = array_into_numpy2(py, u, u_shape)?;
    Ok((py_u, PyArray1::from_vec(py, d)))
}

/// Bidiagonalization `A = U B Vᵀ` with `B` upper-bidiagonal, `U`/`V` orthogonal.
/// Returns `(U, B, V)`.
#[pyfunction]
#[pyo3(name = "bidiagonalize")]
fn bidiagonalize_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        bidiagonalize(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let u = decomp.u().clone();
    let b = decomp.b().clone();
    let v = decomp.v().clone();
    let u_shape = [u.shape()[0], u.shape()[1]];
    let b_shape = [b.shape()[0], b.shape()[1]];
    let v_shape = [v.shape()[0], v.shape()[1]];
    let py_u = array_into_numpy2(py, u, u_shape)?;
    let py_b = array_into_numpy2(py, b, b_shape)?;
    let py_v = array_into_numpy2(py, v, v_shape)?;
    Ok((py_u, py_b, py_v))
}

#[pymodule]
fn leto_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_py, m)?)?;
    m.add_function(wrap_pyfunction!(sub_py, m)?)?;
    m.add_function(wrap_pyfunction!(mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(div_py, m)?)?;
    m.add_function(wrap_pyfunction!(dot_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_py, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_dyn_py, m)?)?;

    m.add_function(wrap_pyfunction!(det_py, m)?)?;
    m.add_function(wrap_pyfunction!(inv_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_py, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky_py, m)?)?;
    m.add_function(wrap_pyfunction!(qr_py, m)?)?;
    m.add_function(wrap_pyfunction!(col_piv_qr_py, m)?)?;
    m.add_function(wrap_pyfunction!(svd_py, m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_eigen_py, m)?)?;
    m.add_function(wrap_pyfunction!(singular_values_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_py, m)?)?;
    m.add_function(wrap_pyfunction!(schur_py, m)?)?;
    m.add_function(wrap_pyfunction!(bunch_kaufman_py, m)?)?;
    m.add_function(wrap_pyfunction!(matexp_py, m)?)?;
    m.add_function(wrap_pyfunction!(kron_py, m)?)?;
    m.add_function(wrap_pyfunction!(trace_py, m)?)?;
    m.add_function(wrap_pyfunction!(hessenberg_py, m)?)?;
    m.add_function(wrap_pyfunction!(eigenvalues_py, m)?)?;
    m.add_function(wrap_pyfunction!(full_piv_lu_py, m)?)?;
    m.add_function(wrap_pyfunction!(udu_py, m)?)?;
    m.add_function(wrap_pyfunction!(bidiagonalize_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Storage;
    use pyo3::ffi::c_str;
    use std::sync::Once;

    static INIT_PYTHON: Once = Once::new();

    fn prepare_python() {
        INIT_PYTHON.call_once(pyo3::prepare_freethreaded_python);
    }

    fn array2<'py>(py: Python<'py>, values: &[Vec<f32>]) -> Bound<'py, PyArray2<f32>> {
        PyArray2::from_vec2(py, values).expect("rectangular test array must construct")
    }

    fn array2_f64<'py>(py: Python<'py>, values: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_vec2(py, values).expect("rectangular test array must construct")
    }

    fn assert_close_slice_f64(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() <= 1e-9 * a.abs().max(1.0),
                "actual {a} expected {e}"
            );
        }
    }

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

    #[test]
    fn det_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![3.0, 8.0], vec![4.0, 6.0]]);
            let det_val = det_py(py, a.readonly()).unwrap();

            let np_det: f64 = py
                .eval(
                    c_str!("__import__('numpy').linalg.det([[3.0, 8.0], [4.0, 6.0]])"),
                    None,
                    None,
                )
                .unwrap()
                .extract()
                .unwrap();

            assert!((det_val - np_det).abs() < 1e-9);
        });
    }

    #[test]
    fn inv_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
            let inv_val = inv_py(py, a.readonly()).unwrap();

            let np_inv = py
                .eval(
                    c_str!("__import__('numpy').linalg.inv([[1.0, 2.0], [3.0, 4.0]])"),
                    None,
                    None,
                )
                .unwrap()
                .extract::<PyReadonlyArray2<'_, f64>>()
                .unwrap();

            let inv_slice = inv_val.readonly().as_slice().unwrap().to_vec();
            let np_slice = np_inv.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&inv_slice, &np_slice);
        });
    }

    #[test]
    fn solve_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![3.0, 1.0], vec![1.0, 2.0]]);
            let b = PyArray1::from_vec(py, vec![9.0, 8.0]);
            let x_val = solve_py(py, a.readonly(), b.readonly()).unwrap();

            let np_x = py
                .eval(
                    c_str!(
                        "__import__('numpy').linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])"
                    ),
                    None,
                    None,
                )
                .unwrap()
                .extract::<PyReadonlyArray1<'_, f64>>()
                .unwrap();

            let x_slice = x_val.readonly().as_slice().unwrap().to_vec();
            let np_slice = np_x.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&x_slice, &np_slice);
        });
    }

    #[test]
    fn cholesky_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![4.0, 12.0], vec![12.0, 37.0]]);
            let l_val = cholesky_py(py, a.readonly()).unwrap();

            let np_l = py
                .eval(
                    c_str!("__import__('numpy').linalg.cholesky([[4.0, 12.0], [12.0, 37.0]])"),
                    None,
                    None,
                )
                .unwrap()
                .extract::<PyReadonlyArray2<'_, f64>>()
                .unwrap();

            let l_slice = l_val.readonly().as_slice().unwrap().to_vec();
            let np_slice = np_l.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&l_slice, &np_slice);
        });
    }

    #[test]
    fn qr_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
            let (q_val, r_val) = qr_py(py, a.readonly()).unwrap();

            let q_readonly = q_val.readonly();
            let r_readonly = r_val.readonly();
            let q_mat = view_from_numpy(&q_readonly).unwrap();
            let r_mat = view_from_numpy(&r_readonly).unwrap();
            let mut reconstruct = output_array::<f64>([2, 2]).unwrap();
            matmul(&q_mat, &r_mat, &mut reconstruct.view_mut()).unwrap();
            assert_close_slice_f64(reconstruct.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn col_piv_qr_matches_numpy_reconstruction() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![2.0, 3.0], vec![5.0, 7.0]]);
            let (q_val, r_val, perm_val) = col_piv_qr_py(py, a.readonly()).unwrap();

            let q_readonly = q_val.readonly();
            let r_readonly = r_val.readonly();
            let q_mat = view_from_numpy(&q_readonly).unwrap();
            let r_mat = view_from_numpy(&r_readonly).unwrap();
            let mut q_r = output_array::<f64>([2, 2]).unwrap();
            matmul(&q_mat, &r_mat, &mut q_r.view_mut()).unwrap();

            let perm_slice = perm_val.readonly().as_slice().unwrap().to_vec();

            let a_readonly = a.readonly();
            let a_view = view_from_numpy(&a_readonly).unwrap();
            for (col, &orig_col_idx) in perm_slice.iter().enumerate().take(2) {
                let orig_col = orig_col_idx as usize;
                for row in 0..2 {
                    let expected = *a_view.get([row, orig_col]).unwrap();
                    let actual = *q_r.get([row, col]).unwrap();
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
        });
    }

    #[test]
    fn svd_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
            let (u_val, s_val, vt_val) = svd_py(py, a.readonly()).unwrap();

            let np_svd = py
                .eval(
                    c_str!("__import__('numpy').linalg.svd([[1.0, 2.0], [3.0, 4.0]])"),
                    None,
                    None,
                )
                .unwrap();
            let np_s = np_svd
                .get_item(1)
                .unwrap()
                .extract::<PyReadonlyArray1<'_, f64>>()
                .unwrap();

            let s_slice = s_val.readonly().as_slice().unwrap().to_vec();
            let np_s_slice = np_s.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&s_slice, &np_s_slice);

            // Verify reconstruction: A_reconstructed = U * S * Vt
            let u_readonly = u_val.readonly();
            let vt_readonly = vt_val.readonly();
            let u_mat = view_from_numpy(&u_readonly).unwrap();
            let vt_mat = view_from_numpy(&vt_readonly).unwrap();

            let mut s_mat = output_array::<f64>([2, 2]).unwrap();
            *s_mat.get_mut([0, 0]).unwrap() = s_slice[0];
            *s_mat.get_mut([1, 1]).unwrap() = s_slice[1];

            let mut us = output_array::<f64>([2, 2]).unwrap();
            matmul(&u_mat, &s_mat.view(), &mut us.view_mut()).unwrap();

            let mut a_reconstructed = output_array::<f64>([2, 2]).unwrap();
            matmul(&us.view(), &vt_mat, &mut a_reconstructed.view_mut()).unwrap();

            assert_close_slice_f64(a_reconstructed.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn singular_values_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
            let s_val = singular_values_py(py, a.readonly()).unwrap();

            let np_s = py
                .eval(
                    c_str!(
                        "__import__('numpy').linalg.svd([[1.0, 2.0], [3.0, 4.0]], compute_uv=False)"
                    ),
                    None,
                    None,
                )
                .unwrap()
                .extract::<PyReadonlyArray1<'_, f64>>()
                .unwrap();

            let s_slice = s_val.readonly().as_slice().unwrap().to_vec();
            let np_s_slice = np_s.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&s_slice, &np_s_slice);
        });
    }

    #[test]
    fn norm_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, -2.0], vec![3.0, 4.0]]);

            let norm_l1 = norm_py(py, a.readonly(), Some("1")).unwrap();
            let np_norm_l1: f64 = py
                .eval(
                    c_str!("__import__('numpy').sum(__import__('numpy').abs([[1.0, -2.0], [3.0, 4.0]]))"),
                    None,
                    None,
                )
                .unwrap()
                .extract()
                .unwrap();
            assert!((norm_l1 - np_norm_l1).abs() < 1e-9);

            let norm_l2 = norm_py(py, a.readonly(), Some("fro")).unwrap();
            let np_norm_l2: f64 = py
                .eval(
                    c_str!("__import__('numpy').linalg.norm([[1.0, -2.0], [3.0, 4.0]], 'fro')"),
                    None,
                    None,
                )
                .unwrap()
                .extract()
                .unwrap();
            assert!((norm_l2 - np_norm_l2).abs() < 1e-9);

            let norm_max = norm_py(py, a.readonly(), Some("max")).unwrap();
            let np_norm_max: f64 = py
                .eval(
                    c_str!("__import__('numpy').max(__import__('numpy').abs([[1.0, -2.0], [3.0, 4.0]]))"),
                    None,
                    None,
                )
                .unwrap()
                .extract()
                .unwrap();
            assert!((norm_max - np_norm_max).abs() < 1e-9);
        });
    }

    #[test]
    fn schur_matches_scipy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
            let (q_val, t_val) = schur_py(py, a.readonly()).unwrap();

            let q_readonly = q_val.readonly();
            let r_readonly = t_val.readonly();
            let q_mat = view_from_numpy(&q_readonly).unwrap();
            let t_mat = view_from_numpy(&r_readonly).unwrap();

            let mut qt = output_array::<f64>([2, 2]).unwrap();
            matmul(&q_mat, &t_mat, &mut qt.view_mut()).unwrap();

            let q_trans = q_mat.transpose([1, 0]).unwrap();
            let mut reconstruct = output_array::<f64>([2, 2]).unwrap();
            matmul(&qt.view(), &q_trans, &mut reconstruct.view_mut()).unwrap();

            assert_close_slice_f64(reconstruct.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn bunch_kaufman_matches_scipy_reconstruction() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![2.0, 3.0], vec![3.0, 5.0]]);
            let (l_val, d_val, perm_val) = bunch_kaufman_py(py, a.readonly()).unwrap();

            let l_readonly = l_val.readonly();
            let d_readonly = d_val.readonly();
            let l_mat = view_from_numpy(&l_readonly).unwrap();
            let d_mat = view_from_numpy(&d_readonly).unwrap();

            let mut ld = output_array::<f64>([2, 2]).unwrap();
            matmul(&l_mat, &d_mat, &mut ld.view_mut()).unwrap();

            let l_trans = l_mat.transpose([1, 0]).unwrap();
            let mut ldlt = output_array::<f64>([2, 2]).unwrap();
            matmul(&ld.view(), &l_trans, &mut ldlt.view_mut()).unwrap();

            let perm_slice = perm_val.readonly().as_slice().unwrap().to_vec();
            let a_readonly = a.readonly();
            let a_view = view_from_numpy(&a_readonly).unwrap();

            for i in 0..2 {
                for j in 0..2 {
                    let expected = *a_view
                        .get([perm_slice[i] as usize, perm_slice[j] as usize])
                        .unwrap();
                    let actual = *ldlt.get([i, j]).unwrap();
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
        });
    }

    #[test]
    fn matexp_matches_scipy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![0.0, 1.0], vec![-1.0, 0.0]]);
            let exp_val = matexp_py(py, a.readonly()).unwrap();

            let scipy_exp = py
                .eval(
                    c_str!("__import__('scipy').linalg.expm([[0.0, 1.0], [-1.0, 0.0]])"),
                    None,
                    None,
                )
                .unwrap()
                .extract::<PyReadonlyArray2<'_, f64>>()
                .unwrap();

            let exp_slice = exp_val.readonly().as_slice().unwrap().to_vec();
            let scipy_slice = scipy_exp.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&exp_slice, &scipy_slice);
        });
    }

    #[test]
    fn kron_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0]]);
            let b = array2_f64(py, &[vec![3.0, 4.0], vec![5.0, 6.0]]);

            let kron_val = kron_py(py, a.readonly(), b.readonly()).unwrap();

            let np_kron = py
                .eval(
                    c_str!("__import__('numpy').kron([[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]])"),
                    None,
                    None,
                )
                .unwrap()
                .extract::<PyReadonlyArray2<'_, f64>>()
                .unwrap();

            let kron_slice = kron_val.readonly().as_slice().unwrap().to_vec();
            let np_slice = np_kron.as_slice().unwrap().to_vec();

            assert_close_slice_f64(&kron_slice, &np_slice);
        });
    }

    #[test]
    fn trace_matches_numpy() {
        prepare_python();
        Python::with_gil(|py| {
            let a = array2_f64(py, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
            let trace_val = trace_py(py, a.readonly()).unwrap();

            let np_trace: f64 = py
                .eval(
                    c_str!("__import__('numpy').trace([[1.0, 2.0], [3.0, 4.0]])"),
                    None,
                    None,
                )
                .unwrap()
                .extract()
                .unwrap();

            assert!((trace_val - np_trace).abs() < 1e-9);
        });
    }
}
