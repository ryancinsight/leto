//! Python bindings for linear-algebra decompositions and solvers.

use crate::numpy_bridge::{
    array_into_numpy1, array_into_numpy2, require_contiguous_1d, require_contiguous_2d,
    view_from_numpy, view_from_numpy_1d,
};
use leto::Array1;
use leto_ops::{
    bidiagonalize, bunch_kaufman, cholesky_decompose, cholesky_inv, cholesky_solve, col_piv_qr,
    det, eigenvalues, full_piv_lu, hessenberg, inv, kron, matexp, norm_l1, norm_l2, norm_max,
    qr_decompose, schur, singular_values, solve, svd_decompose, symmetric_eigen_jacobi, trace,
    udu_decompose,
};
use numpy::{Complex64, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "det")]
pub(crate) fn det_py(py: Python<'_>, a: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    py.allow_threads(|| det(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))
}

#[pyfunction]
#[pyo3(name = "inv")]
pub(crate) fn inv_py<'py>(
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
pub(crate) fn solve_py<'py>(
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
pub(crate) fn cholesky_py<'py>(
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

/// Solve `A x = b` for symmetric positive-definite `A` via its Cholesky factor.
#[pyfunction]
#[pyo3(name = "cholesky_solve")]
pub(crate) fn cholesky_solve_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_2d(&a, "a")?;
    require_contiguous_1d(&b, "b")?;
    let a_view = view_from_numpy(&a)?;
    let b_view = view_from_numpy_1d(&b)?;
    let out_arr = py.allow_threads(|| {
        cholesky_solve(&a_view, &b_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    array_into_numpy1(py, out_arr)
}

/// Inverse of a symmetric positive-definite matrix via its Cholesky factor.
#[pyfunction]
#[pyo3(name = "cholesky_inv")]
pub(crate) fn cholesky_inv_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let out_arr = py.allow_threads(|| {
        cholesky_inv(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let shape = [out_arr.shape()[0], out_arr.shape()[1]];
    array_into_numpy2(py, out_arr, shape)
}

#[pyfunction]
#[pyo3(name = "qr")]
pub(crate) fn qr_py<'py>(
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
pub(crate) fn col_piv_qr_py<'py>(
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
pub(crate) fn svd_py<'py>(
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
pub(crate) fn symmetric_eigen_py<'py>(
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
pub(crate) fn singular_values_py<'py>(
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
pub(crate) fn norm_py(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f64>,
    ord: Option<&str>,
) -> PyResult<f64> {
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
pub(crate) fn schur_py<'py>(
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
pub(crate) fn bunch_kaufman_py<'py>(
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
pub(crate) fn matexp_py<'py>(
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
pub(crate) fn kron_py<'py>(
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
pub(crate) fn trace_py(py: Python<'_>, a: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    py.allow_threads(|| trace(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))
}

/// Hessenberg reduction `A = Q H Qᵀ`. Returns `(Q, H)`.
#[pyfunction]
#[pyo3(name = "hessenberg")]
pub(crate) fn hessenberg_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp =
        py.allow_threads(|| hessenberg(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let q = decomp.q().clone();
    let h = decomp.h().clone();
    let q_shape = [q.shape()[0], q.shape()[1]];
    let h_shape = [h.shape()[0], h.shape()[1]];
    let py_q = array_into_numpy2(py, q, q_shape)?;
    let py_h = array_into_numpy2(py, h, h_shape)?;
    Ok((py_q, py_h))
}

/// Eigenvalues of a general real matrix, returned as a complex vector.
#[pyfunction]
#[pyo3(name = "eigenvalues")]
pub(crate) fn eigenvalues_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let vals = py
        .allow_threads(|| eigenvalues(&a_view).map_err(|e| PyValueError::new_err(e.to_string())))?;
    let out: Vec<Complex64> = vals
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

/// Full-pivoting LU: returns `(L, U, row_perm, col_perm)`.
#[pyfunction]
#[pyo3(name = "full_piv_lu")]
pub(crate) fn full_piv_lu_py<'py>(
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

/// Pivot-free `A = U D Uᵀ`. Returns `(U, d)` with `d` the diagonal of `D`.
#[pyfunction]
#[pyo3(name = "udu")]
pub(crate) fn udu_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    require_contiguous_2d(&a, "a")?;
    let a_view = view_from_numpy(&a)?;
    let decomp = py.allow_threads(|| {
        udu_decompose(&a_view).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let u = decomp.u();
    let u_shape = [u.shape()[0], u.shape()[1]];
    let d = decomp.diagonal().to_vec();
    let py_u = array_into_numpy2(py, u, u_shape)?;
    Ok((py_u, PyArray1::from_vec(py, d)))
}

/// Bidiagonalization `A = U B Vᵀ`. Returns `(U, B, V)`.
#[pyfunction]
#[pyo3(name = "bidiagonalize")]
pub(crate) fn bidiagonalize_py<'py>(
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(det_py, m)?)?;
    m.add_function(wrap_pyfunction!(inv_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_py, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky_py, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky_solve_py, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky_inv_py, m)?)?;
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
mod tests;
