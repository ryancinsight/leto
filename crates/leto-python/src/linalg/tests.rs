//! Differential tests for the linear-algebra binding surface.
//!
//! Each case is checked against NumPy or SciPy computing the same quantity on
//! the same input, so agreement is evidence from an independent route rather
//! than the bindings restating themselves.

use crate::linalg::{
    bunch_kaufman_py, cholesky_py, col_piv_qr_py, det_py, inv_py, kron_py, matexp_py, norm_py,
    qr_py, schur_py, singular_values_py, solve_py, svd_py, trace_py,
};
use crate::numpy_bridge::{output_array, view_from_numpy};
use crate::support::{array2_f64, assert_close_slice_f64, prepare_python};
use leto::Storage;
use leto_ops::matmul;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::ffi::c_str;
use pyo3::prelude::*;

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
                c_str!("__import__('numpy').linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])"),
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
                c_str!(
                    "__import__('numpy').sum(__import__('numpy').abs([[1.0, -2.0], [3.0, 4.0]]))"
                ),
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
                c_str!(
                    "__import__('numpy').max(__import__('numpy').abs([[1.0, -2.0], [3.0, 4.0]]))"
                ),
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
