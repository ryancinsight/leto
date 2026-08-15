//! Batched LU decomposition for a stack of independent square matrices.
//!
//! Each matrix in the batch is decomposed independently with partial pivoting,
//! enabling coarse-grained parallelism across the batch dimension.  The
//! per-matrix kernel is the same `O(n³)` elimination as
//! [`lu_decompose`](super::lu_decompose).

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use crate::domain::real::RealScalar;
#[cfg(not(feature = "parallel"))]
use leto::SliceArg;
use leto::{ArrayView3, LetoError, Result};

use super::lu::{lu_decompose, LuDecomposition};

/// Decompose a batch of square matrices `A_i` into `P_i · A_i = L_i · U_i`,
/// returning one [`LuDecomposition`] per matrix.
///
/// `matrices` has shape `[batch, n, n]`.  Each slice `matrices[i, :, :]` is an
/// `n × n` matrix decomposed independently.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if any matrix is non-square.  Singular matrices
/// return [`LetoError::StorageError`] from the first singular index.
pub fn lu_batch<T: RealScalar>(matrices: &ArrayView3<'_, T>) -> Result<Vec<LuDecomposition<T>>> {
    let [batch, nrows, ncols] = matrices.shape();
    if nrows != ncols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![nrows, ncols],
            rhs: vec![nrows, nrows],
        });
    }

    if batch == 0 {
        return Ok(Vec::new());
    }

    #[cfg(feature = "parallel")]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};

        let results_mtx: Arc<Mutex<Vec<Option<LuDecomposition<T>>>>> =
            Arc::new(Mutex::new((0..batch).map(|_| None).collect()));
        let had_error = Arc::new(AtomicBool::new(false));
        let error_slot: Arc<Mutex<Option<LetoError>>> = Arc::new(Mutex::new(None));
        let data_ptr = matrices.data().as_ptr() as usize;
        let data_len = matrices.data().len();
        let [_, n, _] = matrices.shape();
        let strides = matrices.strides();
        let offset = matrices.offset() as isize;

        let results_mtx_cloned = Arc::clone(&results_mtx);
        let had_error_cloned = Arc::clone(&had_error);
        let error_slot_cloned = Arc::clone(&error_slot);
        moirai::for_each_index_with::<moirai::Adaptive, _>(batch, move |b| {
            if had_error_cloned.load(Ordering::Relaxed) {
                return;
            }

            let layout = leto::Layout::try_new(
                [n, n],
                [strides[1], strides[2]],
                (offset + b as isize * strides[0]) as usize,
            )
            .expect("invariant: batch submatrix layout derives from a validated parent");
            let view = unsafe {
                leto::ArrayView2::<T>::new(
                    layout,
                    core::slice::from_raw_parts(data_ptr as *const T, data_len),
                )
            };
            match lu_decompose(&view) {
                Ok(lu) => {
                    let mut guard = results_mtx_cloned.lock().expect("lock");
                    guard[b] = Some(lu);
                }
                Err(e) => {
                    had_error_cloned.store(true, Ordering::Relaxed);
                    let mut guard = error_slot_cloned.lock().expect("lock");
                    if guard.is_none() {
                        *guard = Some(e);
                    }
                }
            }
        });

        let mut error_guard = error_slot.lock().expect("lock");
        if let Some(e) = error_guard.take() {
            return Err(e);
        }
        drop(error_guard);

        let mut results_guard = results_mtx.lock().expect("lock");
        let inner = core::mem::take(&mut *results_guard);
        Ok(inner
            .into_iter()
            .map(|opt| opt.expect("each slot filled"))
            .collect())
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut results: Vec<Option<LuDecomposition<T>>> = (0..batch).map(|_| None).collect();
        for (b, slot) in results.iter_mut().enumerate() {
            let mat = matrices.slice_with::<2>(&[
                SliceArg::Index(b as isize),
                SliceArg::All,
                SliceArg::All,
            ])?;
            match lu_decompose(&mat) {
                Ok(lu) => *slot = Some(lu),
                Err(e) => return Err(e),
            }
        }
        Ok(results
            .into_iter()
            .map(|opt| opt.expect("each slot filled"))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_batch_single_matrix() {
        let a = leto::Array3::from_shape_vec([1, 2, 2], vec![4.0f64, 3.0, 6.0, 3.0]).unwrap();
        let results = lu_batch(&a.view()).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].det() - (-6.0f64)).abs() < 1e-10);
    }

    #[test]
    fn lu_batch_two_matrices() {
        let a = leto::Array3::from_shape_vec(
            [2, 2, 2],
            vec![1.0f64, 0.0, 0.0, 1.0, 4.0, 3.0, 6.0, 3.0],
        )
        .unwrap();
        let results = lu_batch(&a.view()).unwrap();
        assert_eq!(results.len(), 2);
        assert!((results[0].det() - 1.0f64).abs() < 1e-10);
        assert!((results[1].det() - (-6.0f64)).abs() < 1e-10);
    }

    #[test]
    fn lu_batch_three_by_three() {
        // det = 2*(2*2 - 3*1) - 1*(4*2 - 3*(-2)) + 1*(4*1 - 2*(-2)) = -4
        let data: Vec<f64> = vec![2.0, 1.0, 1.0, 4.0, 2.0, 3.0, -2.0, 1.0, 2.0];
        let a = leto::Array3::from_shape_vec([1, 3, 3], data).unwrap();
        let results = lu_batch(&a.view()).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].det() - (-4.0f64)).abs() < 1e-10);
    }

    #[test]
    fn lu_batch_non_square_error() {
        let a = leto::Array3::<f64>::from_shape_vec([1, 2, 3], vec![1.0; 6]).unwrap();
        assert!(lu_batch(&a.view()).is_err());
    }

    #[test]
    fn lu_batch_empty() {
        let a = leto::Array3::<f64>::from_shape_vec([0, 2, 2], vec![]).unwrap();
        let results = lu_batch(&a.view()).unwrap();
        assert!(results.is_empty());
    }
}
