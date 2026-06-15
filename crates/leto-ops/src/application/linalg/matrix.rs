//! Fluent rank-2 linear-algebra trait surface (ADR 0003).
//!
//! These traits layer a nalgebra-style method surface (`m.lu()`, `m.solve(&b)`,
//! `m.det()`, `m.matmul(&b)`) onto Leto's existing strided matrix
//! (`Array2`/`ArrayView2`) without a second buffer type. Each method is a thin,
//! monomorphized delegator to the authoritative free-function kernel in this
//! module's siblings ([`lu`](crate::application::linalg::lu), [`qr`](crate::application::linalg::qr), etc.); the free
//! functions remain the single source of truth, so no kernel is duplicated.
//!
//! The traits are role-segmented (interface segregation): construction/product,
//! norms, decompositions, and direct solves are separate contracts. A single
//! [`AsMatrixView`] bridge normalizes owned arrays, borrowed arrays, and views to
//! one rank-2 view so each LA trait is written once via a blanket impl. Because
//! arbitrary strided layouts are accepted by the kernels, a transposed or sliced
//! `ArrayView2` carries the full surface unchanged.

use crate::domain::real::RealScalar;
use crate::domain::scalar::Scalar;
use crate::{
    cholesky_decompose, lu_decompose, qr_decompose, svd_decompose, symmetric_eigen_jacobi,
    symmetric_eigenvalues_jacobi, CholeskyDecomposition, LuDecomposition, QrDecomposition,
    SvdDecomposition, SymmetricEigenDecomposition,
};
use crate::{
    det as det_kernel, inv as inv_kernel, matmul as matmul_kernel, norm_l1 as norm_l1_kernel,
    norm_l2 as norm_l2_kernel, norm_max as norm_max_kernel,
    singular_values as singular_values_kernel, solve as solve_kernel,
    solve_least_squares as solve_least_squares_kernel,
};
use leto::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Result, Storage};

/// Borrow any rank-2 receiver as a read-only [`ArrayView2`].
///
/// This is the single bridge that lets every linear-algebra trait below be
/// written once: owned arrays, borrowed arrays, and views all reduce to one view
/// type. It performs no copy — a view is the array's layout plus a borrowed
/// slice.
pub trait AsMatrixView<T> {
    /// Return a read-only rank-2 view of `self`.
    fn as_matrix_view(&self) -> ArrayView2<'_, T>;
}

impl<T> AsMatrixView<T> for ArrayView2<'_, T> {
    #[inline]
    fn as_matrix_view(&self) -> ArrayView2<'_, T> {
        // `data()` returns the underlying borrow; reconstruct a view with the
        // same (Copy) layout. No allocation, no element copy.
        ArrayView::new(self.layout(), self.data())
    }
}

impl<T, S: Storage<T>> AsMatrixView<T> for Array<T, S, 2> {
    #[inline]
    fn as_matrix_view(&self) -> ArrayView2<'_, T> {
        self.view()
    }
}

/// Matrix product surface.
///
/// ```
/// use leto::{Array2, Storage};
/// use leto_ops::MatrixProduct;
///
/// let a = Array2::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = Array2::from_shape_vec([3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let c = a.matmul(&b).unwrap();
/// assert_eq!(c.shape(), [2, 2]);
/// assert_eq!(c.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);
/// ```
pub trait MatrixProduct<T: Scalar> {
    /// Matrix multiply `self · rhs`, allocating a C-contiguous result.
    ///
    /// Allocates the `[m, n]` output and dispatches to the caller-owned
    /// [`matmul`](crate::matmul) kernel; for hot loops that reuse an output
    /// buffer, call that kernel directly.
    ///
    /// # Errors
    /// Propagates [`LetoError`](leto::LetoError) on inner-dimension mismatch.
    fn matmul<R: AsMatrixView<T>>(&self, rhs: &R) -> Result<Array2<T>>;
}

impl<T: Scalar, M: AsMatrixView<T>> MatrixProduct<T> for M {
    #[inline]
    fn matmul<R: AsMatrixView<T>>(&self, rhs: &R) -> Result<Array2<T>> {
        let lhs = self.as_matrix_view();
        let rhs = rhs.as_matrix_view();
        let [rows, _] = lhs.shape();
        let [_, cols] = rhs.shape();
        let mut out = Array2::from_shape_vec([rows, cols], vec![T::ZERO; rows * cols])?;
        matmul_kernel(&lhs, &rhs, &mut out.view_mut())?;
        Ok(out)
    }
}

/// Matrix norms (entrywise; `norm_l2` is the Frobenius norm).
///
/// ```
/// use leto::Array2;
/// use leto_ops::MatrixNorm;
///
/// let a = Array2::from_shape_vec([2, 2], vec![3.0_f64, 0.0, 0.0, 4.0]).unwrap();
/// assert!((a.norm_l2().unwrap() - 5.0).abs() < 1e-12);
/// ```
pub trait MatrixNorm<T: RealScalar> {
    /// Entrywise L1 norm `Σ |aᵢⱼ|`.
    ///
    /// # Errors
    /// Propagates [`LetoError`](leto::LetoError) on an invalid storage span.
    fn norm_l1(&self) -> Result<T>;
    /// Frobenius (entrywise L2) norm `sqrt(Σ aᵢⱼ²)`.
    ///
    /// # Errors
    /// Propagates [`LetoError`](leto::LetoError) on an invalid storage span.
    fn norm_l2(&self) -> Result<T>;
    /// Max-magnitude norm `max |aᵢⱼ|`.
    ///
    /// # Errors
    /// Propagates [`LetoError`](leto::LetoError) on an invalid storage span.
    fn norm_max(&self) -> Result<T>;
}

impl<T: RealScalar, M: AsMatrixView<T>> MatrixNorm<T> for M {
    #[inline]
    fn norm_l1(&self) -> Result<T> {
        norm_l1_kernel(&self.as_matrix_view())
    }
    #[inline]
    fn norm_l2(&self) -> Result<T> {
        norm_l2_kernel(&self.as_matrix_view())
    }
    #[inline]
    fn norm_max(&self) -> Result<T> {
        norm_max_kernel(&self.as_matrix_view())
    }
}

/// Matrix factorizations. Method names omit the algorithm (`symmetric_eigen`,
/// not `_jacobi`) since the algorithm is a kernel detail.
///
/// ```
/// use leto::Array2;
/// use leto_ops::MatrixDecompose;
///
/// let a = Array2::from_shape_vec([2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
/// let mut values = a.symmetric_eigenvalues().unwrap();
/// values.sort_by(|x, y| x.total_cmp(y));
/// // trace is preserved by the spectrum.
/// assert!((values.iter().sum::<f64>() - 7.0).abs() < 1e-9);
/// ```
pub trait MatrixDecompose<T: RealScalar> {
    /// LU decomposition with partial pivoting (`P·A = L·U`).
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-square or singular input.
    fn lu(&self) -> Result<LuDecomposition<T>>;
    /// Householder QR decomposition (`A = Q·R`).
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on invalid shape.
    fn qr(&self) -> Result<QrDecomposition<T>>;
    /// Cholesky factorization of a symmetric positive-definite matrix.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-SPD or non-square input.
    fn cholesky(&self) -> Result<CholeskyDecomposition<T>>;
    /// Thin SVD for finite full-rank matrices.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on rank-deficient or invalid input.
    fn svd(&self) -> Result<SvdDecomposition<T>>;
    /// Singular values (ascending in the kernel's convention), including
    /// rank-deficient inputs.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on a non-finite input.
    fn singular_values(&self) -> Result<Vec<T>>;
    /// Symmetric eigendecomposition (ascending eigenvalues, orthonormal vectors).
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-symmetric or non-finite input.
    fn symmetric_eigen(&self) -> Result<SymmetricEigenDecomposition<T>>;
    /// Symmetric eigenvalues only (no eigenvector storage).
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-symmetric or non-finite input.
    fn symmetric_eigenvalues(&self) -> Result<Vec<T>>;
}

impl<T: RealScalar, M: AsMatrixView<T>> MatrixDecompose<T> for M {
    #[inline]
    fn lu(&self) -> Result<LuDecomposition<T>> {
        lu_decompose(&self.as_matrix_view())
    }
    #[inline]
    fn qr(&self) -> Result<QrDecomposition<T>> {
        qr_decompose(&self.as_matrix_view())
    }
    #[inline]
    fn cholesky(&self) -> Result<CholeskyDecomposition<T>> {
        cholesky_decompose(&self.as_matrix_view())
    }
    #[inline]
    fn svd(&self) -> Result<SvdDecomposition<T>> {
        svd_decompose(&self.as_matrix_view())
    }
    #[inline]
    fn singular_values(&self) -> Result<Vec<T>> {
        singular_values_kernel(&self.as_matrix_view())
    }
    #[inline]
    fn symmetric_eigen(&self) -> Result<SymmetricEigenDecomposition<T>> {
        symmetric_eigen_jacobi(&self.as_matrix_view())
    }
    #[inline]
    fn symmetric_eigenvalues(&self) -> Result<Vec<T>> {
        symmetric_eigenvalues_jacobi(&self.as_matrix_view())
    }
}

/// Direct linear-algebra answers (solve / inverse / determinant).
///
/// ```
/// use leto::{Array, Array2, Storage};
/// use leto_ops::MatrixSolve;
///
/// let a = Array2::from_shape_vec([2, 2], vec![2.0_f64, 1.0, 1.0, 3.0]).unwrap();
/// let b = Array::from_shape_vec([2], vec![3.0_f64, 5.0]).unwrap();
/// let x = a.solve(&b.view()).unwrap();
/// // 2x+y=3, x+3y=5 -> x=0.8, y=1.4
/// assert!((x.storage().as_slice()[0] - 0.8).abs() < 1e-9);
/// assert!((x.storage().as_slice()[1] - 1.4).abs() < 1e-9);
/// ```
pub trait MatrixSolve<T: RealScalar> {
    /// Solve `self · x = rhs` for a square system via LU.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-square, singular, or shape-mismatched input.
    fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>>;
    /// Least-squares solution of an overdetermined system via QR.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on rank-deficient or shape-mismatched input.
    fn solve_least_squares(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>>;
    /// Matrix inverse via LU.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-square or singular input.
    fn inv(&self) -> Result<Array2<T>>;
    /// Determinant via LU (`0` for a singular matrix, per the kernel contract).
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on non-square input.
    fn det(&self) -> Result<T>;
}

impl<T: RealScalar, M: AsMatrixView<T>> MatrixSolve<T> for M {
    #[inline]
    fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        solve_kernel(&self.as_matrix_view(), rhs)
    }
    #[inline]
    fn solve_least_squares(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        solve_least_squares_kernel(&self.as_matrix_view(), rhs)
    }
    #[inline]
    fn inv(&self) -> Result<Array2<T>> {
        inv_kernel(&self.as_matrix_view())
    }
    #[inline]
    fn det(&self) -> Result<T> {
        det_kernel(&self.as_matrix_view())
    }
}
