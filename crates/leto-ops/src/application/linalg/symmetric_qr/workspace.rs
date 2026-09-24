//! Public entry points: the reusable workspace and the owning decomposition.

use super::{ql, reduce};
use crate::application::linalg::SymmetricEigenDecomposition;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Reusable storage for the symmetric tridiagonal-QL eigensolver.
///
/// [`decompose`](Self::decompose) reuses every buffer, so a caller
/// decomposing many matrices of one order allocates only on the first call.
/// After a successful call, [`eigenvalues`](Self::eigenvalues) holds the
/// spectrum in ascending order and [`eigenvectors`](Self::eigenvectors) yields
/// the matching unit eigenvectors, each a contiguous slice.
///
/// See the [module documentation](super) for the algorithm and its accuracy.
///
/// # Examples
///
/// ```
/// use leto::Array2;
/// use leto_ops::SymmetricEigenWorkspace;
///
/// let a = Array2::from_shape_vec([2, 2], vec![2.0_f64, 1.0, 1.0, 2.0])?;
/// let mut workspace = SymmetricEigenWorkspace::new();
/// workspace.decompose(&a.view())?;
/// assert!((workspace.eigenvalues()[0] - 1.0).abs() < 1e-15);
/// assert!((workspace.eigenvalues()[1] - 3.0).abs() < 1e-15);
/// // A·v = λ·v for the top eigenpair.
/// let top = workspace.eigenvectors().next_back().expect("two eigenvectors");
/// assert!((2.0 * top[0] + top[1] - 3.0 * top[0]).abs() < 1e-15);
/// # Ok::<(), leto::LetoError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SymmetricEigenWorkspace<T> {
    order: usize,
    /// The working matrix; after the reduction, row `k` holds reflector `k`.
    reduced: Vec<T>,
    /// Eigenvectors as rows, in the order of `values`.
    vectors: Vec<T>,
    values: Vec<T>,
    off_diagonal: Vec<T>,
    reflector_scales: Vec<T>,
    scratch: Vec<T>,
}

impl<T> Default for SymmetricEigenWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SymmetricEigenWorkspace<T> {
    /// An empty workspace; the first [`decompose`](Self::decompose) sizes it.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            order: 0,
            reduced: Vec::new(),
            vectors: Vec::new(),
            values: Vec::new(),
            off_diagonal: Vec::new(),
            reflector_scales: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Order `n` of the last successfully decomposed matrix (`0` before the
    /// first success or after a failure).
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Eigenvalues of the last decomposed matrix, ascending.
    #[must_use]
    pub fn eigenvalues(&self) -> &[T] {
        &self.values[..self.order]
    }

    /// Unit eigenvectors of the last decomposed matrix, in the order of
    /// [`eigenvalues`](Self::eigenvalues); each item has length
    /// [`order`](Self::order).
    pub fn eigenvectors(&self) -> std::slice::ChunksExact<'_, T> {
        let n = self.order;
        self.vectors[..n * n].chunks_exact(n.max(1))
    }
}

impl<T: RealScalar> SymmetricEigenWorkspace<T> {
    /// Decompose the real symmetric matrix `matrix`, replacing the previous
    /// result.
    ///
    /// Only the lower triangle (diagonal included) is read; the strictly upper
    /// triangle is taken to mirror it, the LAPACK `uplo = 'L'` convention.
    ///
    /// # Errors
    ///
    /// - [`LetoError::ShapeMismatch`] when `matrix` is not square.
    /// - [`LetoError::StorageError`] when the lower triangle holds a NaN or
    ///   infinity, or the QL iteration does not converge within `30·n` sweeps.
    ///
    /// On error [`order`](Self::order) is `0` and no eigenpairs are exposed.
    pub fn decompose(&mut self, matrix: &ArrayView2<'_, T>) -> Result<()> {
        self.order = 0;
        let [rows, cols] = matrix.shape();
        if rows != cols {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![rows, cols],
                rhs: vec![rows, rows],
            });
        }
        let n = rows;
        self.load_lower_triangle(matrix, n)?;
        self.values.resize(n, T::ZERO);
        self.off_diagonal.resize(n, T::ZERO);
        self.reflector_scales.resize(n, T::ZERO);
        self.vectors.resize(n * n, T::ZERO);
        reduce::tridiagonalize(
            &mut self.reduced,
            n,
            &mut self.values,
            &mut self.off_diagonal,
            &mut self.reflector_scales,
            &mut self.scratch,
        );
        reduce::accumulate_transposed_factor(
            &self.reduced,
            n,
            &self.reflector_scales,
            &mut self.vectors,
        );
        ql::diagonalize(
            &mut self.values,
            &mut self.off_diagonal,
            &mut self.vectors,
            n,
        )?;
        self.order = n;
        Ok(())
    }

    /// Copy `matrix` into the working buffer as a full symmetric matrix built
    /// from its lower triangle, rejecting non-finite entries.
    fn load_lower_triangle(&mut self, matrix: &ArrayView2<'_, T>, n: usize) -> Result<()> {
        self.reduced.clear();
        if let Some(slice) = matrix.as_slice() {
            self.reduced.extend_from_slice(slice);
        } else {
            self.reduced = matrix.to_contiguous().into_storage().into_inner();
        }
        for i in 0..n {
            for j in 0..=i {
                let value = self.reduced[i * n + j];
                if !value.is_finite() {
                    return Err(LetoError::StorageError {
                        reason: format!(
                            "symmetric eigensolver input has non-finite entry at ({i}, {j})"
                        ),
                    });
                }
                self.reduced[j * n + i] = value;
            }
        }
        Ok(())
    }
}

/// Eigendecomposition of a real symmetric matrix by Householder
/// tridiagonalization and implicit-shift QL.
///
/// Returns the same [`SymmetricEigenDecomposition`] layout as
/// [`symmetric_eigen_jacobi`](crate::symmetric_eigen_jacobi) — ascending
/// eigenvalues, eigenvectors as columns — at `O(n³)` cost. Only the lower
/// triangle is read. A caller decomposing many matrices reuses a
/// [`SymmetricEigenWorkspace`] instead.
///
/// # Errors
///
/// As [`SymmetricEigenWorkspace::decompose`].
///
/// # Examples
///
/// ```
/// use leto::Array2;
/// use leto_ops::symmetric_eigen_qr;
///
/// // Path-graph Laplacian: eigenvalues 0, 1, 3.
/// let a = Array2::from_shape_vec([3, 3], vec![1.0_f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0])?;
/// let eigen = symmetric_eigen_qr(&a.view())?;
/// for (value, expected) in eigen.eigenvalues.iter().zip([0.0, 1.0, 3.0]) {
///     assert!((value - expected).abs() < 1e-14);
/// }
/// # Ok::<(), leto::LetoError>(())
/// ```
pub fn symmetric_eigen_qr<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<SymmetricEigenDecomposition<T>> {
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(matrix)?;
    let n = workspace.order();
    let mut columns = vec![T::ZERO; n * n];
    for (k, vector) in workspace.eigenvectors().enumerate() {
        for (row, &value) in vector.iter().enumerate() {
            columns[row * n + k] = value;
        }
    }
    Ok(SymmetricEigenDecomposition {
        eigenvalues: workspace.values,
        eigenvectors: Array2::from_shape_vec([n, n], columns)?,
    })
}
