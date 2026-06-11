use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

use super::eigen::symmetric_eigen_jacobi_with_tolerance;

/// Thin singular value decomposition of a full-rank matrix.
///
/// The decomposition stores `A = U * diag(singular_values) * V^T`, where
/// `U` has shape `m x k`, `V` has shape `n x k`, `k = min(m, n)`, and singular
/// values are sorted in descending order. Rank-deficient matrices are rejected
/// instead of returning fabricated vectors for null-space components.
#[derive(Debug, Clone)]
pub struct SvdDecomposition<T> {
    /// Singular values sorted descending.
    pub singular_values: Vec<T>,
    /// Thin left singular vectors, stored as columns.
    pub left_singular_vectors: Array2<T>,
    /// Right singular vectors, stored as columns.
    pub right_singular_vectors: Array2<T>,
}

/// Compute a thin SVD for a full-rank matrix.
///
/// The implementation is generic over [`RealScalar`] and runs in the native
/// precision of `T`: tall/square inputs form `A^T A` and derive
/// `U = A V Σ^-1`; wide inputs form `A A^T` and derive `V = A^T U Σ^-1`. The
/// method is appropriate for small dense matrices and consumer migration tests;
/// it is not a replacement for a rank-revealing bidiagonal SVD on
/// ill-conditioned large matrices.
pub fn svd_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<SvdDecomposition<T>> {
    svd_decompose_with_tolerance(matrix, default_tolerance::<T>())
}

/// Compute a thin SVD with an explicit eigensolver tolerance.
pub fn svd_decompose_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SvdDecomposition<T>> {
    validate_input(matrix, tolerance)?;
    let [rows, cols] = matrix.shape();
    if rows >= cols {
        svd_from_column_gram(matrix, tolerance)
    } else {
        svd_from_row_gram(matrix, tolerance)
    }
}

fn svd_from_column_gram<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SvdDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    let gram = column_gram_matrix(matrix)?;
    let eigen = symmetric_eigen_jacobi_with_tolerance(&gram.view(), tolerance)?;

    let mut singular_values = Vec::with_capacity(cols);
    let mut right_values = vec![T::ZERO; cols * cols];
    let mut left_values = vec![T::ZERO; rows * cols];

    for new_col in 0..cols {
        let old_col = cols - 1 - new_col;
        let eigenvalue = eigen.eigenvalues[old_col];
        let sigma = checked_singular_value(eigenvalue, tolerance)?;
        singular_values.push(sigma);

        for row in 0..cols {
            right_values[row * cols + new_col] = *eigen
                .eigenvectors
                .get([row, old_col])
                .expect("eigenvector bounds");
        }

        for row in 0..rows {
            let mut acc = T::ZERO;
            for col in 0..cols {
                let a = *matrix.get([row, col])?;
                let v = right_values[col * cols + new_col];
                acc = acc.add(a.mul(v));
            }
            left_values[row * cols + new_col] = acc.div(sigma);
        }

        normalize_column(&mut left_values, rows, cols, new_col, tolerance, "left")?;
    }

    Ok(SvdDecomposition {
        singular_values,
        left_singular_vectors: Array2::from_shape_vec([rows, cols], left_values)
            .expect("left singular vector shape matches storage"),
        right_singular_vectors: Array2::from_shape_vec([cols, cols], right_values)
            .expect("right singular vector shape matches storage"),
    })
}

fn svd_from_row_gram<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SvdDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    let gram = row_gram_matrix(matrix)?;
    let eigen = symmetric_eigen_jacobi_with_tolerance(&gram.view(), tolerance)?;

    let mut singular_values = Vec::with_capacity(rows);
    let mut left_values = vec![T::ZERO; rows * rows];
    let mut right_values = vec![T::ZERO; cols * rows];

    for new_col in 0..rows {
        let old_col = rows - 1 - new_col;
        let eigenvalue = eigen.eigenvalues[old_col];
        let sigma = checked_singular_value(eigenvalue, tolerance)?;
        singular_values.push(sigma);

        for row in 0..rows {
            left_values[row * rows + new_col] = *eigen
                .eigenvectors
                .get([row, old_col])
                .expect("eigenvector bounds");
        }

        for col in 0..cols {
            let mut acc = T::ZERO;
            for row in 0..rows {
                let a = *matrix.get([row, col])?;
                let u = left_values[row * rows + new_col];
                acc = acc.add(a.mul(u));
            }
            right_values[col * rows + new_col] = acc.div(sigma);
        }

        normalize_column(&mut right_values, cols, rows, new_col, tolerance, "right")?;
    }

    Ok(SvdDecomposition {
        singular_values,
        left_singular_vectors: Array2::from_shape_vec([rows, rows], left_values)
            .expect("left singular vector shape matches storage"),
        right_singular_vectors: Array2::from_shape_vec([cols, rows], right_values)
            .expect("right singular vector shape matches storage"),
    })
}

/// Return singular values for a full-rank matrix.
pub fn singular_values<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    svd_decompose(matrix).map(|decomposition| decomposition.singular_values)
}

fn default_tolerance<T: RealScalar>() -> T {
    T::ONE.div(T::from_usize(1_000_000_000_000))
}

fn validate_input<T: RealScalar>(matrix: &ArrayView2<'_, T>, tolerance: T) -> Result<()> {
    let [rows, cols] = matrix.shape();
    if rows == 0 || cols == 0 {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows.max(1), cols.max(1)],
        });
    }
    if !tolerance.is_finite() || tolerance < T::ZERO {
        return Err(LetoError::StorageError {
            reason: "SVD tolerance must be finite and non-negative".to_string(),
        });
    }
    for row in 0..rows {
        for col in 0..cols {
            if !matrix.get([row, col])?.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "SVD input contains a non-finite value".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn column_gram_matrix<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    let mut values = vec![T::ZERO; cols * cols];
    for lhs in 0..cols {
        for rhs in lhs..cols {
            let mut acc = T::ZERO;
            for row in 0..rows {
                acc = acc.add(matrix.get([row, lhs])?.mul(*matrix.get([row, rhs])?));
            }
            values[lhs * cols + rhs] = acc;
            values[rhs * cols + lhs] = acc;
        }
    }
    Ok(Array2::from_shape_vec([cols, cols], values).expect("gram shape matches storage"))
}

fn row_gram_matrix<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    let mut values = vec![T::ZERO; rows * rows];
    for lhs in 0..rows {
        for rhs in lhs..rows {
            let mut acc = T::ZERO;
            for col in 0..cols {
                acc = acc.add(matrix.get([lhs, col])?.mul(*matrix.get([rhs, col])?));
            }
            values[lhs * rows + rhs] = acc;
            values[rhs * rows + lhs] = acc;
        }
    }
    Ok(Array2::from_shape_vec([rows, rows], values).expect("gram shape matches storage"))
}

fn checked_singular_value<T: RealScalar>(eigenvalue: T, tolerance: T) -> Result<T> {
    if eigenvalue < T::ZERO {
        if eigenvalue.neg() > tolerance {
            return Err(LetoError::StorageError {
                reason: "SVD normal matrix has a negative eigenvalue beyond tolerance".to_string(),
            });
        }
        return Err(LetoError::StorageError {
            reason: "SVD input is rank-deficient".to_string(),
        });
    }
    let sigma = eigenvalue.sqrt();
    if sigma <= tolerance {
        return Err(LetoError::StorageError {
            reason: "SVD input is rank-deficient".to_string(),
        });
    }
    Ok(sigma)
}

fn normalize_column<T: RealScalar>(
    values: &mut [T],
    rows: usize,
    cols: usize,
    col: usize,
    tolerance: T,
    side: &str,
) -> Result<()> {
    let mut norm_sq = T::ZERO;
    for row in 0..rows {
        let value = values[row * cols + col];
        norm_sq = norm_sq.add(value.mul(value));
    }
    let norm = norm_sq.sqrt();
    if norm <= tolerance {
        return Err(LetoError::StorageError {
            reason: format!("SVD {side} singular vector has zero norm"),
        });
    }
    for row in 0..rows {
        values[row * cols + col] = values[row * cols + col].div(norm);
    }
    Ok(())
}
