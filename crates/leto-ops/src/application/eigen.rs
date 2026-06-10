use leto::{Array2, ArrayView2, LetoError, Result};

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

/// Eigenpairs of a real symmetric matrix.
///
/// Eigenvalues are sorted in ascending order. Eigenvectors are stored as
/// columns in a row-major Leto `Array2`, so eigenvector `k` is read with
/// `eigenvectors.get([row, k])`.
#[derive(Debug, Clone)]
pub struct SymmetricEigenDecomposition {
    /// Eigenvalues sorted in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix with eigenvectors stored in columns.
    pub eigenvectors: Array2<f64>,
}

/// Compute the eigendecomposition of a real symmetric matrix with Jacobi rotations.
///
/// This solver targets the small dense symmetric matrices currently needed by
/// Apollo graph and fractional Fourier plans. The input may be strided; it is
/// copied once into row-major working storage. The returned eigenvector matrix
/// is orthonormal up to the requested tolerance.
pub fn symmetric_eigen_jacobi(matrix: &ArrayView2<'_, f64>) -> Result<SymmetricEigenDecomposition> {
    symmetric_eigen_jacobi_with_tolerance(matrix, DEFAULT_TOLERANCE)
}

/// Compute the eigendecomposition of a real symmetric matrix with an explicit tolerance.
pub fn symmetric_eigen_jacobi_with_tolerance(
    matrix: &ArrayView2<'_, f64>,
    tolerance: f64,
) -> Result<SymmetricEigenDecomposition> {
    validate_symmetric_input(matrix, tolerance)?;
    let [n, _] = matrix.shape();
    let mut a = copy_row_major(matrix);
    let mut v = identity(n);
    let max_sweeps = n.saturating_mul(n).saturating_mul(32).max(1);

    for _ in 0..max_sweeps {
        let Some((p, q, max_abs)) = largest_off_diagonal(&a, n) else {
            break;
        };
        if max_abs <= tolerance {
            break;
        }
        rotate(&mut a, &mut v, n, p, q);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&lhs, &rhs| {
        a[lhs * n + lhs]
            .partial_cmp(&a[rhs * n + rhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = vec![0.0; n * n];
    for (new_col, old_col) in order.into_iter().enumerate() {
        eigenvalues.push(a[old_col * n + old_col]);
        for row in 0..n {
            eigenvectors[row * n + new_col] = v[row * n + old_col];
        }
    }

    Ok(SymmetricEigenDecomposition {
        eigenvalues,
        eigenvectors: Array2::from_shape_vec([n, n], eigenvectors)
            .expect("eigenvector shape matches storage"),
    })
}

fn validate_symmetric_input(matrix: &ArrayView2<'_, f64>, tolerance: f64) -> Result<()> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(LetoError::StorageError {
            reason: "eigensolver tolerance must be finite and non-negative".to_string(),
        });
    }
    for row in 0..rows {
        for col in 0..cols {
            let value = *matrix.get([row, col])?;
            if !value.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "symmetric eigensolver input contains a non-finite value".to_string(),
                });
            }
            let transposed = *matrix.get([col, row])?;
            if (value - transposed).abs() > tolerance {
                return Err(LetoError::StorageError {
                    reason: "symmetric eigensolver input is not symmetric".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn copy_row_major(matrix: &ArrayView2<'_, f64>) -> Vec<f64> {
    let [rows, cols] = matrix.shape();
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(*matrix.get([row, col]).expect("validated matrix bounds"));
        }
    }
    values
}

fn identity(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n * n];
    for index in 0..n {
        values[index * n + index] = 1.0;
    }
    values
}

fn largest_off_diagonal(a: &[f64], n: usize) -> Option<(usize, usize, f64)> {
    let mut best = None;
    let mut best_abs = 0.0;
    for row in 0..n {
        for col in (row + 1)..n {
            let value = a[row * n + col].abs();
            if value > best_abs {
                best_abs = value;
                best = Some((row, col, value));
            }
        }
    }
    best
}

fn rotate(a: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let app = a[p * n + p];
    let aqq = a[q * n + q];
    let apq = a[p * n + q];
    if apq == 0.0 {
        return;
    }

    let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
    let c = theta.cos();
    let s = theta.sin();

    for k in 0..n {
        if k != p && k != q {
            let akp = a[k * n + p];
            let akq = a[k * n + q];
            let new_kp = c * akp - s * akq;
            let new_kq = s * akp + c * akq;
            a[k * n + p] = new_kp;
            a[p * n + k] = new_kp;
            a[k * n + q] = new_kq;
            a[q * n + k] = new_kq;
        }
    }

    let c2 = c * c;
    let s2 = s * s;
    let sc = s * c;
    a[p * n + p] = c2 * app - 2.0 * sc * apq + s2 * aqq;
    a[q * n + q] = s2 * app + 2.0 * sc * apq + c2 * aqq;
    a[p * n + q] = 0.0;
    a[q * n + p] = 0.0;

    for row in 0..n {
        let vkp = v[row * n + p];
        let vkq = v[row * n + q];
        v[row * n + p] = c * vkp - s * vkq;
        v[row * n + q] = s * vkp + c * vkq;
    }
}
