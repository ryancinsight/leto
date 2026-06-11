use crate::domain::real::RealScalar;
use crate::domain::scalar::Scalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Eigenpairs of a real symmetric matrix.
///
/// Eigenvalues are sorted in ascending order. Eigenvectors are stored as
/// columns in a row-major Leto `Array2`, so eigenvector `k` is read with
/// `eigenvectors.get([row, k])`.
///
/// The decomposition is generic over the scalar type `T`. All iteration runs
/// in the native precision of `T` per the `Scalar` native-precision contract;
/// no hidden wider accumulator is introduced. A caller needing higher working
/// precision than the storage type converts the input first, making the
/// precision choice explicit.
#[derive(Debug, Clone)]
pub struct SymmetricEigenDecomposition<T> {
    /// Eigenvalues sorted in ascending order.
    pub eigenvalues: Vec<T>,
    /// Eigenvector matrix with eigenvectors stored in columns.
    pub eigenvectors: Array2<T>,
}

/// Default convergence tolerance: `1 / 10^12` expressed in `T`.
///
/// For `f64`/`f32` this is exactly `1e-12`. For reduced-precision types the
/// `10^12` literal saturates to infinity, yielding a tolerance of zero so the
/// solver runs to its bounded sweep cap rather than stopping early at a value
/// the type cannot represent.
#[inline]
fn default_tolerance<T: RealScalar>() -> T {
    T::ONE.div(T::from_usize(1_000_000_000_000))
}

/// Compute the eigendecomposition of a real symmetric matrix with Jacobi rotations.
///
/// This solver targets the small dense symmetric matrices currently needed by
/// Apollo graph and fractional Fourier plans. The input may be strided; it is
/// copied once into row-major working storage. The returned eigenvector matrix
/// is orthonormal up to the requested tolerance.
pub fn symmetric_eigen_jacobi<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<SymmetricEigenDecomposition<T>> {
    symmetric_eigen_jacobi_with_tolerance(matrix, default_tolerance::<T>())
}

/// Compute only the eigenvalues of a real symmetric matrix with Jacobi rotations.
///
/// This uses the same native-precision Jacobi diagonalization contract as
/// [`symmetric_eigen_jacobi`] but routes rotations through a zero-sized target
/// that does not allocate or update an eigenvector matrix.
pub fn symmetric_eigenvalues_jacobi<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    symmetric_eigenvalues_jacobi_with_tolerance(matrix, default_tolerance::<T>())
}

/// Compute only the eigenvalues of a real symmetric matrix with an explicit tolerance.
pub fn symmetric_eigenvalues_jacobi_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<Vec<T>> {
    validate_symmetric_input(matrix, tolerance)?;
    let [n, _] = matrix.shape();
    let mut a = copy_row_major(matrix);
    let mut target = NoEigenvectors;

    diagonalize(&mut a, n, tolerance, &mut target);
    Ok(sort_diagonal(&a, n))
}

/// Compute the eigendecomposition of a real symmetric matrix with an explicit tolerance.
pub fn symmetric_eigen_jacobi_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SymmetricEigenDecomposition<T>> {
    validate_symmetric_input(matrix, tolerance)?;
    let [n, _] = matrix.shape();
    let mut a = copy_row_major(matrix);
    let mut v = identity::<T>(n);
    let mut target = EigenvectorWorkspace { values: &mut v };
    diagonalize(&mut a, n, tolerance, &mut target);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&lhs, &rhs| {
        a[lhs * n + lhs]
            .partial_cmp(&a[rhs * n + rhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = vec![T::ZERO; n * n];
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

fn validate_symmetric_input<T: RealScalar>(matrix: &ArrayView2<'_, T>, tolerance: T) -> Result<()> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    if !tolerance.is_finite() || tolerance < T::ZERO {
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
            if value.sub(transposed).abs() > tolerance {
                return Err(LetoError::StorageError {
                    reason: "symmetric eigensolver input is not symmetric".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn copy_row_major<T: Scalar>(matrix: &ArrayView2<'_, T>) -> Vec<T> {
    let [rows, cols] = matrix.shape();
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(*matrix.get([row, col]).expect("validated matrix bounds"));
        }
    }
    values
}

fn identity<T: Scalar>(n: usize) -> Vec<T> {
    let mut values = vec![T::ZERO; n * n];
    for index in 0..n {
        values[index * n + index] = T::ONE;
    }
    values
}

fn sort_diagonal<T: RealScalar>(a: &[T], n: usize) -> Vec<T> {
    let mut eigenvalues = Vec::with_capacity(n);
    for index in 0..n {
        eigenvalues.push(a[index * n + index]);
    }
    eigenvalues.sort_by(|lhs, rhs| {
        lhs.partial_cmp(rhs)
            .expect("invariant: finite symmetric input yields finite diagonal")
    });
    eigenvalues
}

fn largest_off_diagonal<T: RealScalar>(a: &[T], n: usize) -> Option<(usize, usize, T)> {
    let mut best = None;
    let mut best_abs = T::ZERO;
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

trait RotationTarget<T: RealScalar> {
    fn rotate_columns(&mut self, n: usize, p: usize, q: usize, c: T, s: T);
}

struct NoEigenvectors;

impl<T: RealScalar> RotationTarget<T> for NoEigenvectors {
    #[inline]
    fn rotate_columns(&mut self, _n: usize, _p: usize, _q: usize, _c: T, _s: T) {}
}

struct EigenvectorWorkspace<'a, T> {
    values: &'a mut [T],
}

impl<T: RealScalar> RotationTarget<T> for EigenvectorWorkspace<'_, T> {
    #[inline]
    fn rotate_columns(&mut self, n: usize, p: usize, q: usize, c: T, s: T) {
        for row in 0..n {
            let vkp = self.values[row * n + p];
            let vkq = self.values[row * n + q];
            self.values[row * n + p] = c.mul(vkp).sub(s.mul(vkq));
            self.values[row * n + q] = s.mul(vkp).add(c.mul(vkq));
        }
    }
}

fn diagonalize<T, R>(a: &mut [T], n: usize, tolerance: T, target: &mut R)
where
    T: RealScalar,
    R: RotationTarget<T>,
{
    let max_sweeps = n.saturating_mul(n).saturating_mul(32).max(1);

    for _ in 0..max_sweeps {
        let Some((p, q, max_abs)) = largest_off_diagonal(a, n) else {
            break;
        };
        if max_abs <= tolerance {
            break;
        }
        rotate(a, target, n, p, q);
    }
}

fn rotate<T, R>(a: &mut [T], target: &mut R, n: usize, p: usize, q: usize)
where
    T: RealScalar,
    R: RotationTarget<T>,
{
    let app = a[p * n + p];
    let aqq = a[q * n + q];
    let apq = a[p * n + q];
    if apq == T::ZERO {
        return;
    }

    let two = T::from_usize(2);
    let half = T::ONE.div(two);
    // theta = 0.5 * atan2(2*apq, aqq - app)
    let theta = half.mul(two.mul(apq).atan2(aqq.sub(app)));
    let c = theta.cos();
    let s = theta.sin();

    for k in 0..n {
        if k != p && k != q {
            let akp = a[k * n + p];
            let akq = a[k * n + q];
            // new_kp = c*akp - s*akq ; new_kq = s*akp + c*akq
            let new_kp = c.mul(akp).sub(s.mul(akq));
            let new_kq = s.mul(akp).add(c.mul(akq));
            a[k * n + p] = new_kp;
            a[p * n + k] = new_kp;
            a[k * n + q] = new_kq;
            a[q * n + k] = new_kq;
        }
    }

    let c2 = c.mul(c);
    let s2 = s.mul(s);
    let sc = s.mul(c);
    // app' = c2*app - 2*sc*apq + s2*aqq
    a[p * n + p] = c2.mul(app).sub(two.mul(sc).mul(apq)).add(s2.mul(aqq));
    // aqq' = s2*app + 2*sc*apq + c2*aqq
    a[q * n + q] = s2.mul(app).add(two.mul(sc).mul(apq)).add(c2.mul(aqq));
    a[p * n + q] = T::ZERO;
    a[q * n + p] = T::ZERO;

    target.rotate_columns(n, p, q, c, s);
}
