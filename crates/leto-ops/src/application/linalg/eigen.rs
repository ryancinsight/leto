use crate::application::linalg::thresholds::{machine_epsilon, scaled_frobenius};
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

/// Default convergence tolerance: `ε`, the machine epsilon of `T`.
///
/// Rotation stops when every off-diagonal entry is negligible against the
/// diagonal entries it couples, `|a_pq| ≤ ε·√(|a_pp|·|a_qq|)` — the
/// criterion under which Jacobi computes each eigenvalue to high *relative*
/// accuracy, not only to `ε·‖A‖` (Demmel & Veselić 1992, "Jacobi's method is
/// more accurate than QR", *SIAM J. Matrix Anal. Appl.* 13(4), §4) — with the
/// normwise floor `ε·(ε·‖A‖_F)` for pairs whose diagonal has vanished, where
/// the pair criterion alone would demand an exact zero. Both terms scale with
/// `A`, so no magnitude stops early; a pair whose diagonals are near `‖A‖`
/// stops at `ε·‖A‖`, so well-scaled matrices are not charged the `ε²`
/// normwise cost.
#[inline]
fn default_tolerance<T: RealScalar>() -> T {
    machine_epsilon::<T>()
}

/// Accept `A` as symmetric when every pair satisfies
/// `|aᵢⱼ − aⱼᵢ| ≤ 2ε·max(|aᵢⱼ|, |aⱼᵢ|, ‖A‖_F/n)`.
///
/// Derivation: a matrix symmetric in exact arithmetic but assembled along two
/// rounding paths carries up to two roundings (`u = ε/2` each) on each side of
/// a pair, so `|aᵢⱼ − aⱼᵢ| ≤ 4u·max(|aᵢⱼ|, |aⱼᵢ|) = 2ε·max(…)`; `0.1 + 0.2`
/// against `0.3` differs by one ulp of `0.3`, `5.55e-17 ≤ 2ε·0.3 = 1.3e-16`. The
/// `‖A‖_F/n` floor covers entries produced by cancellation, whose rounding is
/// relative to their operands rather than to themselves: an asymmetry of
/// `2ε·‖A‖_F/n` in each of the `n²` entries has Frobenius norm `2ε·‖A‖_F`,
/// inside the solver's own backward error.
fn symmetry_bound<T: RealScalar>(lhs: T, rhs: T, floor: T) -> T {
    let two_epsilon = machine_epsilon::<T>().mul(T::from_usize(2));
    let (lhs, rhs) = (lhs.abs(), rhs.abs());
    let larger = if lhs > rhs { lhs } else { rhs };
    let scale = if larger > floor { larger } else { floor };
    two_epsilon.mul(scale)
}

/// Compute the eigendecomposition of a real symmetric matrix with Jacobi rotations.
///
/// This solver targets the small dense symmetric matrices currently needed by
/// Apollo graph and fractional Fourier plans. The input may be strided; it is
/// copied once into row-major working storage. The returned eigenvector matrix
/// is orthonormal up to the requested tolerance. The tolerance is `ε`,
/// applied relative to each coupled diagonal pair; see
/// [`symmetric_eigen_jacobi_with_tolerance`].
///
/// # Errors
///
/// As [`symmetric_eigen_jacobi_with_tolerance`].
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
///
/// # Errors
///
/// As [`symmetric_eigen_jacobi_with_tolerance`].
pub fn symmetric_eigenvalues_jacobi<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    symmetric_eigenvalues_jacobi_with_tolerance(matrix, default_tolerance::<T>())
}

/// Compute only the eigenvalues of a real symmetric matrix with an explicit
/// relative tolerance (see [`symmetric_eigen_jacobi_with_tolerance`]).
///
/// # Errors
///
/// As [`symmetric_eigen_jacobi_with_tolerance`].
pub fn symmetric_eigenvalues_jacobi_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<Vec<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;
    let mut a = copy_row_major(matrix);
    validate_symmetric_input(&a, n, tolerance)?;
    let mut target = NoEigenvectors;

    diagonalize(&mut a, n, tolerance, &mut target)?;
    Ok(sort_diagonal(&a, n))
}

/// Compute the eigendecomposition of a real symmetric matrix with an explicit
/// relative tolerance.
///
/// Rotations continue until every off-diagonal entry satisfies
/// `|a_pq| ≤ τ·max(√(|a_pp|·|a_qq|), τ·‖A‖_F)`, `τ = tolerance`: negligible
/// against the diagonal pair it couples (relative accuracy of small
/// eigenvalues), floored at `τ²·‖A‖_F` where that pair has vanished. The
/// stopping point scales with `A`, so a matrix of any magnitude is solved to
/// the same relative accuracy. Symmetry acceptance is
/// independent of the tolerance: `|aᵢⱼ − aⱼᵢ| ≤ 2ε·max(|aᵢⱼ|, |aⱼᵢ|, ‖A‖_F/n)`,
/// the rounding a matrix assembled symmetric in exact arithmetic can carry.
///
/// # Errors
///
/// - [`LetoError::ShapeMismatch`] when the matrix is not square.
/// - [`LetoError::InvalidInput`] for a negative or non-finite tolerance, a
///   non-finite entry, or an asymmetric matrix.
/// - [`LetoError::ConvergenceError`] when `32·n²` rotations leave an
///   off-diagonal entry above its threshold; `residual` is the largest such
///   magnitude relative to `‖A‖_F`.
pub fn symmetric_eigen_jacobi_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SymmetricEigenDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;
    let mut a = copy_row_major(matrix);
    validate_symmetric_input(&a, n, tolerance)?;
    let mut v = identity::<T>(n);
    let mut target = EigenvectorWorkspace { values: &mut v };
    diagonalize(&mut a, n, tolerance, &mut target)?;

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

fn validate_symmetric_input<T: RealScalar>(a: &[T], n: usize, tolerance: T) -> Result<()> {
    if !tolerance.is_finite() || tolerance < T::ZERO {
        return Err(LetoError::InvalidInput(
            "eigensolver tolerance must be finite and non-negative".to_string(),
        ));
    }
    if let Some(index) = a.iter().position(|value| !value.is_finite()) {
        return Err(LetoError::InvalidInput(format!(
            "symmetric eigensolver input has a non-finite entry at ({}, {})",
            index / n,
            index % n
        )));
    }
    let floor = scaled_frobenius(a).div(T::from_usize(n.max(1)));
    for row in 0..n {
        for col in (row + 1)..n {
            let (upper, lower) = (a[row * n + col], a[col * n + row]);
            if upper.sub(lower).abs() > symmetry_bound(upper, lower, floor) {
                return Err(LetoError::InvalidInput(format!(
                    "symmetric eigensolver input is not symmetric at ({row}, {col})"
                )));
            }
        }
    }
    Ok(())
}

fn copy_row_major<T: Scalar>(matrix: &ArrayView2<'_, T>) -> Vec<T> {
    // One bulk row-major copy instead of per-element bounds-checked gets.
    if let Some(slice) = matrix.as_slice() {
        slice.to_vec()
    } else {
        matrix.to_contiguous().into_storage().into_inner()
    }
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

/// The largest off-diagonal entry not yet negligible against its pair's
/// diagonal: `|a_pq| > τ·max(√|a_pp|·√|a_qq|, τ·‖A‖_F)`, as
/// `(p, q, |a_pq|)`. `roots` is scratch for the `n` diagonal square roots.
fn largest_unconverged<T: RealScalar>(
    a: &[T],
    n: usize,
    tolerance: T,
    floor: T,
    roots: &mut [T],
) -> Option<(usize, usize, T)> {
    for (index, root) in roots.iter_mut().enumerate() {
        *root = a[index * n + index].abs().sqrt();
    }
    let mut best = None;
    let mut best_abs = T::ZERO;
    for row in 0..n {
        for col in (row + 1)..n {
            let value = a[row * n + col].abs();
            if value <= best_abs {
                continue;
            }
            let coupling = roots[row].mul(roots[col]);
            let scale = if coupling > floor { coupling } else { floor };
            if value > tolerance.mul(scale) {
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

/// Rotation budget: `32·n²`, about sixteen cyclic sweeps' worth; classical
/// Jacobi converges quadratically once the off-diagonal is small, in a few
/// sweeps of `n²/2` rotations.
fn rotation_budget(n: usize) -> usize {
    n.saturating_mul(n).saturating_mul(32).max(1)
}

fn diagonalize<T, R>(a: &mut [T], n: usize, tolerance: T, target: &mut R) -> Result<()>
where
    T: RealScalar,
    R: RotationTarget<T>,
{
    diagonalize_within(a, n, tolerance, rotation_budget(n), target)
}

/// [`diagonalize`] with an explicit rotation budget.
fn diagonalize_within<T, R>(
    a: &mut [T],
    n: usize,
    tolerance: T,
    max_rotations: usize,
    target: &mut R,
) -> Result<()>
where
    T: RealScalar,
    R: RotationTarget<T>,
{
    let norm = scaled_frobenius(a);
    let floor = tolerance.mul(norm);
    let mut roots = vec![T::ZERO; n];

    for _ in 0..max_rotations {
        let Some((p, q, _)) = largest_unconverged(a, n, tolerance, floor, &mut roots) else {
            return Ok(());
        };
        rotate(a, target, n, p, q);
    }
    match largest_unconverged(a, n, tolerance, floor, &mut roots) {
        Some((_, _, max_abs)) => Err(LetoError::ConvergenceError {
            max_iters: max_rotations,
            residual: max_abs.div(norm).to_f64(),
            tol: tolerance.to_f64(),
        }),
        None => Ok(()),
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

#[cfg(test)]
mod tests {
    use super::{diagonalize_within, NoEigenvectors};
    use leto::LetoError;

    #[test]
    fn exhausted_rotation_budget_is_a_typed_convergence_error() {
        // [[2,1,1],[1,2,1],[1,1,2]]: the first rotation (pivot (0,1), equal
        // diagonals, θ = π/4) zeroes a₀₁ and a₀₂ = (1 − 1)/√2 but leaves
        // a₁₂ = (1 + 1)/√2 = √2, so one rotation stops at residual √2/‖A‖_F
        // = √2/√18 = 1/3 (a few roundings, 8ε relative).
        let mut a = [2.0_f64, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0];
        let result = diagonalize_within(&mut a, 3, 1e-12, 1, &mut NoEigenvectors);
        let Err(LetoError::ConvergenceError {
            max_iters,
            residual,
            tol,
        }) = result
        else {
            panic!("expected ConvergenceError, got {result:?}");
        };
        assert_eq!(max_iters, 1);
        assert_eq!(tol, 1e-12);
        let expected = 1.0 / 3.0;
        assert!(
            (residual / expected - 1.0).abs() <= 8.0 * f64::EPSILON,
            "{residual}"
        );
    }
}
