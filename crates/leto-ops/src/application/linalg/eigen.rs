use crate::application::linalg::scaling;
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
/// diagonal entries it couples, `|a_pq| ≤ ε·√(|a_pp|·|a_qq|)`, with the
/// normwise floor `ε·(ε·‖A‖_F)` for pairs whose diagonal has vanished, where
/// the pair criterion alone would demand an exact zero. Both terms scale with
/// `A`, so no magnitude stops early; a pair whose diagonals are near `‖A‖`
/// stops at `ε·‖A‖`, so well-scaled matrices are not charged the `ε²`
/// normwise cost.
///
/// What the criterion guarantees depends on definiteness and is stated for
/// the exactly symmetric matrix the solver works on — the input's upper
/// triangle mirrored (see [`symmetric_eigen_jacobi_with_tolerance`]); an
/// accepted asymmetry is a perturbation of that matrix, bounded by the
/// symmetry check. For a positive definite `A = D·H·D` (`D` the square root of
/// the diagonal), Jacobi stopped
/// this way computes every eigenvalue to relative error `O(n·ε·κ(H))` — high
/// *relative* accuracy even for eigenvalues far below `ε·‖A‖` (Demmel &
/// Veselić 1992, "Jacobi's method is more accurate than QR", *SIAM J. Matrix
/// Anal. Appl.* 13(4), Theorem 4.1 and §4). For an indefinite matrix that
/// guarantee does not hold: each remaining entry is at most
/// `ε·max(√(|a_pp a_qq|), ε·‖A‖_F) ≤ ε·‖A‖_F`, so the remainder has Frobenius
/// norm at most `n·ε·‖A‖_F` and every eigenvalue is accurate to that normwise
/// bound plus the rotations' own `O(n²)·ε·‖A‖_F` backward error — the accuracy
/// class of the QR algorithm.
#[inline]
fn default_tolerance<T: RealScalar>() -> T {
    machine_epsilon::<T>()
}

/// Largest asymmetry `|aᵢⱼ − aⱼᵢ|` accepted: `(n + 2)·ε·‖A‖_F`.
///
/// Derivation: a matrix symmetric in exact arithmetic is typically assembled
/// entry by entry as a length-`n` inner product of computed factors — `Q·D·Qᵀ`,
/// `XᵀX`, a Laplacian from weights — along a different rounding path for
/// `aᵢⱼ` than for `aⱼᵢ`. Each such entry carries error at most
/// `γ_{n+2}·Σₖ|terms|` (Higham 2002, §3.1: `n` additions and two products per
/// term, `γ_m ≈ m·u`, `u = ε/2`), and `Σₖ|terms| ≤ ‖A‖₂ ≤ ‖A‖_F` for the
/// orthogonal and Gram constructions (Cauchy–Schwarz over unit rows), so the
/// two paths differ by at most `2γ_{n+2}·‖A‖_F ≈ (n + 2)·ε·‖A‖_F`. An accepted
/// asymmetry `E` then has `‖E‖_F ≤ n·(n + 2)·ε·‖A‖_F`, the same order as the
/// `O(n²)·ε·‖A‖_F` backward error the rotations commit, so accepting it costs
/// no accuracy the solver had.
fn symmetry_bound<T: RealScalar>(norm: T, n: usize) -> T {
    T::from_usize(n + 2).mul(machine_epsilon::<T>()).mul(norm)
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
    mirror_upper_triangle(&mut a, n);
    let exponent = jacobi_gate_exponent(&a);
    scaling::scale_by_power_of_two(&mut a, -exponent);
    let mut target = NoEigenvectors;

    diagonalize(&mut a, n, tolerance, &mut target)?;
    let mut diagonal: Vec<T> = (0..n).map(|i| a[i * n + i]).collect();
    scaling::restore(
        &mut diagonal,
        exponent,
        "Jacobi eigensolver: an eigenvalue exceeds the scalar range",
    )?;
    for (i, value) in diagonal.into_iter().enumerate() {
        a[i * n + i] = value;
    }
    Ok(sort_diagonal(&a, n))
}

/// Compute the eigendecomposition of a real symmetric matrix with an explicit
/// relative tolerance.
///
/// Rotations continue until every off-diagonal entry satisfies
/// `|a_pq| ≤ τ·max(√(|a_pp|·|a_qq|), τ·‖A‖_F)`, `τ = tolerance`: negligible
/// against the diagonal pair it couples, floored at `τ²·‖A‖_F` where that pair
/// has vanished. The stopping point scales with `A`, so no magnitude stops
/// early. For positive definite `A` this gives eigenvalues to high relative
/// accuracy; otherwise to the normwise `O(n²)·ε·‖A‖_F` of the QR algorithm
/// (see the default tolerance's derivation). Symmetry acceptance is
/// independent of the tolerance: `|aᵢⱼ − aⱼᵢ| ≤ (n + 2)·ε·‖A‖_F`, the rounding
/// a matrix assembled symmetric in exact arithmetic can carry. Only the upper
/// triangle (diagonal included) is then used: it is mirrored onto the lower
/// before the first rotation, so an accepted asymmetric input is resolved by
/// its upper triangle. ([`symmetric_eigen_qr`](crate::symmetric_eigen_qr)
/// resolves by the lower triangle instead, the LAPACK `uplo = 'L'` convention.)
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
    mirror_upper_triangle(&mut a, n);
    let exponent = jacobi_gate_exponent(&a);
    scaling::scale_by_power_of_two(&mut a, -exponent);
    let mut v = identity::<T>(n);
    let mut target = EigenvectorWorkspace { values: &mut v };
    diagonalize(&mut a, n, tolerance, &mut target)?;
    let mut diagonal: Vec<T> = (0..n).map(|i| a[i * n + i]).collect();
    scaling::restore(
        &mut diagonal,
        exponent,
        "Jacobi eigensolver: an eigenvalue exceeds the scalar range",
    )?;
    for (i, value) in diagonal.into_iter().enumerate() {
        a[i * n + i] = value;
    }

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
    let bound = symmetry_bound(scaled_frobenius(a), n);
    for row in 0..n {
        for col in (row + 1)..n {
            if a[row * n + col].sub(a[col * n + row]).abs() > bound {
                return Err(LetoError::InvalidInput(format!(
                    "symmetric eigensolver input is not symmetric at ({row}, {col})"
                )));
            }
        }
    }
    Ok(())
}

/// The matrix-tier gate for Jacobi (degree 1, bound `2^(1+r)`), `0` when the
/// input is factored unscaled.
///
/// Every entry of every iterate of a symmetric `A` is at most
/// `‖A‖₂ ≤ ‖A‖_F ≤ 2^r·‖A‖_max` (orthogonal similarity; `r` from
/// [`scaling::norm_ratio_log2`]). The largest intermediates of `rotate` are
/// `2·a_pq` and `a_qq − a_pp` (the `atan2` arguments) and the partial sum
/// `c²·a_pp − 2sc·a_pq` of the diagonal update, each at most `2‖A‖₂`
/// (`c² + s² = 1`, `|2sc| ≤ 1`); the row updates `c·a_kp − s·a_kq` are at most
/// `√2·‖A‖₂`. So the gate's upper end is the overflow threshold divided by
/// `2^(1+r)` — no `ε/safmin` margin: a diagonal input such as
/// `diag(1e300, 1e-300)` or `f32` `diag(1e38, 1e-38)` has `r = 0` and is
/// factored unscaled, exactly. The lower end (`smlnum`) only ever scales up,
/// which is exact.
fn jacobi_gate_exponent<T: RealScalar>(a: &[T]) -> i32 {
    scaling::gate_exponent(a, 1, |values, largest| {
        1 + scaling::norm_ratio_log2(values, largest)
    })
    .unwrap_or(0)
}

/// Copy the strictly upper triangle onto the lower, so an accepted but
/// asymmetric input is resolved by its upper triangle alone. The rotations
/// read whole rows and columns; without this they would mix both triangles of
/// the first pairs they touch.
fn mirror_upper_triangle<T: Scalar>(a: &mut [T], n: usize) {
    for row in 0..n {
        for col in (row + 1)..n {
            a[col * n + row] = a[row * n + col];
        }
    }
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
    use super::{
        diagonalize_within, symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi, NoEigenvectors,
    };
    use leto::{Array2, LetoError};

    #[test]
    fn jacobi_typed_overflow_replaces_a_wrong_ok_near_max() {
        // M = 0.75·MAX: [[M, M], [M, M]] has true eigenvalues {0, 2M}. Since
        // M > MAX/2, `2M` itself is not representable in f64 — the correct
        // outcome is a typed `Overflow`, never a finite answer. Before
        // balancing (finding LETO-DENSE-SCALE-RANGE-2026-09-24, item J) the
        // unscaled rotation overflowed internally and returned the wrong
        // `Ok([M, M])` instead (neither eigenvalue is `M`).
        let m = 0.75 * f64::MAX;
        let a = Array2::from_shape_vec([2, 2], vec![m, m, m, m]).expect("2x2");
        assert!(matches!(
            symmetric_eigen_jacobi(&a.view()),
            Err(LetoError::Overflow { .. })
        ));
        // The eigenvalues-only path (`symmetric_eigenvalues_jacobi`) scales
        // independently of the full-decomposition path: this kills the
        // mutant that drops its own balancing call.
        assert!(matches!(
            symmetric_eigenvalues_jacobi(&a.view()),
            Err(LetoError::Overflow { .. })
        ));
    }

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
