//! The classical Jacobi iteration: the largest-unconverged pivot search and
//! the two-sided rotation, accumulated into an optional eigenvector target.

use crate::application::linalg::thresholds::scaled_frobenius;
use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

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

pub(super) trait RotationTarget<T: RealScalar> {
    fn rotate_columns(&mut self, n: usize, p: usize, q: usize, c: T, s: T);
}

pub(super) struct NoEigenvectors;

impl<T: RealScalar> RotationTarget<T> for NoEigenvectors {
    #[inline]
    fn rotate_columns(&mut self, _n: usize, _p: usize, _q: usize, _c: T, _s: T) {}
}

pub(super) struct EigenvectorWorkspace<'a, T> {
    pub(super) values: &'a mut [T],
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

pub(super) fn diagonalize<T, R>(a: &mut [T], n: usize, tolerance: T, target: &mut R) -> Result<()>
where
    T: RealScalar,
    R: RotationTarget<T>,
{
    diagonalize_within(a, n, tolerance, rotation_budget(n), target)
}

/// [`diagonalize`] with an explicit rotation budget.
pub(super) fn diagonalize_within<T, R>(
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
