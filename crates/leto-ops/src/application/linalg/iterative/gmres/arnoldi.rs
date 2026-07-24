//! Arnoldi iteration (modified Gram-Schmidt) for GMRES.

use super::super::traits::{LinearOperator, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, Array2, LetoError, Result};

type KrylovBasis<T> = Array2<T>;
type Hessenberg<T> = Array2<T>;

#[inline]
fn basis_col<T: Copy>(v: &KrylovBasis<T>, n: usize, col: usize, out: &mut Array1<T>) {
    for row in 0..n {
        out[row] = v[[row, col]];
    }
}

#[inline]
fn vec_dot<T: RealField + Copy>(work: &Array1<T>, v: &KrylovBasis<T>, n: usize, col: usize) -> T {
    let mut s = <T as NumericElement>::ZERO;
    for row in 0..n {
        s += work[row] * v[[row, col]];
    }
    s
}

#[inline]
fn vec_norm<T: NumericElement>(work: &Array1<T>, n: usize) -> T {
    let mut s = T::ZERO;
    for i in 0..n {
        s += work[i] * work[i];
    }
    s.sqrt()
}

/// One Arnoldi step: extends the Krylov basis by one vector and fills the
/// corresponding column of the Hessenberg matrix.
///
/// # Returns
/// The 2-norm of the new (unnormalized) basis vector, used as `H[k+1, k]`.
///
/// # Errors
/// Returns [`LetoError::InvalidInput`] if preconditioner workspace is missing.
#[allow(clippy::too_many_arguments)]
pub fn arnoldi_step<T, Op, P>(
    a: &Op,
    v: &mut KrylovBasis<T>,
    h: &mut Hessenberg<T>,
    k: usize,
    n: usize,
    basis_work: &mut Array1<T>,
    work: &mut Array1<T>,
    precond: Option<&P>,
    precond_work: Option<&mut Array1<T>>,
) -> Result<T>
where
    T: RealField + Copy + FloatElement,
    Op: LinearOperator<T> + ?Sized,
    P: Preconditioner<T> + ?Sized,
{
    // Extract k-th basis column into basis_work.
    basis_col(v, n, k, basis_work);

    // work ← A · v_k
    a.apply(basis_work, work)?;

    // Left preconditioning (optional): work ← M⁻¹ · work
    if let Some(prec) = precond {
        if let Some(pw) = precond_work {
            prec.apply_to(work, pw)?;
            for i in 0..n {
                work[i] = pw[i];
            }
        } else {
            return Err(LetoError::InvalidInput(
                "Arnoldi: preconditioner workspace required but not provided".into(),
            ));
        }
    }

    // Modified Gram-Schmidt orthogonalisation.
    for j in 0..=k {
        let h_jk = vec_dot(work, v, n, j);
        h[[j, k]] = h_jk;
        for i in 0..n {
            work[i] -= h_jk * v[[i, j]];
        }
    }

    let new_norm = vec_norm(work, n);
    h[[k + 1, k]] = new_norm;

    if new_norm > <T as RealField>::EPSILON {
        let inv_norm = <T as NumericElement>::ONE / new_norm;
        for i in 0..n {
            v[[i, k + 1]] = work[i] * inv_norm;
        }
    }
    Ok(new_norm)
}
