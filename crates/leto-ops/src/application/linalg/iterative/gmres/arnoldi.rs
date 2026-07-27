//! Arnoldi iteration (modified Gram-Schmidt) for GMRES.
//!
//! The Krylov basis is stored **row-contiguously**: basis vector `k` occupies
//! `basis[k·n .. (k+1)·n]`. Both the orthogonalisation sweep and the solution
//! update walk whole basis vectors, so this layout makes every inner loop a
//! unit-stride slice traversal. Storing the basis as columns of an `n × (m+1)`
//! row-major array instead would give each traversal a stride of `m+1`
//! elements — one cache line per element for the dominant `O(n·k)` work.

use super::super::traits::{LinearOperator, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, Result};

/// Result of one Arnoldi step.
pub(super) enum ArnoldiOutcome<T> {
    /// The basis was extended by one vector; the payload is `H[k+1, k]`.
    Extended(T),
    /// `H[k+1, k]` vanished relative to `‖M⁻¹·A·v_k‖`. The Krylov subspace is
    /// invariant under `M⁻¹A`, so the least-squares problem already attains its
    /// minimum (happy breakdown, Saad and Schultz §3.2) and no further basis
    /// vector exists.
    HappyBreakdown,
    /// The recurrence produced a non-finite value; the basis is unusable.
    NonFinite,
}

#[inline]
fn dot<T: NumericElement>(lhs: &[T], rhs: &[T]) -> T {
    let mut sum = T::ZERO;
    for (&l, &r) in lhs.iter().zip(rhs.iter()) {
        sum += l * r;
    }
    sum
}

#[inline]
fn norm<T: NumericElement>(v: &[T]) -> T {
    dot(v, v).sqrt()
}

/// Extend the Krylov basis by one vector and fill column `k` of the Hessenberg
/// matrix.
///
/// `hessenberg_column` receives `H[0..=k+1, k]` and must have length at least
/// `k + 2`. `basis` holds `m + 1` consecutive length-`n` vectors; on
/// [`ArnoldiOutcome::Extended`] the vector at index `k + 1` is written.
///
/// # Errors
/// Propagates operator and preconditioner failures. Numerical outcomes are
/// returned value-semantically through [`ArnoldiOutcome`].
#[allow(clippy::too_many_arguments)]
pub(super) fn arnoldi_step<T, Op, P>(
    operator: &Op,
    preconditioner: &P,
    basis: &mut [T],
    hessenberg_column: &mut [T],
    k: usize,
    n: usize,
    basis_work: &mut Array1<T>,
    work: &mut Array1<T>,
    precond_work: &mut Array1<T>,
) -> Result<ArnoldiOutcome<T>>
where
    T: RealField + Copy + FloatElement,
    Op: LinearOperator<T> + ?Sized,
    P: Preconditioner<T> + ?Sized,
{
    // The operator boundary is `Array1`-valued, so the current basis vector is
    // copied out of the flat buffer before the product is formed.
    for (slot, &value) in (0..n).zip(basis[k * n..(k + 1) * n].iter()) {
        basis_work[slot] = value;
    }
    operator.apply(basis_work, work)?;
    preconditioner.apply_to(work, precond_work)?;

    let w = super::flat_mut("preconditioner workspace", precond_work);

    // Reference scale for the breakdown test, taken before orthogonalisation.
    let reference_norm = norm(w);
    if !reference_norm.is_finite() {
        return Ok(ArnoldiOutcome::NonFinite);
    }

    // Modified Gram-Schmidt: each projection is subtracted before the next
    // coefficient is formed, which keeps the loss of orthogonality first-order
    // in the machine epsilon, where the classical variant is second-order.
    for j in 0..=k {
        let basis_j = &basis[j * n..(j + 1) * n];
        let coefficient = dot(w, basis_j);
        hessenberg_column[j] = coefficient;
        if !coefficient.is_finite() {
            return Ok(ArnoldiOutcome::NonFinite);
        }
        for (target, &value) in w.iter_mut().zip(basis_j.iter()) {
            *target -= coefficient * value;
        }
    }

    let new_norm = norm(w);
    hessenberg_column[k + 1] = new_norm;
    if !new_norm.is_finite() {
        return Ok(ArnoldiOutcome::NonFinite);
    }

    // Breakdown criterion, relative to the pre-orthogonalisation norm: the
    // subtractions above cancel at most `‖M⁻¹A v_k‖` of magnitude, so a
    // remainder at or below `ε·‖M⁻¹A v_k‖` carries no significant digits and
    // normalising it would seed the basis with rounding noise. An absolute
    // threshold, as used by both reference implementations, would instead make
    // the test scale-dependent.
    if new_norm <= <T as RealField>::EPSILON * reference_norm {
        return Ok(ArnoldiOutcome::HappyBreakdown);
    }

    let inverse_norm = <T as NumericElement>::ONE / new_norm;
    let (previous, next) = basis.split_at_mut((k + 1) * n);
    debug_assert!(previous.len() == (k + 1) * n, "invariant: basis is packed");
    for (target, &value) in next[..n].iter_mut().zip(w.iter()) {
        *target = value * inverse_norm;
    }
    Ok(ArnoldiOutcome::Extended(new_norm))
}
