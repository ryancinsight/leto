//! Implicit-shift QL iteration on a symmetric tridiagonal matrix
//! (`tql2`, Bowdler, Martin, Reinsch & Wilkinson 1968).

use crate::application::linalg::scaling::hypot;
use crate::application::linalg::thresholds::machine_epsilon;
use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// QL sweeps allowed per eigenvalue before declaring non-convergence.
///
/// The Wilkinson-shifted iteration converges globally and, in practice,
/// cubically, taking under two sweeps per eigenvalue; LAPACK `dsteqr` bounds
/// the total at `30·n` sweeps, the budget adopted here.
const SWEEPS_PER_EIGENVALUE: usize = 30;

/// The sweep budget for an order-`n` tridiagonal: `30·n`.
pub(super) fn sweep_budget(n: usize) -> usize {
    SWEEPS_PER_EIGENVALUE.saturating_mul(n)
}

/// `scale + |value| == scale`: `value` is below the rounding of `scale`.
#[inline]
fn negligible<T: RealScalar>(value: T, scale: T, absolute: Option<T>) -> bool {
    absolute.map_or_else(
        || scale.add(value.abs()) == scale,
        |limit| value.abs() <= limit,
    )
}

/// Completed shifts and a bound on the discarded off-diagonal Frobenius norm.
pub(crate) struct IterationResult<T> {
    pub(crate) iterations: usize,
    pub(crate) discarded_norm: T,
}

/// The same QL kernel with an absolute Frobenius convergence contract.
/// Each of at most `n - 1` deflated edges is bounded by `tolerance / (2n)`.
/// Removing an edge and its symmetric partner costs `sqrt(2) * |edge|`;
/// the triangle inequality bounds the total perturbation by `tolerance`.
pub(crate) fn diagonalize_absolute<T: RealScalar>(
    diagonal: &mut [T],
    off_diagonal: &mut [T],
    vectors: &mut [T],
    n: usize,
    budget: usize,
    tolerance: T,
) -> Result<IterationResult<T>> {
    diagonalize_with_policy(diagonal, off_diagonal, vectors, n, budget, Some(tolerance))
}

/// Rotate rows `i` and `i + 1` of the row-major `_ × n` matrix `rows`:
/// `row_{i+1} ← s·row_i + c·row_{i+1}`, `row_i ← c·row_i − s·row_{i+1}`.
///
/// This is the column rotation `Z ← Z G` of `tql2` applied to `Zᵀ`, which
/// turns the stride-`n` column walk into two contiguous rows.
#[inline]
fn rotate_adjacent_rows<T: RealScalar>(rows: &mut [T], n: usize, i: usize, c: T, s: T) {
    let (upper, lower) = rows[i * n..(i + 2) * n].split_at_mut(n);
    for (a, b) in upper.iter_mut().zip(lower.iter_mut()) {
        let (va, vb) = (*a, *b);
        *b = s.mul(va).add(c.mul(vb));
        *a = c.mul(va).sub(s.mul(vb));
    }
}

/// Diagonalize the tridiagonal `(diagonal, off_diagonal)` in place,
/// accumulating the rotations into the rows of `vectors` (`n × n`, row-major,
/// holding `Qᵀ` on entry and the eigenvectors as rows on exit), then sort the
/// eigenpairs by ascending eigenvalue.
///
/// `off_diagonal[k] = T[k, k+1]` with `off_diagonal[n−1] = 0`.
///
/// The norm estimate `t` is fixed before the first sweep (see the body).
///
/// # Errors
///
/// [`LetoError::ConvergenceError`] when `budget` sweeps do not deflate the
/// matrix: `residual` is the undeflated `|eₗ|` relative to the norm estimate
/// `t`, and `tol` is `ε/2`, the relative size below which `t + |eₗ|` rounds
/// to `t`.
pub(super) fn diagonalize<T: RealScalar>(
    diagonal: &mut [T],
    off_diagonal: &mut [T],
    vectors: &mut [T],
    n: usize,
    budget: usize,
) -> Result<()> {
    diagonalize_with_policy(diagonal, off_diagonal, vectors, n, budget, None).map(|_| ())
}

fn diagonalize_with_policy<T: RealScalar>(
    diagonal: &mut [T],
    off_diagonal: &mut [T],
    vectors: &mut [T],
    n: usize,
    budget: usize,
    absolute_tolerance: Option<T>,
) -> Result<IterationResult<T>> {
    let mut shift_total = T::ZERO;
    let mut sweeps = 0_usize;
    let two = T::from_count(2);
    let threshold = absolute_tolerance.map(|tol| tol.div(two.mul(T::from_count(n.max(1)))));
    let mut discarded_norm = T::ZERO;
    // `tql2` grows `t` as `l` advances; the full `t = maxᵢ(|dᵢ| + |eᵢ|)` from
    // the start keeps deflation normwise over the whole matrix, which is what
    // the backward-error bound assumes, and bounds the shift ratio below: a
    // surviving `|eₗ| ≥ ε·t/2` gives `|p| = |d_{l+1} − dₗ|/(2|eₗ|) ≤ 4/ε`
    // (`4096` in `F16`), where a running `t` taken over a tiny leading
    // diagonal lets `p` overflow the narrow formats.
    let norm_estimate = diagonal
        .iter()
        .zip(off_diagonal.iter())
        .fold(T::ZERO, |acc, (&d, &e)| {
            let local = d.abs().add(e.abs());
            if local > acc {
                local
            } else {
                acc
            }
        });
    'eigenvalues: for l in 0..n {
        // `off_diagonal[n − 1] = 0` bounds the scan at `n − 1`.
        let mut m = l;
        while m + 1 < n && !negligible(off_diagonal[m], norm_estimate, threshold) {
            m += 1;
        }
        if m > l {
            if absolute_tolerance.is_some() {
                discarded_norm = discarded_norm.add(two.sqrt().mul(off_diagonal[m].abs()));
                off_diagonal[m] = T::ZERO;
            }
            loop {
                if sweeps == budget {
                    let remaining = absolute_tolerance.map(|_| {
                        let norm = off_diagonal[l..]
                            .iter()
                            .fold(T::ZERO, |acc, &value| hypot(acc, value));
                        discarded_norm.add(two.sqrt().mul(norm))
                    });
                    if let (Some(norm), Some(tolerance)) = (remaining, absolute_tolerance) {
                        // The per-edge threshold is sufficient but not necessary.
                        // Honor a satisfied global contract on the final allowed
                        // step, restoring the pending common shift before exit.
                        if norm <= tolerance {
                            for d in &mut diagonal[l..] {
                                *d = d.add(shift_total);
                            }
                            off_diagonal[l..].fill(T::ZERO);
                            discarded_norm = norm;
                            break 'eigenvalues;
                        }
                    }
                    return Err(LetoError::ConvergenceError {
                        max_iters: budget,
                        residual: remaining
                            .unwrap_or_else(|| off_diagonal[l].abs().div(norm_estimate))
                            .to_f64(),
                        tol: absolute_tolerance.map_or_else(
                            || machine_epsilon::<T>().to_f64() / 2.0,
                            |tol| tol.to_f64(),
                        ),
                    });
                }
                sweeps += 1;
                // Wilkinson shift from the leading 2×2 of the active block.
                let g = diagonal[l];
                let difference = diagonal[l + 1].sub(g);
                let mut p = difference.div(two.mul(off_diagonal[l]));
                // The absolute policy may retain edges far below epsilon times
                // the matrix norm. An overflowing ratio means a vanishing shift;
                // forming e * infinity would lose the finite other root.
                let mut r = if absolute_tolerance.is_some() && !p.is_finite() {
                    T::ZERO
                } else {
                    hypot(p, T::ONE)
                };
                if p < T::ZERO {
                    r = r.neg();
                }
                diagonal[l] = off_diagonal[l].div(p.add(r));
                diagonal[l + 1] = if absolute_tolerance.is_some() {
                    difference.add(diagonal[l])
                } else {
                    off_diagonal[l].mul(p.add(r))
                };
                let leading = diagonal[l + 1];
                let mut h = g.sub(diagonal[l]);
                for d in &mut diagonal[l + 2..] {
                    *d = d.sub(h);
                }
                shift_total = shift_total.add(h);

                // Implicit QL sweep from the bottom of the block up to `l`.
                p = diagonal[m];
                let (mut c, mut c2, mut c3) = (T::ONE, T::ONE, T::ONE);
                let first_off = off_diagonal[l + 1];
                let (mut s, mut s2) = (T::ZERO, T::ZERO);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c.mul(off_diagonal[i]);
                    h = c.mul(p);
                    r = hypot(p, off_diagonal[i]);
                    off_diagonal[i + 1] = s.mul(r);
                    s = off_diagonal[i].div(r);
                    c = p.div(r);
                    p = c.mul(diagonal[i]).sub(s.mul(g));
                    diagonal[i + 1] = h.add(s.mul(c.mul(g).add(s.mul(diagonal[i]))));
                    rotate_adjacent_rows(vectors, n, i, c, s);
                }
                let coefficient = s.neg().mul(s2).mul(c3);
                p = if absolute_tolerance.is_some() {
                    // |e_l / leading| <= 1 by the Wilkinson root selection.
                    // Divide first: e_1 * e_l may underflow in a tiny block
                    // even when the global matrix needs no range scaling.
                    coefficient.mul(first_off).mul(off_diagonal[l].div(leading))
                } else {
                    coefficient.mul(first_off).mul(off_diagonal[l]).div(leading)
                };
                off_diagonal[l] = s.mul(p);
                diagonal[l] = c.mul(p);
                if negligible(off_diagonal[l], norm_estimate, threshold) {
                    break;
                }
            }
        }
        diagonal[l] = diagonal[l].add(shift_total);
        if absolute_tolerance.is_some() {
            discarded_norm = discarded_norm.add(two.sqrt().mul(off_diagonal[l].abs()));
        }
        off_diagonal[l] = T::ZERO;
    }
    sort_ascending(diagonal, vectors, n);
    crate::application::linalg::transformation_counts::record_ql_sweeps(sweeps);
    Ok(IterationResult {
        iterations: sweeps,
        discarded_norm,
    })
}

/// Selection-sort the eigenvalues ascending, swapping eigenvector rows along.
fn sort_ascending<T: RealScalar>(diagonal: &mut [T], vectors: &mut [T], n: usize) {
    for i in 0..n {
        let mut smallest = i;
        for j in i + 1..n {
            if diagonal[j] < diagonal[smallest] {
                smallest = j;
            }
        }
        if smallest != i {
            diagonal.swap(i, smallest);
            let (upper, lower) = vectors.split_at_mut(smallest * n);
            upper[i * n..(i + 1) * n].swap_with_slice(&mut lower[..n]);
        }
    }
}
