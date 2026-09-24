//! Implicit-shift QL iteration on a symmetric tridiagonal matrix
//! (`tql2`, Bowdler, Martin, Reinsch & Wilkinson 1968).

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

/// `√(x² + y²)` without the overflow or underflow of squaring the larger
/// operand.
#[inline]
fn hypot<T: RealScalar>(x: T, y: T) -> T {
    let (x, y) = (x.abs(), y.abs());
    let (large, small) = if x >= y { (x, y) } else { (y, x) };
    if large == T::ZERO {
        return T::ZERO;
    }
    let ratio = small.div(large);
    large.mul(T::ONE.add(ratio.mul(ratio)).sqrt())
}

/// `scale + |value| == scale`: `value` is below the rounding of `scale`.
#[inline]
fn negligible<T: RealScalar>(value: T, scale: T) -> bool {
    scale.add(value.abs()) == scale
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
    let mut shift_total = T::ZERO;
    let mut sweeps = 0_usize;
    let two = T::from_usize(2);
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
    for l in 0..n {
        // `off_diagonal[n − 1] = 0` bounds the scan at `n − 1`.
        let mut m = l;
        while m + 1 < n && !negligible(off_diagonal[m], norm_estimate) {
            m += 1;
        }
        if m > l {
            loop {
                sweeps += 1;
                if sweeps > budget {
                    return Err(LetoError::ConvergenceError {
                        max_iters: budget,
                        residual: off_diagonal[l].abs().div(norm_estimate).to_f64(),
                        tol: machine_epsilon::<T>().to_f64() / 2.0,
                    });
                }
                // Wilkinson shift from the leading 2×2 of the active block.
                let g = diagonal[l];
                let mut p = diagonal[l + 1].sub(g).div(two.mul(off_diagonal[l]));
                let mut r = hypot(p, T::ONE);
                if p < T::ZERO {
                    r = r.neg();
                }
                diagonal[l] = off_diagonal[l].div(p.add(r));
                diagonal[l + 1] = off_diagonal[l].mul(p.add(r));
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
                p = s
                    .neg()
                    .mul(s2)
                    .mul(c3)
                    .mul(first_off)
                    .mul(off_diagonal[l])
                    .div(leading);
                off_diagonal[l] = s.mul(p);
                diagonal[l] = c.mul(p);
                if negligible(off_diagonal[l], norm_estimate) {
                    break;
                }
            }
        }
        diagonal[l] = diagonal[l].add(shift_total);
        off_diagonal[l] = T::ZERO;
    }
    sort_ascending(diagonal, vectors, n);
    Ok(())
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
