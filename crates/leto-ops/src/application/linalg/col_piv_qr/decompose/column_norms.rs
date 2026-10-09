//! Cached squared column norms with LAPACK's downdating safeguard.
//!
//! The pivot rule and the partial-norm update follow LAPACK 3.12.0 `DLAQP2`
//! (<https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f>):
//! the pivot is the first maximum of the cached norms (`IDAMAX`, line 197), the
//! reliability threshold is `TOL3Z = √DLAMCH('Epsilon')` (line 187), and each
//! cached norm is downdated by the removed row or recomputed exactly when the
//! downdate is no longer trustworthy (lines 236-249; the analysis is LAPACK
//! Working Note 176).
//!
//! The state is kept as squared norms, `vn1² = current`, `vn2² = reference`.
//! `DLAQP2`'s test `(1 − (|a|/vn1)²)·(vn1/vn2)² ≤ TOL3Z` equals
//! `(vn1² − a²)/vn2² ≤ TOL3Z`, the downdated squared norm over the reference
//! squared norm, so no square root or per-column division by `vn1` is needed.
//!
//! The cached norms carry the rounding of the downdate. Columns whose exact tail
//! norms differ by less than the tolerance derived in ADR 0035 may therefore be
//! ordered either way, as in LAPACK.

use crate::application::linalg::thresholds::machine_epsilon;
use crate::domain::real::RealScalar;

/// Squared Euclidean norm of column `j` over rows `[r0 .. m)`.
pub(super) fn tail_norm_sq<T: RealScalar>(r: &[T], n: usize, m: usize, j: usize, r0: usize) -> T {
    let mut acc = T::ZERO;
    for i in r0..m {
        let x = r[i * n + j];
        acc = acc.add(x.mul(x));
    }
    acc
}

/// Per-column squared tail norms: the downdated value used as the pivot key and
/// the value at the last exact computation used as the reliability reference.
pub(super) struct ColumnNorms<T> {
    current_squared: Box<[T]>,
    reference_squared: Box<[T]>,
    recompute_threshold: T,
}

impl<T: RealScalar> ColumnNorms<T> {
    /// Exact squared norms of the `n` columns of the row-major `m × n` matrix `r`.
    pub(super) fn new(r: &[T], n: usize, m: usize) -> Self {
        let current_squared: Box<[T]> = (0..n)
            .map(|column| tail_norm_sq(r, n, m, column, 0))
            .collect();
        Self {
            reference_squared: current_squared.clone(),
            current_squared,
            // `TOL3Z` is `√DLAMCH('Epsilon')` (dlaqp2.f line 187), and
            // `DLAMCH('Epsilon')` is the unit roundoff, half the spacing above
            // one that `machine_epsilon` returns.
            recompute_threshold: machine_epsilon::<T>().div(T::from_count(2)).sqrt(),
        }
    }

    /// Largest cached squared norm: the exact squared norm of the largest column
    /// before any downdate.
    pub(super) fn largest_squared(&self) -> T {
        self.current_squared.iter().fold(
            T::ZERO,
            |largest, &norm| {
                if norm > largest {
                    norm
                } else {
                    largest
                }
            },
        )
    }

    /// Pivot among columns `first_column..`: the first maximum of the cached
    /// squared norms (dlaqp2.f line 197, `IDAMAX`). Squared norms are compared
    /// because rounding the root can merge keys that differ in the square.
    pub(super) fn pivot(&self, first_column: usize) -> usize {
        let mut best = first_column;
        let mut best_squared = self.current_squared[first_column];
        for column in (first_column + 1)..self.current_squared.len() {
            let squared = self.current_squared[column];
            if squared > best_squared {
                best_squared = squared;
                best = column;
            }
        }
        best
    }

    /// Exchange the state of two columns together with the columns themselves
    /// (dlaqp2.f lines 204-205).
    #[inline]
    pub(super) fn swap(&mut self, lhs: usize, rhs: usize) {
        self.current_squared.swap(lhs, rhs);
        self.reference_squared.swap(lhs, rhs);
    }

    /// Downdate the columns right of `row` after the reflector of step `row`
    /// has been applied: the removed entry is `r[row, column]` (dlaqp2.f lines
    /// 236-249).
    ///
    /// The estimate `current − a²` is clamped at zero. When it falls to
    /// `TOL3Z` of the reference the cached norm is recomputed from the rows
    /// `row + 1..m` and becomes the new reference. A cached zero stays zero
    /// (line 231).
    pub(super) fn remove_row(&mut self, r: &[T], n: usize, m: usize, row: usize) {
        for column in (row + 1)..n {
            let current_squared = self.current_squared[column];
            if current_squared == T::ZERO {
                continue;
            }
            let removed = r[row * n + column];
            let difference = current_squared.sub(removed.mul(removed));
            let estimate = if difference > T::ZERO {
                difference
            } else {
                T::ZERO
            };
            let reference_squared = self.reference_squared[column];
            if estimate.div(reference_squared) <= self.recompute_threshold {
                let recomputed = tail_norm_sq(r, n, m, column, row + 1);
                self.current_squared[column] = recomputed;
                self.reference_squared[column] = recomputed;
            } else {
                self.current_squared[column] = estimate;
            }
        }
    }
}

#[cfg(test)]
#[path = "column_norms/tests.rs"]
mod tests;
