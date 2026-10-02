//! Cached squared column norms and conservative pivot intervals.
//!
//! The partial-norm downdate and its exact-recompute safeguard follow LAPACK
//! 3.12.0 `DLAQP2` (`TOL3Z` at line 187, the update at lines 236-249 of
//! <https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f>),
//! which cites LAPACK Working Note 176 for the update. The state is
//! kept as squared norms, so `DLAQP2`'s test
//! `(1 - (|a|/vn1)²)·(vn1/vn2)² ≤ TOL3Z` reduces to the estimated remaining
//! squared norm over the reference squared norm.

use super::norm_bounds::{Arithmetic, TailBounds};
use crate::application::linalg::householder::Reflector;
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

pub(super) trait ColumnNorms<T: RealScalar> {
    fn new(r: &[T], n: usize, m: usize) -> Self;

    fn squared_norm(&self, r: &[T], n: usize, m: usize, column: usize, first_row: usize) -> T;

    fn refresh_remaining(&mut self, r: &[T], n: usize, m: usize, first_row: usize);

    fn certified_pivot(&self, first_column: usize, n: usize, tail_len: usize) -> Option<usize>;

    fn swap(&mut self, lhs: usize, rhs: usize);

    fn remove_row(
        &mut self,
        r: &[T],
        n: usize,
        m: usize,
        row: usize,
        reflector: Option<&Reflector<T>>,
    );

    #[cfg(test)]
    fn assert_bounds_cover_exact_keys(&self, r: &[T], n: usize, m: usize, first_row: usize);
}

pub(super) struct PartialColumnNorms<T> {
    pub(super) current_squared: Box<[T]>,
    reference_squared: Box<[T]>,
    bounds: Box<[Option<TailBounds<T>>]>,
    arithmetic: Option<Arithmetic<T>>,
    recompute_threshold: T,
}

impl<T: RealScalar> ColumnNorms<T> for PartialColumnNorms<T> {
    fn new(r: &[T], n: usize, m: usize) -> Self {
        let current_squared: Box<[T]> = (0..n)
            .map(|column| tail_norm_sq(r, n, m, column, 0))
            .collect();
        let reference_squared = current_squared.clone();
        let arithmetic = Arithmetic::<T>::new();
        let bounds: Box<[Option<TailBounds<T>>]> = current_squared
            .iter()
            .map(|&norm| arithmetic.and_then(|ops| ops.enclose_squared_sum(norm, m)))
            .collect();
        Self {
            current_squared,
            reference_squared,
            bounds,
            arithmetic,
            // LAPACK 3.12.0 DLAQP2 sets TOL3Z to √(DLAMCH('Epsilon')).
            // This crate's machine_epsilon is the spacing above one, while
            // DLAMCH('Epsilon') is the unit roundoff (half that spacing).
            // https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f#L187
            recompute_threshold: machine_epsilon::<T>().div(T::from_count(2)).sqrt(),
        }
    }

    #[inline]
    fn squared_norm(&self, _r: &[T], _n: usize, _m: usize, column: usize, _first_row: usize) -> T {
        self.current_squared[column]
    }

    fn refresh_remaining(&mut self, r: &[T], n: usize, m: usize, first_row: usize) {
        for column in first_row..n {
            let recomputed = tail_norm_sq(r, n, m, column, first_row);
            self.current_squared[column] = recomputed;
            self.reference_squared[column] = recomputed;
            self.bounds[column] = self
                .arithmetic
                .and_then(|ops| ops.enclose_squared_sum(recomputed, m - first_row));
        }
    }

    fn certified_pivot(&self, first_column: usize, n: usize, tail_len: usize) -> Option<usize> {
        if first_column + 1 == n {
            return Some(first_column);
        }
        let arithmetic = self.arithmetic?;
        let mut best_column = first_column;
        let mut best_lower = T::ZERO;
        let mut largest_upper_column = first_column;
        let mut largest_upper = T::ZERO;
        let mut second_upper = T::ZERO;

        for column in first_column..n {
            let norm = self.bounds[column]?;
            let key = arithmetic.pivot_key(norm, tail_len)?;
            if column == first_column || key.lower > best_lower {
                best_lower = key.lower;
                best_column = column;
            }
            if column == first_column || key.upper > largest_upper {
                second_upper = largest_upper;
                largest_upper = key.upper;
                largest_upper_column = column;
            } else if key.upper > second_upper {
                second_upper = key.upper;
            }
        }

        (best_column == largest_upper_column && best_lower > second_upper).then_some(best_column)
    }

    #[inline]
    fn swap(&mut self, lhs: usize, rhs: usize) {
        self.current_squared.swap(lhs, rhs);
        self.reference_squared.swap(lhs, rhs);
        self.bounds.swap(lhs, rhs);
    }

    fn remove_row(
        &mut self,
        r: &[T],
        n: usize,
        m: usize,
        row: usize,
        reflector: Option<&Reflector<T>>,
    ) {
        let mut reflector_error = None;
        let mut reflector_error_computed = false;
        for column in (row + 1)..n {
            let current_squared = self.current_squared[column];
            // Keep the squared sum as the pivot key. Taking its square root
            // first can round distinct column tails to the same norm and
            // change the pivot selected by the exact-recompute algorithm.
            let removed_squared = r[row * n + column].mul(r[row * n + column]);
            let difference = current_squared.sub(removed_squared);
            let estimate = if difference > T::ZERO {
                difference
            } else {
                T::ZERO
            };

            // DLAQP2's cancellation test is
            // (1 - (|a|/current_norm)^2) * (current_norm/reference_norm)^2.
            // In squared-norm state this reduces algebraically to the
            // estimated remaining squared norm over the reference squared
            // norm. Recompute exactly when that ratio falls below TOL3Z.
            // https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f#L236-L249
            let reference_squared = self.reference_squared[column];
            let reliability = if reference_squared > T::ZERO {
                estimate.div(reference_squared)
            } else {
                T::ZERO
            };
            if !estimate.is_finite()
                || !reference_squared.is_finite()
                || reliability <= self.recompute_threshold
            {
                let recomputed = tail_norm_sq(r, n, m, column, row + 1);
                self.current_squared[column] = recomputed;
                self.reference_squared[column] = recomputed;
                self.bounds[column] = self
                    .arithmetic
                    .and_then(|ops| ops.enclose_squared_sum(recomputed, m - row - 1));
            } else {
                self.current_squared[column] = estimate;
                if !reflector_error_computed {
                    reflector_error = self
                        .arithmetic
                        .zip(reflector)
                        .and_then(|(ops, reflector)| ops.reflector_error(reflector));
                    reflector_error_computed = true;
                }
                self.bounds[column] = match (self.arithmetic, self.bounds[column], reflector) {
                    (Some(ops), Some(bounds), Some(_)) => reflector_error.and_then(|effect| {
                        ops.after_reflector(bounds, effect, r[row * n + column])
                    }),
                    (Some(ops), Some(bounds), None) => {
                        ops.remove_entry(bounds, r[row * n + column])
                    }
                    _ => None,
                };
            }
        }
    }

    #[cfg(test)]
    fn assert_bounds_cover_exact_keys(&self, r: &[T], n: usize, m: usize, first_row: usize) {
        let Some(arithmetic) = self.arithmetic else {
            return;
        };
        for column in first_row..n {
            let Some(bounds) = self.bounds[column] else {
                continue;
            };
            let exact_key = tail_norm_sq(r, n, m, column, first_row);
            if exact_key.is_finite() {
                let key_bounds = arithmetic
                    .pivot_key(bounds, m - first_row)
                    .expect("finite norm bounds produce a finite pivot key");
                assert!(
                    key_bounds.contains(exact_key),
                    "column {column}: exact key {exact_key:?} outside [{:?}, {:?}]",
                    key_bounds.lower,
                    key_bounds.upper
                );
            }
        }
    }
}

#[cfg(test)]
pub(super) struct RecomputedColumnNorms;

#[cfg(test)]
impl<T: RealScalar> ColumnNorms<T> for RecomputedColumnNorms {
    fn new(_r: &[T], _n: usize, _m: usize) -> Self {
        Self
    }

    fn squared_norm(&self, r: &[T], n: usize, m: usize, column: usize, first_row: usize) -> T {
        tail_norm_sq(r, n, m, column, first_row)
    }

    fn refresh_remaining(&mut self, _r: &[T], _n: usize, _m: usize, _first_row: usize) {}

    fn certified_pivot(&self, _first_column: usize, _n: usize, _tail_len: usize) -> Option<usize> {
        None
    }

    fn swap(&mut self, _lhs: usize, _rhs: usize) {}

    fn remove_row(
        &mut self,
        _r: &[T],
        _n: usize,
        _m: usize,
        _row: usize,
        _reflector: Option<&Reflector<T>>,
    ) {
    }

    #[cfg(test)]
    fn assert_bounds_cover_exact_keys(&self, _r: &[T], _n: usize, _m: usize, _first_row: usize) {}
}

#[cfg(test)]
#[path = "column_norms/tests.rs"]
mod tests;
