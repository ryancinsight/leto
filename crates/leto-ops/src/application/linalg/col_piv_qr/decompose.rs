//! Householder QR with column pivoting: `A P = Q R`.

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::application::linalg::thresholds::{machine_epsilon, rank_pivot_ratio};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Outcome of the pivoted QR: orthogonal `Q` (row-major `m×m`), upper-triangular
/// `R` (`m×n`), the column permutation (`perm[k]` = original column now at
/// position `k`), the numerical rank, and the dimensions.
pub(super) struct Factored<T> {
    pub(super) q: Vec<T>,
    pub(super) r: Vec<T>,
    pub(super) perm: Vec<usize>,
    pub(super) rank: usize,
    pub(super) m: usize,
    pub(super) n: usize,
}

/// Squared Euclidean norm of column `j` over rows `[r0 .. m)`.
fn tail_norm_sq<T: RealScalar>(r: &[T], n: usize, m: usize, j: usize, r0: usize) -> T {
    let mut acc = T::ZERO;
    for i in r0..m {
        let x = r[i * n + j];
        acc = acc.add(x.mul(x));
    }
    acc
}

trait ColumnNorms<T: RealScalar> {
    fn new(r: &[T], n: usize, m: usize) -> Self;

    fn squared_norm(&self, r: &[T], n: usize, m: usize, column: usize, first_row: usize) -> T;

    fn swap(&mut self, lhs: usize, rhs: usize);

    fn remove_row(&mut self, r: &[T], n: usize, m: usize, row: usize);
}

struct PartialColumnNorms<T> {
    current_squared: Box<[T]>,
    reference_squared: Box<[T]>,
    recompute_threshold: T,
}

impl<T: RealScalar> ColumnNorms<T> for PartialColumnNorms<T> {
    fn new(r: &[T], n: usize, m: usize) -> Self {
        let current_squared: Box<[T]> = (0..n)
            .map(|column| tail_norm_sq(r, n, m, column, 0))
            .collect();
        let reference_squared = current_squared.clone();
        Self {
            current_squared,
            reference_squared,
            // LAPACK 3.12.0 DLAQP2 sets TOL3Z to √(DLAMCH('Epsilon')).
            // This crate's machine_epsilon is the spacing above one, while
            // DLAMCH('Epsilon') is the unit roundoff (half that spacing).
            // https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f#L1067-L1069
            recompute_threshold: machine_epsilon::<T>().div(T::from_usize(2)).sqrt(),
        }
    }

    #[inline]
    fn squared_norm(&self, _r: &[T], _n: usize, _m: usize, column: usize, _first_row: usize) -> T {
        self.current_squared[column]
    }

    #[inline]
    fn swap(&mut self, lhs: usize, rhs: usize) {
        self.current_squared.swap(lhs, rhs);
        self.reference_squared.swap(lhs, rhs);
    }

    fn remove_row(&mut self, r: &[T], n: usize, m: usize, row: usize) {
        for column in (row + 1)..n {
            let current_squared = self.current_squared[column];
            if current_squared == T::ZERO {
                continue;
            }

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
            // https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f#L1164-L1190
            let reliability = estimate.div(self.reference_squared[column]);
            if reliability <= self.recompute_threshold {
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
struct RecomputedColumnNorms;

#[cfg(test)]
impl<T: RealScalar> ColumnNorms<T> for RecomputedColumnNorms {
    fn new(_r: &[T], _n: usize, _m: usize) -> Self {
        Self
    }

    fn squared_norm(&self, r: &[T], n: usize, m: usize, column: usize, first_row: usize) -> T {
        tail_norm_sq(r, n, m, column, first_row)
    }

    fn swap(&mut self, _lhs: usize, _rhs: usize) {}

    fn remove_row(&mut self, _r: &[T], _n: usize, _m: usize, _row: usize) {}
}

/// Factor `A` (m×n) with column pivoting.
///
/// At step `k` the column of largest remaining (rows `k..m`) norm is pivoted to
/// position `k`, then a Householder reflector zeroes the sub-column below the
/// diagonal. Pivoting makes `|R₀₀| ≥ |R₁₁| ≥ …`, so the first diagonal entry
/// that drops below a relative threshold reveals the rank.
pub(super) fn factor<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Factored<T>> {
    factor_with_norms::<T, PartialColumnNorms<T>>(matrix)
}

fn factor_with_norms<T: RealScalar, N: ColumnNorms<T>>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Factored<T>> {
    let [m, n] = matrix.shape();

    let mut r = if let Some(slice) = matrix.as_slice() {
        slice.to_vec()
    } else {
        matrix.to_contiguous().into_storage().into_inner()
    };
    for &value in &r {
        if !value.is_finite() {
            return Err(LetoError::StorageError {
                reason: "ColPivQR input contains a non-finite value".to_string(),
            });
        }
    }

    // Q ← Iₘ.
    let mut q = vec![T::ZERO; m * m];
    for i in 0..m {
        q[i * m + i] = T::ONE;
    }
    let mut perm: Vec<usize> = (0..n).collect();

    let p = m.min(n);
    let mut norms = N::new(&r, n, m);
    // Relative threshold from the largest initial full column norm.
    let mut ref_norm_sq = T::ZERO;
    for j in 0..n {
        let norm_squared = norms.squared_norm(&r, n, m, j, 0);
        if norm_squared > ref_norm_sq {
            ref_norm_sq = norm_squared;
        }
    }
    let tol = ref_norm_sq.sqrt().mul(rank_pivot_ratio::<T>());
    let mut rank = p;

    let mut alw: Vec<T> = Vec::with_capacity(n);
    for k in 0..p {
        // Pivot: column with the largest tail norm among k..n.
        let mut best = k;
        let mut best_norm_sq = norms.squared_norm(&r, n, m, k, k);
        for j in (k + 1)..n {
            let norm_squared = norms.squared_norm(&r, n, m, j, k);
            if norm_squared > best_norm_sq {
                best_norm_sq = norm_squared;
                best = j;
            }
        }
        if best_norm_sq.sqrt() <= tol {
            rank = k;
            break;
        }
        if best != k {
            for i in 0..m {
                r.swap(i * n + k, i * n + best);
            }
            perm.swap(k, best);
            norms.swap(k, best);
        }

        // Householder on column k, rows k..m.
        let len = m - k;
        let mut col_stack = [T::ZERO; 128];
        let mut col_vec = Vec::new();
        let col = if len <= 128 {
            for i in 0..len {
                col_stack[i] = r[(k + i) * n + k];
            }
            &col_stack[..len]
        } else {
            col_vec.reserve_exact(len);
            for i in 0..len {
                col_vec.push(r[(k + i) * n + k]);
            }
            &col_vec[..]
        };

        if let Some((refl, _alpha)) = reflector(col) {
            apply_left(&refl, &mut r, n, k, k, n, &mut alw); // rows k..m, cols k..n
            apply_right(&refl, &mut q, m, k, 0, m); // Q ← Q Hₖ
        }
        norms.remove_row(&r, n, m, k);
    }

    // Present the exact upper-triangular R (zero the reflector tails below the diagonal).
    for i in 1..m {
        for j in 0..i.min(n) {
            r[i * n + j] = T::ZERO;
        }
    }

    Ok(Factored {
        q,
        r,
        perm,
        rank,
        m,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        factor, factor_with_norms, machine_epsilon, tail_norm_sq, ColumnNorms, PartialColumnNorms,
        RecomputedColumnNorms,
    };
    use crate::domain::real::RealScalar;
    use leto::Array2;

    fn assert_matches_recomputed<T: RealScalar>(
        case: &str,
        rows: usize,
        columns: usize,
        values: &[f64],
        permutation_must_match: bool,
    ) {
        let matrix = Array2::from_shape_vec(
            [rows, columns],
            values.iter().copied().map(T::from_f64).collect(),
        )
        .expect("invariant: fixture count matches its matrix shape");
        let downdated = factor(&matrix.view()).expect("finite fixture must factor");
        let recomputed = factor_with_norms::<T, RecomputedColumnNorms>(&matrix.view())
            .expect("finite fixture must factor");
        if permutation_must_match {
            assert_eq!(
                downdated.perm,
                recomputed.perm,
                "{case} permutation for {}",
                core::any::type_name::<T>()
            );
        }
        assert_eq!(
            downdated.rank,
            recomputed.rank,
            "{case} rank for {}",
            core::any::type_name::<T>()
        );
    }

    fn existing_contract_fixtures_match_recomputed<T: RealScalar>() {
        assert_matches_recomputed::<T>(
            "reconstruction",
            4,
            3,
            &[
                4.0, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
            ],
            true,
        );
        assert_matches_recomputed::<T>(
            "least squares",
            4,
            2,
            &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
            true,
        );
        // Equal residual norms permit either tied pivot; rank is the contract.
        assert_matches_recomputed::<T>(
            "rank deficiency",
            4,
            3,
            &[1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0],
            false,
        );
    }

    #[test]
    fn downdated_pivots_preserve_existing_contract_across_scalar_types() {
        use eunomia::{Bf16, F16};

        existing_contract_fixtures_match_recomputed::<f64>();
        existing_contract_fixtures_match_recomputed::<f32>();
        existing_contract_fixtures_match_recomputed::<F16>();
        existing_contract_fixtures_match_recomputed::<Bf16>();
    }

    #[test]
    fn partial_norm_downdates_recompute_cancellation_boundary() {
        let delta = f64::EPSILON.sqrt() / 2.0;
        let r = vec![
            2.0,
            1.0,
            1.0, // Removed row.
            0.0,
            delta,
            0.0, // First remaining tail.
            0.0,
            0.0,
            2.0 * delta,
        ];
        let mut partial = PartialColumnNorms::new(&r, 3, 3);

        partial.remove_row(&r, 3, 3, 0);

        for column in 1..3 {
            assert_eq!(
                partial.current_squared[column],
                tail_norm_sq(&r, 3, 3, column, 1),
                "boundary trailing squared norm for column {column}"
            );
        }
    }

    fn squared_norm_order_survives_a_rounded_root_tie<T: RealScalar>() {
        let delta = machine_epsilon::<T>().sqrt();
        let matrix = Array2::from_shape_vec([2, 2], vec![T::ONE, T::ONE, T::ZERO, delta])
            .expect("invariant: fixture count matches its matrix shape");

        assert_eq!(
            factor(&matrix.view())
                .expect("finite fixture must factor")
                .perm,
            [1, 0],
            "squared column norms remain distinguishable for {}",
            core::any::type_name::<T>()
        );
    }

    #[test]
    fn pivot_order_uses_squared_norms_across_scalar_types() {
        use eunomia::{Bf16, F16};

        squared_norm_order_survives_a_rounded_root_tie::<f64>();
        squared_norm_order_survives_a_rounded_root_tie::<f32>();
        squared_norm_order_survives_a_rounded_root_tie::<F16>();
        squared_norm_order_survives_a_rounded_root_tie::<Bf16>();
    }
}
