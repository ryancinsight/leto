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

#[inline]
fn partial_norm_factor<T: RealScalar>(ratio: T) -> T {
    let factor = T::ONE.sub(ratio.mul(ratio));
    if factor > T::ZERO {
        factor
    } else {
        T::ZERO
    }
}

trait ColumnNorms<T: RealScalar> {
    fn new(r: &[T], n: usize, m: usize) -> Self;

    fn value(&self, r: &[T], n: usize, m: usize, column: usize, first_row: usize) -> T;

    fn swap(&mut self, lhs: usize, rhs: usize);

    fn remove_row(&mut self, r: &[T], n: usize, m: usize, row: usize);
}

struct PartialColumnNorms<T> {
    current: Box<[T]>,
    reference: Box<[T]>,
    recompute_threshold: T,
}

impl<T: RealScalar> ColumnNorms<T> for PartialColumnNorms<T> {
    fn new(r: &[T], n: usize, m: usize) -> Self {
        let current: Box<[T]> = (0..n)
            .map(|column| tail_norm_sq(r, n, m, column, 0).sqrt())
            .collect();
        let reference = current.clone();
        Self {
            current,
            reference,
            // LAPACK 3.12.0 DLAQP2 sets TOL3Z to √(DLAMCH('Epsilon')).
            // This crate's machine_epsilon is the spacing above one, while
            // DLAMCH('Epsilon') is the unit roundoff (half that spacing).
            // https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f#L1067-L1069
            recompute_threshold: machine_epsilon::<T>().div(T::from_usize(2)).sqrt(),
        }
    }

    #[inline]
    fn value(&self, _r: &[T], _n: usize, _m: usize, column: usize, _first_row: usize) -> T {
        self.current[column]
    }

    #[inline]
    fn swap(&mut self, lhs: usize, rhs: usize) {
        self.current.swap(lhs, rhs);
        self.reference.swap(lhs, rhs);
    }

    fn remove_row(&mut self, r: &[T], n: usize, m: usize, row: usize) {
        for column in (row + 1)..n {
            let current = self.current[column];
            if current == T::ZERO {
                continue;
            }

            // LAPACK 3.12.0 DLAQP2 lines 227–248: downdate the partial
            // column norm after removing this row, but recompute the tail
            // exactly when cancellation makes the estimate unreliable.
            // https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlaqp2.f#L227-L248
            let ratio = r[row * n + column].abs().div(current);
            // DLAQP2 computes 1 - (|a| / norm)^2 directly.  Keeping the
            // square as one operation avoids the extra rounding from the
            // algebraically equivalent (1 + ratio) * (1 - ratio) form.
            let estimate = partial_norm_factor(ratio);
            let relative = current.div(self.reference[column]);
            let reliability = estimate.mul(relative.mul(relative));
            if reliability <= self.recompute_threshold {
                let recomputed = tail_norm_sq(r, n, m, column, row + 1).sqrt();
                self.current[column] = recomputed;
                self.reference[column] = recomputed;
            } else {
                self.current[column] = current.mul(estimate.sqrt());
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

    fn value(&self, r: &[T], n: usize, m: usize, column: usize, first_row: usize) -> T {
        tail_norm_sq(r, n, m, column, first_row).sqrt()
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
    let mut ref_norm = T::ZERO;
    for j in 0..n {
        let nrm = norms.value(&r, n, m, j, 0);
        if nrm > ref_norm {
            ref_norm = nrm;
        }
    }
    let tol = ref_norm.mul(rank_pivot_ratio::<T>());
    let mut rank = p;

    let mut alw: Vec<T> = Vec::with_capacity(n);
    for k in 0..p {
        // Pivot: column with the largest tail norm among k..n.
        let mut best = k;
        let mut best_norm = norms.value(&r, n, m, k, k);
        for j in (k + 1)..n {
            let nrm = norms.value(&r, n, m, j, k);
            if nrm > best_norm {
                best_norm = nrm;
                best = j;
            }
        }
        if best_norm <= tol {
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
        factor, factor_with_norms, machine_epsilon, partial_norm_factor, RecomputedColumnNorms,
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
        // After the sum column pivots first, the other two residual columns
        // are negatives with equal exact norms. Either tied column may pivot
        // next; rank is the value-semantic contract for this fixture.
        assert_matches_recomputed::<T>(
            "rank deficiency",
            4,
            3,
            &[1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0],
            false,
        );
    }

    #[test]
    fn downdated_pivots_match_recomputed_norms_across_scalar_types() {
        use eunomia::{Bf16, F16};

        existing_contract_fixtures_match_recomputed::<f64>();
        existing_contract_fixtures_match_recomputed::<f32>();
        existing_contract_fixtures_match_recomputed::<F16>();
        existing_contract_fixtures_match_recomputed::<Bf16>();
    }

    #[test]
    fn partial_norm_update_matches_lapack_rounding_contract() {
        let ratio = 0.75_f64;
        assert_eq!(partial_norm_factor(ratio), 1.0 - ratio * ratio);
        let expected_threshold = (f64::EPSILON / 2.0).sqrt();
        let actual_threshold = (machine_epsilon::<f64>() / 2.0).sqrt();
        assert_eq!(actual_threshold, expected_threshold);
    }

    #[test]
    fn partial_norm_downdates_match_recomputed_tails_at_boundary() {
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
            2.0 * delta, // Second remaining tail.
        ];
        let mut partial = PartialColumnNorms::new(&r, 3, 3);

        partial.remove_row(&r, 3, 3, 0);

        for column in 1..3 {
            let expected = tail_norm_sq(&r, 3, 3, column, 1).sqrt();
            assert_eq!(
                partial.current[column], expected,
                "boundary trailing norm for column {column}"
            );
        }
    }
}
