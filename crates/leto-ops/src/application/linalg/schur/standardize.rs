//! Canonicalize the real Schur form: split every 2×2 diagonal block that has
//! **real** eigenvalues into two 1×1 blocks, leaving 2×2 blocks only for genuine
//! complex-conjugate pairs.

use crate::domain::real::RealScalar;

#[inline]
fn at<T: Copy>(h: &[T], i: usize, j: usize, n: usize) -> T {
    h[i * n + j]
}

/// Apply the rotation similarity `G = [[c, −s], [s, c]]` on indices `(p, p+1)`:
/// `T ← Gᵀ T G`, `Z ← Z G`.
fn apply_rotation<T: RealScalar>(h: &mut [T], z: &mut [T], p: usize, c: T, s: T, n: usize) {
    // Left: Gᵀ T over rows (p, p+1), all columns.
    for j in 0..n {
        let t0 = at(h, p, j, n);
        let t1 = at(h, p + 1, j, n);
        h[p * n + j] = c.mul(t0).add(s.mul(t1));
        h[(p + 1) * n + j] = s.neg().mul(t0).add(c.mul(t1));
    }
    // Right: T G over columns (p, p+1), all rows.
    for i in 0..n {
        let t0 = at(h, i, p, n);
        let t1 = at(h, i, p + 1, n);
        h[i * n + p] = c.mul(t0).add(s.mul(t1));
        h[i * n + (p + 1)] = s.neg().mul(t0).add(c.mul(t1));
    }
    // Accumulate Z G over columns (p, p+1), all rows.
    for i in 0..n {
        let z0 = at(z, i, p, n);
        let z1 = at(z, i, p + 1, n);
        z[i * n + p] = c.mul(z0).add(s.mul(z1));
        z[i * n + (p + 1)] = s.neg().mul(z0).add(c.mul(z1));
    }
}

/// Scan the (quasi-triangular) `h`, splitting each real-eigenvalue 2×2 block.
///
/// For a block `[[a,b],[c,d]]` the discriminant of the characteristic
/// polynomial is `(a−d)² + 4bc`. When it is `≥ 0` the eigenvalues are real; the
/// rotation whose first column is the (unit) eigenvector `u` (so `T u = λ u`)
/// triangularizes the block, since `Gᵀ T G e₁ = λ e₁`. Complex blocks
/// (`disc < 0`) are left intact.
pub(super) fn standardize<T: RealScalar>(h: &mut [T], z: &mut [T], n: usize) {
    let mut p = 0usize;
    while p + 1 < n {
        if at(h, p + 1, p, n) == T::ZERO {
            p += 1;
            continue;
        }
        let a = at(h, p, p, n);
        let b = at(h, p, p + 1, n);
        let c = at(h, p + 1, p, n);
        let d = at(h, p + 1, p + 1, n);
        let diff = a.sub(d);
        let four = T::from_f64(4.0);
        let disc = diff.mul(diff).add(four.mul(b).mul(c));
        if disc < T::ZERO {
            // Complex conjugate pair: leave the 2×2 block as a Schur block.
            p += 2;
            continue;
        }

        // Real eigenvalues: triangularize. λ uses the cancellation-avoiding sign.
        let sign = if diff < T::ZERO { T::ONE.neg() } else { T::ONE };
        let half = T::from_f64(0.5);
        let lambda = a.add(d).add(sign.mul(disc.sqrt())).mul(half);

        // Eigenvector u = (b, λ − a) or (λ − d, c); pick the larger-magnitude
        // generator for stability.
        let (ex, ey) = if b.abs() >= (lambda.sub(d)).abs() {
            (b, lambda.sub(a))
        } else {
            (lambda.sub(d), c)
        };
        let norm = ex.mul(ex).add(ey.mul(ey)).sqrt();
        if norm > T::ZERO {
            let cs = ex.div(norm);
            let sn = ey.div(norm);
            apply_rotation(h, z, p, cs, sn, n);
            h[(p + 1) * n + p] = T::ZERO; // exact triangular zero
        }
        p += 2;
    }
}
