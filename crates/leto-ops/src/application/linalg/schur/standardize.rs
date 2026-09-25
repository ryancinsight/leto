//! Canonicalize the real Schur form: split every 2×2 diagonal block that has
//! **real** eigenvalues into two 1×1 blocks, and bring each genuine
//! complex-conjugate block to `dlanv2`'s standard form.

use super::standard_block::standardize_block;
use crate::domain::real::RealScalar;

#[inline]
fn at<T: Copy>(h: &[T], i: usize, j: usize, n: usize) -> T {
    h[i * n + j]
}

/// Apply the rotation similarity `G = [[c, −s], [s, c]]` on indices `(p, p+1)`
/// outside the 2×2 block, as `dlahqr` does after `dlanv2` (its `DROT` calls):
/// rows `(p, p+1)` right of the block, columns `(p, p+1)` above it, and all of
/// `Z ← Z G`. The block itself is written from `dlanv2`'s standardized entries,
/// and the entries left of and below it — zero in the Schur form, rounding
/// residue in the computed one — are not mixed into the block's subdiagonal.
fn apply_rotation<T: RealScalar>(h: &mut [T], z: &mut [T], p: usize, c: T, s: T, n: usize) {
    // Left: Gᵀ T over rows (p, p+1), the columns right of the block.
    for j in (p + 2)..n {
        let t0 = at(h, p, j, n);
        let t1 = at(h, p + 1, j, n);
        h[p * n + j] = c.mul(t0).add(s.mul(t1));
        h[(p + 1) * n + j] = s.neg().mul(t0).add(c.mul(t1));
    }
    // Right: T G over columns (p, p+1), the rows above the block.
    for i in 0..p {
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

/// Standardize every 2×2 diagonal block of the quasi-triangular `h` by
/// LAPACK `dlanv2` ([`standard_block`](super::standard_block)), as `dlahqr`
/// does when a 2×2 block deflates: the block's rotation `(cs, sn)` is applied
/// as the similarity `Gᵀ·H·G` to all of `h` and accumulated into `z`
/// (`dlahqr`'s `DROT` calls), and the block itself is replaced by `dlanv2`'s
/// standardized entries — upper triangular for real eigenvalues, equal
/// diagonals with `b·c < 0` for a complex pair.
pub(super) fn standardize<T: RealScalar>(h: &mut [T], z: &mut [T], n: usize) {
    let mut p = 0usize;
    while p + 1 < n {
        if at(h, p + 1, p, n) == T::ZERO {
            p += 1;
            continue;
        }
        let block = standardize_block(
            at(h, p, p, n),
            at(h, p, p + 1, n),
            at(h, p + 1, p, n),
            at(h, p + 1, p + 1, n),
        );
        apply_rotation(h, z, p, block.cs, block.sn, n);
        h[p * n + p] = block.a;
        h[p * n + p + 1] = block.b;
        h[(p + 1) * n + p] = block.c;
        h[(p + 1) * n + p + 1] = block.d;
        p += 2;
    }
}
