//! Wilkinson-shifted single-shift complex QR iteration on an upper Hessenberg
//! matrix, with one-eigenvalue-at-a-time deflation.

use super::complex::Cplx;
use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// Iteration cap before declaring non-convergence (with Wilkinson + exceptional
/// shifts, real spectra converge in `O(n)` steps; this is a safety bound).
const MAX_ITER: usize = 2000;

/// A complex Givens rotation `G = [[c, s], [−s̄, c]]` (`c` real, `G` unitary)
/// chosen so that `G·(a, b)ᵀ = (r, 0)ᵀ`.
struct Givens<T> {
    c: T,
    s: Cplx<T>,
}

/// Construct the rotation that zeroes `b`.
///
/// Derivation (verified in the module test): for `a, b ≠ 0`,
/// `c = |a|/ρ`, `s = (a/|a|)·b̄/ρ` with `ρ = √(|a|²+|b|²)`. Then `c² + |s|² = 1`
/// (unitary) and `c·a + s·b = (a/|a|)·ρ`, `−s̄·a + c·b = 0`.
fn givens<T: RealScalar>(a: Cplx<T>, b: Cplx<T>) -> Givens<T> {
    if b.abs_sq() <= T::ZERO {
        return Givens {
            c: T::ONE,
            s: Cplx::zero(),
        };
    }
    if a.abs_sq() <= T::ZERO {
        // Rotation [[0, 1], [−1, 0]] sends (0, b) → (b, 0).
        return Givens {
            c: T::ZERO,
            s: Cplx::real(T::ONE),
        };
    }
    let abs_a = a.abs();
    let rho = a.abs_sq().add(b.abs_sq()).sqrt();
    let c = abs_a.div(rho);
    // s = (a/|a|)·conj(b)/ρ.
    let s = a
        .scale(T::ONE.div(abs_a))
        .mul(b.conj())
        .scale(T::ONE.div(rho));
    Givens { c, s }
}

/// Left-apply `G` to rows `(k, k+1)` across columns `lo..=hi`
/// (`[row_k; row_{k+1}] ← G · [row_k; row_{k+1}]`).
fn apply_rows<T: RealScalar>(
    g: &Givens<T>,
    h: &mut [Cplx<T>],
    n: usize,
    k: usize,
    lo: usize,
    hi: usize,
) {
    for j in lo..=hi {
        let t0 = h[k * n + j];
        let t1 = h[(k + 1) * n + j];
        h[k * n + j] = t0.scale(g.c).add(g.s.mul(t1));
        h[(k + 1) * n + j] = Cplx::zero().sub(g.s.conj().mul(t0)).add(t1.scale(g.c));
    }
}

/// Right-apply `Gᴴ` to columns `(k, k+1)` across rows `lo..=hi`
/// (`[col_k, col_{k+1}] ← [col_k, col_{k+1}] · Gᴴ`), which combined with
/// [`apply_rows`] makes the QR step a unitary similarity `H ← Gᴴ H G`.
fn apply_cols<T: RealScalar>(
    g: &Givens<T>,
    h: &mut [Cplx<T>],
    n: usize,
    k: usize,
    lo: usize,
    hi: usize,
) {
    for i in lo..=hi {
        let t0 = h[i * n + k];
        let t1 = h[i * n + (k + 1)];
        h[i * n + k] = t0.scale(g.c).add(t1.mul(g.s.conj()));
        h[i * n + (k + 1)] = Cplx::zero().sub(t0.mul(g.s)).add(t1.scale(g.c));
    }
}

/// The two eigenvalues of the 2×2 block at rows/cols `(p, p+1)`.
fn eig_2x2<T: RealScalar>(h: &[Cplx<T>], n: usize, p: usize) -> (Cplx<T>, Cplx<T>) {
    let a = h[p * n + p];
    let b = h[p * n + p + 1];
    let c = h[(p + 1) * n + p];
    let d = h[(p + 1) * n + p + 1];
    let tr = a.add(d);
    let det = a.mul(d).sub(b.mul(c));
    let two = Cplx::real(T::ONE.add(T::ONE));
    let four = Cplx::real(T::ONE.add(T::ONE).add(T::ONE).add(T::ONE));
    let disc = tr.mul(tr).sub(four.mul(det)).sqrt();
    (tr.add(disc).div(two), tr.sub(disc).div(two))
}

/// Wilkinson shift: the eigenvalue of the trailing 2×2 block (rows `hi-1, hi`)
/// closest to the corner `H[hi][hi]`.
fn wilkinson_shift<T: RealScalar>(h: &[Cplx<T>], n: usize, hi: usize) -> Cplx<T> {
    let (e1, e2) = eig_2x2(h, n, hi - 1);
    let corner = h[hi * n + hi];
    if e1.sub(corner).abs() <= e2.sub(corner).abs() {
        e1
    } else {
        e2
    }
}

/// Run the shifted QR iteration on the complex Hessenberg `h` (n×n, row-major,
/// mutated), returning the `n` eigenvalues.
///
/// # Errors
/// [`LetoError::StorageError`] if a block fails to converge within [`MAX_ITER`].
pub(super) fn run_qr<T: RealScalar>(h: &mut [Cplx<T>], n: usize) -> Result<Vec<Cplx<T>>> {
    let mut eigs = Vec::with_capacity(n);
    if n == 0 {
        return Ok(eigs);
    }
    let mut hi = n - 1;
    let mut iter = 0usize;

    loop {
        // Find the top of the bottom-most unreduced block: scan up while the
        // subdiagonal is non-negligible. Negligibility uses the precision-exact
        // test `d + |sub| == d` (LAPACK-style), valid for any float width.
        let mut lo = hi;
        while lo > 0 {
            let sub = h[lo * n + (lo - 1)].abs();
            let d = h[(lo - 1) * n + (lo - 1)].abs().add(h[lo * n + lo].abs());
            if d.add(sub) == d {
                h[lo * n + (lo - 1)] = Cplx::zero();
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            eigs.push(h[hi * n + hi]);
            if hi == 0 {
                break;
            }
            hi -= 1;
            iter = 0;
            continue;
        }
        if lo == hi - 1 {
            let (e1, e2) = eig_2x2(h, n, lo);
            eigs.push(e1);
            eigs.push(e2);
            if lo == 0 {
                break;
            }
            hi = lo - 1;
            iter = 0;
            continue;
        }

        iter += 1;
        if iter > MAX_ITER {
            return Err(LetoError::StorageError {
                reason: "eigenvalue QR iteration failed to converge".to_string(),
            });
        }

        // Exceptional shift every 10 stalled iterations to break cycling.
        let shift = if iter.is_multiple_of(10) {
            h[hi * n + hi].add(Cplx::real(h[hi * n + (hi - 1)].abs()))
        } else {
            wilkinson_shift(h, n, hi)
        };

        // Shifted QR step (similarity H ← Qᴴ H Q with QR = H − μI):
        // subtract shift, triangularize with Givens (forming R), apply the
        // rotations on the right (forming RQ), add the shift back.
        for i in lo..=hi {
            h[i * n + i] = h[i * n + i].sub(shift);
        }
        let mut rotations = Vec::with_capacity(hi - lo);
        for k in lo..hi {
            let g = givens(h[k * n + k], h[(k + 1) * n + k]);
            apply_rows(&g, h, n, k, lo, hi);
            rotations.push(g);
        }
        for (offset, g) in rotations.iter().enumerate() {
            apply_cols(g, h, n, lo + offset, lo, hi);
        }
        for i in lo..=hi {
            h[i * n + i] = h[i * n + i].add(shift);
        }
    }

    Ok(eigs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn givens_zeroes_second_component_and_is_unitary() {
        let a = Cplx::new(3.0_f64, -1.0);
        let b = Cplx::new(2.0_f64, 4.0);
        let g = givens(a, b);
        // c² + |s|² = 1.
        assert!((g.c * g.c + g.s.abs_sq() - 1.0).abs() < 1e-12);
        // Lower component −s̄·a + c·b must vanish.
        let lower = Cplx::zero().sub(g.s.conj().mul(a)).add(b.scale(g.c));
        assert!(lower.abs() < 1e-12);
    }
}
