//! LAPACK `dbdsqr`'s zero-shift QR sweep on a bidiagonal block.

use super::bidiagonal_qr::{givens, rotate_row_pair};
use crate::application::linalg::scaling::KernelWindow;
use crate::domain::real::RealScalar;

/// LAPACK `dbdsqr`'s zero-shift QR sweep (Demmel & Kahan, loop 120, chasing
/// top to bottom only): one implicit QR step on `BᵀB` with shift `0`, which
/// converges where a shift near a tiny singular value would only flip signs
/// (`F16` skew-symmetric tridiagonals cycled with period 2 on the shifted
/// step). Without `dbdsqr`'s chase-direction choice (bottom to top when the
/// bottom of the block is heavier) the tiny singular values of a
/// bottom-heavy block are resolved only to the normwise backward error, not
/// to high relative accuracy (`LETO-BIDIAGONAL-CHASE-DIRECTION-2026-09-25`).
/// Right rotations update `V`, left rotations `U`.
#[allow(clippy::too_many_arguments)]
pub(super) fn zero_shift_sweep<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    u: &mut [T],
    m: usize,
    v: &mut [T],
    n: usize,
    rotation: KernelWindow<T>,
) {
    let (mut cs, mut oldcs, mut oldsn) = (T::ONE, T::ONE, T::ZERO);
    for i in p..q {
        let (c, s, r) = givens(d[i].mul(cs), e[i], rotation);
        if VEC {
            rotate_row_pair(v, n, i, i + 1, c, s); // V accumulated transposed
        }
        if i > p {
            e[i - 1] = oldsn.mul(r);
        }
        let (oc, os, di) = givens(oldcs.mul(r), d[i + 1].mul(s), rotation);
        if VEC {
            rotate_row_pair(u, m, i, i + 1, oc, os); // U accumulated transposed
        }
        d[i] = di;
        (cs, oldcs, oldsn) = (c, oc, os);
    }
    let h = d[q].mul(cs);
    d[q] = h.mul(oldcs);
    e[q - 1] = h.mul(oldsn);
}
