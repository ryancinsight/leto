//! LAPACK `dbdsqr`'s zero-shift QR sweep on a bidiagonal block.

use super::chase::{Chase, Oriented};
use super::rotation::{givens, TransposedFactors};
use crate::application::linalg::scaling::KernelWindow;
use crate::domain::real::RealScalar;

/// LAPACK `dbdsqr`'s zero-shift QR sweep (Demmel & Kahan 1990, §3; loops
/// 120 and 130 of the reference `dbdsqr.f`): one implicit QR step on `BᵀB`
/// with shift `0`, chased in orientation `C` — top to bottom
/// ([`Down`](super::chase::Down)) or, as the reflection, bottom to top
/// ([`Up`](super::chase::Up)). Each sweep perturbs every singular value by
/// a relative `69n²ε/(1 − 69n²ε)` at most (Demmel & Kahan, Theorem 6), so
/// the tiny singular values keep high relative accuracy; it converges where
/// a shift near a tiny singular value would only flip signs (`F16`
/// skew-symmetric tridiagonals cycled with period 2 on the shifted step).
/// Oriented column rotations are accumulated by
/// [`Oriented::rotate_columns`], row rotations by
/// [`Oriented::rotate_rows`].
pub(super) fn zero_shift_sweep<T: RealScalar, const VEC: bool, C: Chase>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    factors: &mut TransposedFactors<'_, T>,
    rotation: KernelWindow<T>,
) {
    let mut block = Oriented::<T, C>::new(d, e, p, q);
    let (mut cs, mut oldcs, mut oldsn) = (T::ONE, T::ONE, T::ZERO);
    for i in p..q {
        let (c, s, r) = givens(block.d(i).mul(cs), block.e(i), rotation);
        if VEC {
            block.rotate_columns(factors, i, i + 1, (c, s));
        }
        if i > p {
            block.set_e(i - 1, oldsn.mul(r));
        }
        let (oc, os, di) = givens(oldcs.mul(r), block.d(i + 1).mul(s), rotation);
        if VEC {
            block.rotate_rows(factors, i, i + 1, (oc, os));
        }
        block.set_d(i, di);
        (cs, oldcs, oldsn) = (c, oc, os);
    }
    let h = block.d(q).mul(cs);
    block.set_d(q, h.mul(oldcs));
    block.set_e(q - 1, h.mul(oldsn));
}
