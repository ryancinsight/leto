//! The standardized Schur factorization of a real 2×2 block — LAPACK
//! `dlanv2` (Bai & Demmel's algorithm, LAPACK 3.x).
//!
//! For `B = [[a, b], [c, d]]` it computes a rotation `R = [[cs, −sn],
//! [sn, cs]]` and the standardized `Rᵀ·B·R = [[a′, b′], [c′, d′]]`: either
//! upper triangular (`c′ = 0`, real eigenvalues `a′`, `d′`), or with
//! `a′ = d′` and `b′·c′ < 0` (a complex pair `a′ ± i·√|b′|·√|c′|`).
//!
//! It never forms the characteristic polynomial's discriminant
//! `(a + d)² − 4(ad − bc)`, which cancels to rounding level on a double
//! eigenvalue and returns `O(√ε)` errors there; it works from `p = (a − d)/2`
//! and `bc` scaled by `max(|p|, |b|, |c|)`, decides real versus complex only
//! when `p² + bc` is clearly away from rounding (`≥ 4ε` relative), and
//! otherwise first rotates the block to equal diagonals. Every product is of
//! scaled or square-rooted operands, so no intermediate over- or underflows
//! for entries anywhere in the range; the one exception, `σ = b + c` and
//! `a − d` in the equal-diagonal branch, is rescaled by `dlanv2`'s own
//! `SAFMN2`/`SAFMX2` loop.

use crate::application::linalg::scaling::hypot;
use crate::application::linalg::thresholds::{machine_epsilon, safe_min};
use crate::domain::real::RealScalar;

/// A standardized 2×2 block and the rotation that produced it.
#[derive(Clone, Copy)]
pub(super) struct StandardBlock<T> {
    pub(super) a: T,
    pub(super) b: T,
    pub(super) c: T,
    pub(super) d: T,
    pub(super) cs: T,
    pub(super) sn: T,
}

impl<T: RealScalar> StandardBlock<T> {
    /// The block's eigenvalues `(re₁, im₁, re₂, im₂)`.
    pub(super) fn eigenvalues(self) -> (T, T, T, T) {
        if self.c == T::ZERO {
            (self.a, T::ZERO, self.d, T::ZERO)
        } else {
            let im = self.b.abs().sqrt().mul(self.c.abs().sqrt());
            (self.a, im, self.d, im.neg())
        }
    }
}

/// `sign(1, x)` of Fortran: `+1` for `x ≥ 0` (and `−0`), `−1` otherwise.
fn sign_of<T: RealScalar>(x: T) -> T {
    if x < T::ZERO {
        T::ONE.neg()
    } else {
        T::ONE
    }
}

/// `|magnitude|` carrying the sign of `sign` (Fortran `SIGN(magnitude, sign)`).
fn with_sign<T: RealScalar>(magnitude: T, sign: T) -> T {
    magnitude.abs().mul(sign_of(sign))
}

/// `dlanv2`'s `SAFMN2 = 2^⌊log₂(safmin/ε)/2⌋` (rounded toward zero): the
/// rescaling step of its equal-diagonal branch.
fn safmn2<T: RealScalar>() -> T {
    let smlnum = safe_min::<T>().div(machine_epsilon::<T>());
    let exponent = smlnum.binary_exponent().unwrap_or(0) / 2;
    T::ONE.scale_binary(exponent)
}

/// LAPACK `dlanv2` on the block `[[a, b], [c, d]]`.
#[allow(
    clippy::many_single_char_names,
    reason = "the names are dlanv2's own a, b, c, d, p, z"
)]
pub(super) fn standardize_block<T: RealScalar>(a: T, b: T, c: T, d: T) -> StandardBlock<T> {
    let (mut a, mut b, mut c, mut d) = (a, b, c, d);
    let (mut cs, mut sn);
    let zero = T::ZERO;
    let half = T::from_f64(0.5);
    let eps = machine_epsilon::<T>();
    if c == zero {
        cs = T::ONE;
        sn = zero;
    } else if b == zero {
        // Swap rows and columns.
        cs = zero;
        sn = T::ONE;
        core::mem::swap(&mut a, &mut d);
        b = c.neg();
        c = zero;
    } else if a.sub(d) == zero && sign_of(b) != sign_of(c) {
        cs = T::ONE;
        sn = zero;
    } else {
        let mut temp = a.sub(d);
        let mut p = half.mul(temp);
        let bcmax = if b.abs() > c.abs() { b.abs() } else { c.abs() };
        let bcmis = (if b.abs() < c.abs() { b.abs() } else { c.abs() })
            .mul(sign_of(b))
            .mul(sign_of(c));
        let scale = if p.abs() > bcmax { p.abs() } else { bcmax };
        let mut z = p.div(scale).mul(p).add(bcmax.div(scale).mul(bcmis));
        // `MULTPL = 4`: decide real eigenvalues only clear of rounding.
        // `dlanv2` compares `z` itself — which carries the block's units —
        // with `4ε`, so its decision changes with the block's scale (a block
        // multiplied by `2⁻⁶⁰⁰` took the equal-diagonal branch); the
        // comparison here is on the scale-free `z/scale`.
        if z.div(scale) >= T::from_f64(4.0).mul(eps) {
            // Real eigenvalues: compute a and d.
            z = p.add(with_sign(scale.sqrt().mul(z.sqrt()), p));
            a = d.add(z);
            d = d.sub(bcmax.div(z).mul(bcmis));
            // The rotation.
            let tau = hypot(c, z);
            cs = z.div(tau);
            sn = c.div(tau);
            b = b.sub(c);
            c = zero;
        } else {
            // Complex, or real (almost) equal eigenvalues: make the diagonal
            // elements equal.
            let small = safmn2::<T>();
            let large = T::ONE.div(small);
            let mut sigma = b.add(c);
            for _ in 0..=20 {
                let scale = if temp.abs() > sigma.abs() {
                    temp.abs()
                } else {
                    sigma.abs()
                };
                if scale >= large {
                    sigma = sigma.mul(small);
                    temp = temp.mul(small);
                } else if scale <= small {
                    sigma = sigma.mul(large);
                    temp = temp.mul(large);
                } else {
                    break;
                }
            }
            p = half.mul(temp);
            let tau = hypot(sigma, temp);
            cs = half.mul(T::ONE.add(sigma.abs().div(tau))).sqrt();
            sn = p.div(tau.mul(cs)).neg().mul(sign_of(sigma));
            // [aa bb; cc dd] = [a b; c d]·[cs −sn; sn cs]
            let aa = a.mul(cs).add(b.mul(sn));
            let bb = a.neg().mul(sn).add(b.mul(cs));
            let cc = c.mul(cs).add(d.mul(sn));
            let dd = c.neg().mul(sn).add(d.mul(cs));
            // [a b; c d] = [cs sn; −sn cs]·[aa bb; cc dd]
            a = aa.mul(cs).add(cc.mul(sn));
            b = bb.mul(cs).add(dd.mul(sn));
            c = aa.neg().mul(sn).add(cc.mul(cs));
            d = bb.neg().mul(sn).add(dd.mul(cs));
            let temp = half.mul(a.add(d));
            a = temp;
            d = temp;
            if c != zero {
                if b == zero {
                    b = c.neg();
                    c = zero;
                    let rotated = cs;
                    cs = sn.neg();
                    sn = rotated;
                } else if sign_of(b) == sign_of(c) {
                    // Real eigenvalues: reduce to upper triangular form.
                    let sab = b.abs().sqrt();
                    let sac = c.abs().sqrt();
                    p = with_sign(sab.mul(sac), c);
                    let tau = T::ONE.div(b.add(c).abs().sqrt());
                    a = temp.add(p);
                    d = temp.sub(p);
                    b = b.sub(c);
                    c = zero;
                    let cs1 = sab.mul(tau);
                    let sn1 = sac.mul(tau);
                    let rotated = cs.mul(cs1).sub(sn.mul(sn1));
                    sn = cs.mul(sn1).add(sn.mul(cs1));
                    cs = rotated;
                }
            }
        }
    }
    StandardBlock { a, b, c, d, cs, sn }
}
