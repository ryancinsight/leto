//! The singular value decomposition of a 2×2 upper-triangular matrix,
//! LAPACK `dlasv2` (Demmel & Kahan, "Accurate singular values of bidiagonal
//! matrices", SIAM J. Sci. Stat. Comput. 11(5), 1990, §6):
//!
//! ```text
//! [ csl  snl ] [ f  g ] [ csr −snr ]   [ σ_max    0   ]
//! [−snl  csl ] [ 0  h ] [ snr  csr ] = [   0    σ_min ]
//! ```
//!
//! `|σ_max| ≥ |σ_min|`, both correct to a few ulps relative to themselves,
//! the rotations to a few ulps (`dlasv2`'s documented accuracy, barring
//! over- or underflow). Every quantity is a ratio of entries or a square of
//! such a ratio (the sums of squares through `dlapy2`), so no intermediate
//! over- or underflows for entries anywhere in the normal range;
//! `σ_min = h_a/a` and `σ_max = f_a·a`, `1 ≤ a ≤ 1 + |g/f|`, are the only
//! products of entries.

use crate::application::linalg::{scaling, thresholds};
use crate::domain::real::RealScalar;

/// `dlasv2`'s result: `σ_max`, `σ_min` (signed) and the right and left
/// rotations.
#[derive(Clone, Copy, Debug)]
pub(super) struct TriangularSvd<T> {
    pub(super) ssmax: T,
    pub(super) ssmin: T,
    pub(super) csr: T,
    pub(super) snr: T,
    pub(super) csl: T,
    pub(super) snl: T,
}

/// Fortran `SIGN(a, b)`: `|a|` with the sign of `b`.
fn sign<T: RealScalar>(a: T, b: T) -> T {
    if b < T::ZERO {
        a.abs().neg()
    } else {
        a.abs()
    }
}

/// LAPACK `dlasv2` on `[[f, g], [0, h]]`.
#[expect(
    clippy::many_single_char_names,
    reason = "the dlasv2 names f, g, h, l, m, t, s, r, a carry its published derivation"
)]
pub(super) fn triangular_svd<T: RealScalar>(f: T, g: T, h: T) -> TriangularSvd<T> {
    let two = T::ONE.add(T::ONE);
    let half = T::ONE.scale_binary(-1);
    // `dlamch('EPS')`, the unit roundoff.
    let unit_roundoff = thresholds::machine_epsilon::<T>().scale_binary(-1);
    let (mut ft, mut fa, mut ht, mut ha) = (f, f.abs(), h, h.abs());
    // `pmax`: which of f (1), g (2), h (3) is largest in magnitude.
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        (ft, ht) = (ht, ft);
        (fa, ha) = (ha, fa);
    }
    let (gt, ga) = (g, g.abs());
    let (ssmin, ssmax, clt, crt, slt, srt);
    if ga == T::ZERO {
        // Diagonal.
        (ssmin, ssmax) = (ha, fa);
        (clt, crt, slt, srt) = (T::ONE, T::ONE, T::ZERO, T::ZERO);
    } else if ga > fa && fa.div(ga) < unit_roundoff {
        // Very large g.
        pmax = 2;
        ssmax = ga;
        ssmin = if ha > T::ONE {
            fa.div(ga.div(ha))
        } else {
            fa.div(ga).mul(ha)
        };
        clt = T::ONE;
        slt = ht.div(gt);
        srt = T::ONE;
        crt = ft.div(gt);
    } else {
        if ga > fa {
            pmax = 2;
        }
        // Normal case.
        let d = fa.sub(ha);
        let l = if d == fa { T::ONE } else { d.div(fa) }; // 0 ≤ l ≤ 1
        let m = gt.div(ft); // |m| ≤ 1/u
        let t = two.sub(l); // t ≥ 1
                            // `dlasv2` forms `√(t² + m²)` and `√(l² + m²)` directly, which needs
                            // `1/u²` representable; in `F16` (`1/u² = 2²²` against `Ω < 2¹⁶`) it
                            // is not, so both go through `dlapy2`. `m·m` is kept only for the
                            // exact-underflow test below (an overflow to `+∞` compares unequal
                            // to zero, as the large product it stands for does).
        let mm = m.mul(m);
        let s = scaling::hypot(t, m); // 1 ≤ s ≤ 1 + 1/u
        let r = if l == T::ZERO {
            m.abs()
        } else {
            scaling::hypot(l, m)
        };
        let a = half.mul(s.add(r)); // 1 ≤ a ≤ 1 + |m|
        ssmin = ha.div(a);
        ssmax = fa.mul(a);
        let t = if mm == T::ZERO {
            // m is tiny.
            if l == T::ZERO {
                sign(two, ft).mul(sign(T::ONE, gt))
            } else {
                gt.div(sign(d, ft)).add(m.div(t))
            }
        } else {
            m.div(s.add(t)).add(m.div(r.add(l))).mul(T::ONE.add(a))
        };
        let l = scaling::hypot(t, two);
        crt = two.div(l);
        srt = t.div(l);
        clt = crt.add(srt.mul(m)).div(a);
        slt = ht.div(ft).mul(srt).div(a);
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    // Signs of σ_max and σ_min.
    let tsign = match pmax {
        1 => sign(T::ONE, csr)
            .mul(sign(T::ONE, csl))
            .mul(sign(T::ONE, f)),
        2 => sign(T::ONE, snr)
            .mul(sign(T::ONE, csl))
            .mul(sign(T::ONE, g)),
        _ => sign(T::ONE, snr)
            .mul(sign(T::ONE, snl))
            .mul(sign(T::ONE, h)),
    };
    TriangularSvd {
        ssmax: sign(ssmax, tsign),
        ssmin: sign(ssmin, tsign.mul(sign(T::ONE, f)).mul(sign(T::ONE, h))),
        csr,
        snr,
        csl,
        snl,
    }
}

#[cfg(test)]
mod tests {
    use super::triangular_svd;
    use eunomia::F16;

    /// `[csl snl; −snl csl]·[f g; 0 h]·[csr −snr; snr csr]` in `f64`.
    fn rotated(f: f64, g: f64, h: f64) -> [f64; 4] {
        let r = triangular_svd(f, g, h);
        let left = [r.csl, r.snl, -r.snl, r.csl];
        let right = [r.csr, -r.snr, r.snr, r.csr];
        let b = [f, g, 0.0, h];
        let mul = |x: [f64; 4], y: [f64; 4]| {
            [
                x[0] * y[0] + x[1] * y[2],
                x[0] * y[1] + x[1] * y[3],
                x[2] * y[0] + x[3] * y[2],
                x[2] * y[1] + x[3] * y[3],
            ]
        };
        mul(mul(left, b), right)
    }

    /// The rotated matrix is `diag(σ_max, σ_min)` to within a few ulps of
    /// `‖B‖` (`dlasv2`: rotations accurate to a few ulps; each product of
    /// three 2×2 factors adds `γ₄` per entry — `8u‖B‖` covers both), and
    /// `|σ_max|·|σ_min| = |f·h|` (the determinant) to a few ulps of itself.
    #[test]
    fn diagonalizes_graded_and_subnormal_blocks() {
        let cases = [
            (3.0_f64, 1.0, 2.0),
            (1.0, 1e10, 1.0),
            (2.764e-311, 7.513e-305, 1.107e-291),
            (1e-300, 1.0, 1e-300),
            (-4.0, 0.5, 1e-12),
            (1.0, 0.0, -2.0),
        ];
        for (f, g, h) in cases {
            let r = triangular_svd(f, g, h);
            let norm = (f * f + g * g + h * h).sqrt();
            let out = rotated(f, g, h);
            let tolerance = 8.0 * f64::EPSILON * norm;
            assert!(
                (out[0] - r.ssmax).abs() <= tolerance,
                "{f} {g} {h}: {out:?} {r:?}"
            );
            assert!(
                out[1].abs() <= tolerance && out[2].abs() <= tolerance,
                "{f} {g} {h}: {out:?}"
            );
            assert!(
                (out[3] - r.ssmin).abs() <= tolerance,
                "{f} {g} {h}: {out:?} {r:?}"
            );
            assert!(r.ssmax.abs() >= r.ssmin.abs());

            let det = (f * h).abs();
            assert!(
                (r.ssmax.abs() * r.ssmin.abs() - det).abs() <= 8.0 * f64::EPSILON * det,
                "{f} {g} {h}: {} vs {det}",
                r.ssmax.abs() * r.ssmin.abs()
            );
        }
    }

    /// `F16`, where `1/u² = 2²²` exceeds `Ω`: `|g/f| = 1024` would overflow
    /// `dlasv2`'s `m²`. The pair stays finite and each value matches the `f64`
    /// decomposition of the same entries to a few `F16` ulps of itself
    /// (`dlasv2`'s relative accuracy; eight ulps).
    #[test]
    fn stays_finite_in_f16() {
        use eunomia::NumericElement;
        let (f, g, h) = (1.0, 1024.0, 0.5);
        let narrow = triangular_svd(F16::from_f64(f), F16::from_f64(g), F16::from_f64(h));
        let wide = triangular_svd(f, g, h);
        let eps = 2.0_f64.powi(-10);
        for (a, b) in [(narrow.ssmax, wide.ssmax), (narrow.ssmin, wide.ssmin)] {
            assert!(a.to_f64().is_finite());
            assert!(
                (a.to_f64() - b).abs() <= 8.0 * eps * b.abs(),
                "{} vs {b}",
                a.to_f64()
            );
        }
    }
}
