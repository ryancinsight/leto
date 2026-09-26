//! Shift selection: `dlahqr`'s Wilkinson and exceptional shift pairs and the
//! first column of the implicit double shift.

use super::at;
use crate::domain::real::RealScalar;

/// Which shift pair a Francis step uses — LAPACK `dlahqr`'s choice by the
/// iteration count since the last deflation (`KDEFL`, `KEXSH = 10`).
#[derive(Clone, Copy)]
pub(super) enum Shift {
    /// The eigenvalues of the trailing 2×2 block.
    Wilkinson,
    /// Exceptional, from the top of the active block (`KDEFL ≡ 10 mod 20`).
    ExceptionalTop,
    /// Exceptional, from the bottom of the active block (`KDEFL ≡ 0 mod 20`).
    ExceptionalBottom,
}

impl Shift {
    /// `dlahqr`: every `2·KEXSH` iterations an exceptional shift from the
    /// bottom, every other `KEXSH` one from the top.
    pub(super) fn for_iteration(iteration: usize) -> Self {
        const KEXSH: usize = 10;
        if iteration.is_multiple_of(2 * KEXSH) {
            Self::ExceptionalBottom
        } else if iteration.is_multiple_of(KEXSH) {
            Self::ExceptionalTop
        } else {
            Self::Wilkinson
        }
    }
}

/// `dlahqr`'s shift pair `(rt1r, rt1i, rt2r, rt2i)` from the 2×2
/// `[[h11, h12], [h21, h22]]`, computed on the block divided by
/// `s = |h11| + |h12| + |h21| + |h22|` (so no product over- or underflows).
/// A complex pair is returned as is; two real shifts are replaced by the one
/// nearer `h22`, used twice.
pub(super) fn shift_pair<T: RealScalar>(h11: T, h12: T, h21: T, h22: T) -> (T, T, T, T) {
    let s = h11.abs().add(h12.abs()).add(h21.abs()).add(h22.abs());
    if s == T::ZERO {
        return (T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    }
    let (h11, h12, h21, h22) = (h11.div(s), h12.div(s), h21.div(s), h22.div(s));
    let half = T::from_f64(0.5);
    let tr = h11.add(h22).mul(half);
    let det = h11.sub(tr).mul(h22.sub(tr)).sub(h12.mul(h21));
    let rtdisc = det.abs().sqrt();
    if det >= T::ZERO {
        let re = tr.mul(s);
        let im = rtdisc.mul(s);
        (re, im, re, im.neg())
    } else {
        let rt1r = tr.add(rtdisc);
        let rt2r = tr.sub(rtdisc);
        let nearer = if rt1r.sub(h22).abs() <= rt2r.sub(h22).abs() {
            rt1r
        } else {
            rt2r
        };
        let re = nearer.mul(s);
        (re, T::ZERO, re, T::ZERO)
    }
}

/// First column of `(H − μ₁I)(H − μ₂I)` at row `m`, as `dlahqr` forms it:
/// divided by `S = |h_{mm} − μ₂| + |Im μ₂| + |h_{m+1,m}|` before any product,
/// then normalized by `|v₁| + |v₂| + |v₃|` — every product is of ratios, so
/// none over- or underflows for entries anywhere in the range. Only the
/// direction enters the first reflector.
pub(super) fn first_column<T: RealScalar>(
    h: &[T],
    n: usize,
    m: usize,
    shifts: (T, T, T, T),
) -> (T, T, T) {
    let (rt1r, rt1i, rt2r, rt2i) = shifts;
    let h00 = at(h, m, m, n);
    let h01 = at(h, m, m + 1, n);
    let h10 = at(h, m + 1, m, n);
    let h11 = at(h, m + 1, m + 1, n);
    let h21 = at(h, m + 2, m + 1, n);
    let s = h00.sub(rt2r).abs().add(rt2i.abs()).add(h10.abs());
    let h10s = h10.div(s);
    let x = h10s
        .mul(h01)
        .add(h00.sub(rt1r).mul(h00.sub(rt2r).div(s)))
        .sub(rt1i.mul(rt2i.div(s)));
    let y = h10s.mul(h00.add(h11).sub(rt1r).sub(rt2r));
    let z = h10s.mul(h21);
    let norm1 = x.abs().add(y.abs()).add(z.abs());
    if norm1 > T::ZERO {
        (x.div(norm1), y.div(norm1), z.div(norm1))
    } else {
        (x, y, z)
    }
}
