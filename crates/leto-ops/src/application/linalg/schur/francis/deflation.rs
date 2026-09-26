//! Deflation: the run's small-subdiagonal threshold and `dlahqr`'s
//! Ahues–Tisseur negligibility test.

use super::{at, deflation_floor};
use crate::domain::real::RealScalar;

/// The deflation threshold of a run on the order-`n` Hessenberg `h`: the
/// larger of [`deflation_floor`] and LAPACK `dlahqr`'s
/// `SMLNUM = safmin·(n/ulp)` taken in units of the matrix,
/// `2^e·min(2^⌈log₂ n⌉·safmin/ε, ε)` with `2^e ≤ ‖H‖_max/2^⌈log₂ n⌉`.
///
/// *Why in units of the matrix.* `dlahqr`'s `SMLNUM` is absolute, calibrated
/// for entries of order one. A zero diagonal entry (every skew-symmetric
/// iterate) makes the Ahues–Tisseur right side `ulp·bb·aa/s` zero, so such a
/// subdiagonal deflates only at `SMLNUM`; on a matrix of scale `S` it must
/// first fall to `SMLNUM`, but the bulge's first column divides it by `S`
/// (`h₁₀/S` in [`first_column`](super::shift::first_column)), which underflows once it passes
/// `safmin·S` — for `S ≳ n/ulp` before it reaches `SMLNUM` — and the
/// iteration freezes (skew-symmetric tridiagonals with entries near `10⁵⁸`
/// stalled this way). In units of the matrix the test is the same for every
/// power-of-two scaling, exactly.
///
/// *Why capped at `ε`.* `safmin/ε` exceeds `ε` in `F16` (`2⁻⁴` against
/// `2⁻¹⁰`); the cap keeps every deflation at or below `ε·2^e`, and
/// `2^e ≤ ‖A‖_max/n` (`|h_{ij}| ≤ ‖H‖₂ = ‖A‖₂ ≤ n·‖A‖_max`), so at most `n`
/// of them cost at most `ε·‖A‖_max/√n ≤ ε·‖A‖_F` jointly, within
/// `backward_error.rs`'s count.
pub(super) fn run_floor<T: RealScalar>(h: &[T], n: usize) -> T {
    use crate::application::linalg::thresholds;
    let absolute = deflation_floor::<T>();
    let largest = h
        .iter()
        .fold(T::ZERO, |acc, &v| if v.abs() > acc { v.abs() } else { acc });
    let Some(exponent) = largest.binary_exponent() else {
        return absolute;
    };
    let eps = thresholds::machine_epsilon::<T>();
    let smlnum = thresholds::safe_min::<T>()
        .div(eps)
        .scale_binary(thresholds::ceil_log2_count(n));
    let relative = if smlnum < eps { smlnum } else { eps };
    let relative = relative.scale_binary(exponent - thresholds::ceil_log2_count(n));
    if relative > absolute {
        relative
    } else {
        absolute
    }
}

/// LAPACK `dlahqr`'s small-subdiagonal test for `h_{k,k−1}`: negligible when `|h_{k,k−1}| ≤ floor`
/// ([`run_floor`], `dlahqr`'s `SMLNUM` in units of the matrix), or when it passes
/// the ulp-relative pre-check `|h_{k,k−1}| ≤ ulp·tst` (`tst = |h_{k−1,k−1}| +
/// |h_{k,k}|`, and when both are zero the neighbouring subdiagonals
/// `|h_{k−1,k−2}| + |h_{k+1,k}|` as `dlahqr` takes them) **and** the
/// Ahues–Tisseur test (Ahues & Tisseur 1997, LAPACK
/// Working Note 122): with `ab = max(|h_{k,k−1}|, |h_{k−1,k}|)`,
/// `ba = min(…)`, `aa = max(|h_{k,k}|, |h_{k−1,k−1} − h_{k,k}|)`,
/// `bb = min(…)`, `s = aa + ab`,
/// `ba·(ab/s) ≤ max(floor, ulp·(bb·(aa/s)))`. `ulp = ε` (`dlamch('P')`).
pub(super) fn negligible_subdiagonal<T: RealScalar>(
    h: &[T],
    n: usize,
    k: usize,
    ulp: T,
    floor: T,
) -> bool {
    let sub = at(h, k, k - 1, n).abs();
    if sub <= floor {
        return true;
    }
    let mut tst = at(h, k - 1, k - 1, n).abs().add(at(h, k, k, n).abs());
    if tst == T::ZERO {
        if k >= 2 {
            tst = tst.add(at(h, k - 1, k - 2, n).abs());
        }
        if k + 1 < n {
            tst = tst.add(at(h, k + 1, k, n).abs());
        }
    }
    if sub > ulp.mul(tst) {
        return false;
    }
    let sup = at(h, k - 1, k, n).abs();
    let (ab, ba) = if sub > sup { (sub, sup) } else { (sup, sub) };
    let hkk = at(h, k, k, n).abs();
    let gap = at(h, k - 1, k - 1, n).sub(at(h, k, k, n)).abs();
    let (aa, bb) = if hkk > gap { (hkk, gap) } else { (gap, hkk) };
    let s = aa.add(ab);
    let relative = ulp.mul(bb.mul(aa.div(s)));
    ba.mul(ab.div(s)) <= if floor > relative { floor } else { relative }
}
