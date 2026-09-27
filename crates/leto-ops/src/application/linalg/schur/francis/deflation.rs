//! Deflation: the run's small-subdiagonal threshold and `dlahqr`'s
//! Ahues–Tisseur negligibility test.

use super::shift::KEXSH;
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

/// Which small-subdiagonal test a deflation scan applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Criterion {
    /// `dlahqr`'s (reference `dlahqr.f` loop 30, lines 319–344): the
    /// ulp-relative pre-check **and** Ahues–Tisseur.
    AhuesTisseur,
    /// `|h_{k,k−1}| ≤ ulp·(|h_{k−1,k−1}| + |h_{k,k}| + |h_{k−1,k−2}| +
    /// |h_{k+1,k}|)`, [`neighbourhood_scale`]. Leto's own test, for a
    /// [`STALLED`] block only. It is the normwise form of EISPACK `hqr`'s
    /// (`hqr.f` lines 88–92: `s = |h(l−1,l−1)| + |h(l,l)|`, the whole-matrix
    /// sum `norm` when `s = 0`, negligible when `s + |h(l,l−1)| = s`) with a
    /// different scale: both diagonal entries **and** both adjacent
    /// subdiagonals, because on the skew iterates this stage exists for the
    /// diagonal is rounding residue (not `0`, so `hqr` never reaches its
    /// `norm` fallback) and the scale that makes the coupling negligible is
    /// the neighbouring subdiagonals'; the whole-matrix sum would charge up to
    /// `n·‖H‖_F` per deflation. Charge: each entry of the scale is at most
    /// `‖H‖₂`, so a deflation perturbs `H` by at most `4·ulp·‖H‖₂`.
    Neighbourhood,
}

/// The iterations without a deflation after which a block is stalled:
/// `2·KEXSH`, once `dlahqr`'s exceptional shifts from both ends of the
/// block (`dlahqr.f` lines 369–387, at `KDEFL = KEXSH` and `2·KEXSH`) have
/// failed to break the stagnation they exist for. Leto's choice, not
/// `dlahqr`'s, which keeps iterating to its budget.
///
/// A stalled block's scan switches to [`Criterion::Neighbourhood`]. On
/// skew-symmetric iterates a block `[[0, a], [−a, 0]]` coupled by `δ` to
/// another `[[0, a], [−a, 0]]` has eigenvalues `±i(√(a² + δ²/4) ± δ/2)`, two
/// clusters the Wilkinson shift `±ia` sits exactly between; once `δ` is
/// within a few ulps of `a` the Bf16 rounding restores the equal corners
/// after every exceptional shift and the step cycles with period 2
/// (`LETO-BF16-SKEW-FRANCIS-STALL-2026-09-25`). The diagonal there is zero
/// or rounding residue, so neither `dlahqr`'s `tst` nor Ahues–Tisseur sees
/// the scale `a` that makes `δ ≤ ulp·2a` negligible. Switching after the
/// exceptional shifts rather than at [`iteration_cap`] bounds the cycling:
/// every step also updates the Schur vectors, and in Bf16 the `30·67`
/// steps of the order-67 budget left them with `‖QᵀQ − I‖ > 1`.
pub(super) const STALLED: usize = 2 * KEXSH;

/// `dlahqr`'s iteration budget per deflation, `ITMAX = 30·max(10, NH)`
/// (`dlahqr.f` line 293, `NH` the order of the active range), past which
/// the iteration reports failure (the reference's `INFO > 0`).
pub(super) fn iteration_cap(n: usize) -> usize {
    30 * n.max(10)
}

/// `dlahqr`'s `tst` for `h_{k,k−1}`: `|h_{k−1,k−1}| + |h_{k,k}|`, or, when
/// both are zero, the neighbouring subdiagonals `|h_{k−1,k−2}| + |h_{k+1,k}|`.
pub(super) fn local_scale<T: RealScalar>(h: &[T], n: usize, k: usize) -> T {
    let diagonal = at(h, k - 1, k - 1, n).abs().add(at(h, k, k, n).abs());
    if diagonal != T::ZERO {
        return diagonal;
    }
    neighbours(h, n, k)
}

/// [`Criterion::Neighbourhood`]'s scale for `h_{k,k−1}`: `|h_{k−1,k−1}| +
/// |h_{k,k}| + |h_{k−1,k−2}| + |h_{k+1,k}|`.
pub(super) fn neighbourhood_scale<T: RealScalar>(h: &[T], n: usize, k: usize) -> T {
    at(h, k - 1, k - 1, n)
        .abs()
        .add(at(h, k, k, n).abs())
        .add(neighbours(h, n, k))
}

/// `|h_{k−1,k−2}| + |h_{k+1,k}|`, the subdiagonals beside `h_{k,k−1}`.
fn neighbours<T: RealScalar>(h: &[T], n: usize, k: usize) -> T {
    let below = if k >= 2 {
        at(h, k - 1, k - 2, n).abs()
    } else {
        T::ZERO
    };
    let above = if k + 1 < n {
        at(h, k + 1, k, n).abs()
    } else {
        T::ZERO
    };
    below.add(above)
}

/// LAPACK `dlahqr`'s small-subdiagonal test for `h_{k,k−1}` (`dlahqr.f`
/// lines 319–344): negligible when `|h_{k,k−1}| ≤ floor`
/// ([`run_floor`], `dlahqr`'s `SMLNUM` in units of the matrix), or when it passes
/// the ulp-relative pre-check `|h_{k,k−1}| ≤ ulp·tst` (`tst = |h_{k−1,k−1}| +
/// |h_{k,k}|`, and when both are zero the neighbouring subdiagonals
/// `|h_{k−1,k−2}| + |h_{k+1,k}|` as `dlahqr` takes them) **and** the
/// Ahues–Tisseur test (Ahues & Tisseur 1997, LAPACK
/// Working Note 122): with `ab = max(|h_{k,k−1}|, |h_{k−1,k}|)`,
/// `ba = min(…)`, `aa = max(|h_{k,k}|, |h_{k−1,k−1} − h_{k,k}|)`,
/// `bb = min(…)`, `s = aa + ab`,
/// `ba·(ab/s) ≤ max(floor, ulp·(bb·(aa/s)))`. `ulp = ε` (`dlamch('P')`).
///
/// Under [`Criterion::Neighbourhood`] its normwise test alone decides.
pub(super) fn negligible_subdiagonal<T: RealScalar>(
    h: &[T],
    n: usize,
    k: usize,
    (ulp, floor): (T, T),
    criterion: Criterion,
) -> bool {
    let sub = at(h, k, k - 1, n).abs();
    if sub <= floor {
        return true;
    }
    if criterion == Criterion::Neighbourhood {
        return sub <= ulp.mul(neighbourhood_scale(h, n, k));
    }
    if sub > ulp.mul(local_scale(h, n, k)) {
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

#[cfg(test)]
mod tests {
    use super::{negligible_subdiagonal, Criterion};
    use crate::application::linalg::thresholds;
    use crate::domain::real::RealScalar;
    use eunomia::{Bf16, F16};

    /// The skew tie `[[0, −1, 0, 0], [1, 0, −δ, 0], [0, δ, 0, −1],
    /// [0, 0, 1, 0]]` at `k = 2`: the diagonal is zero, so Ahues–Tisseur
    /// keeps `δ` at any size above the floor, while the neighbourhood test
    /// deflates exactly `δ ≤ ulp·(0 + 0 + 1 + 1) = 2ε` — `2ε` itself and not
    /// the next value above it, `2ε·(1 + ε)` — in every format.
    fn threshold_is_two_ulp<T: RealScalar>() {
        let ulp = thresholds::machine_epsilon::<T>();
        let floor = thresholds::safe_min::<T>();
        let skew = |delta: T| {
            let mut h = vec![T::ZERO; 16];
            for (k, value) in [(1, T::ONE), (2, delta), (3, T::ONE)] {
                h[k * 4 + k - 1] = value;
                h[(k - 1) * 4 + k] = value.neg();
            }
            h
        };
        let two_ulp = ulp.add(ulp);
        let at_threshold = skew(two_ulp);
        let above = skew(two_ulp.add(two_ulp.mul(ulp)));
        let test = |h: &[T], criterion| negligible_subdiagonal(h, 4, 2, (ulp, floor), criterion);
        let label = core::any::type_name::<T>();
        assert!(test(&at_threshold, Criterion::Neighbourhood), "{label}");
        assert!(!test(&at_threshold, Criterion::AhuesTisseur), "{label}");
        assert!(!test(&above, Criterion::Neighbourhood), "{label}");
    }

    #[test]
    fn neighbourhood_test_deflates_a_tie_ahues_tisseur_keeps() {
        threshold_is_two_ulp::<f64>();
        threshold_is_two_ulp::<f32>();
        threshold_is_two_ulp::<F16>();
        threshold_is_two_ulp::<Bf16>();
    }
}
