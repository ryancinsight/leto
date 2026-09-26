use super::{
    gate_exponent, norm_ratio_floor_log2, norm_ratio_log2, restore, scale_by_power_of_two,
    GateBound, KernelWindow,
};
use crate::application::linalg::thresholds::homogeneous_safe_range;
use crate::domain::real::RealScalar;
use leto::LetoError;

fn no_factor(_: &[f64], _: f64) -> GateBound {
    GateBound::factor(0)
}

#[test]
fn in_range_values_are_not_scaled() {
    // Degree 2, bound 2⁰: f64's range is [√(2⁻⁹⁷⁰), √Ω) ≈ [2⁻⁴⁸⁵, 2⁵¹²).
    let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 0, 0).unwrap();
    assert!(
        rmin > 2.0_f64.powi(-486) && rmin < 2.0_f64.powi(-484),
        "{rmin}"
    );
    assert!(
        rmax > 2.0_f64.powi(511) && rmax < 2.0_f64.powi(512),
        "{rmax}"
    );
    for value in [1.0_f64, 3.99, 4.0, 0.5, 0.2, 1e100, 1e-100, 1e150, 1e-145] {
        assert_eq!(
            gate_exponent(&[value, -value / 3.0], 2, no_factor).unwrap(),
            None,
            "{value} should factor unscaled"
        );
    }
    assert_eq!(gate_exponent(&[0.0_f64, -0.0], 2, no_factor).unwrap(), None);
    assert_eq!(
        gate_exponent(&[1.0_f64, f64::INFINITY], 2, no_factor).unwrap(),
        None
    );
    assert_eq!(
        gate_exponent(&Vec::<f64>::new(), 2, no_factor).unwrap(),
        None
    );
}

#[test]
fn degree_one_upper_end_is_the_overflow_threshold() {
    // Degree 1 takes no root: the upper end is exactly Ω·2⁻ᶠ, so
    // `1.5·2¹⁰²²` (≤ Ω/2) is in range at bound 2¹ and out at 2².
    let value = 1.5 * 2.0_f64.powi(1022);
    assert_eq!(
        gate_exponent(&[value], 1, |_, _| GateBound::factor(1)).unwrap(),
        None
    );
    assert_eq!(
        gate_exponent(&[value], 1, |_, _| GateBound::factor(2)).unwrap(),
        Some(1)
    );
    let (_, rmax) = homogeneous_safe_range::<f64>(1, 1, 0).unwrap();
    assert_eq!(rmax, f64::MAX / 2.0);
}

#[test]
fn out_of_range_values_scale_by_the_minimal_move() {
    let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 0, 0).unwrap();
    let k = gate_exponent(&[1e300_f64], 2, no_factor)
        .unwrap()
        .expect("1e300 is out of range");
    let mut scaled = [1e300_f64];
    scale_by_power_of_two(&mut scaled, -k);
    assert!(scaled[0] <= rmax, "{} > {rmax}", scaled[0]);
    let mut one_less = [1e300_f64];
    scale_by_power_of_two(&mut one_less, -(k - 1));
    assert!(one_less[0] > rmax, "{k} was not minimal");

    let tiny = f64::from_bits(1);
    let k = gate_exponent(&[tiny], 2, no_factor)
        .unwrap()
        .expect("subnormal is out of range");
    assert!(k < 0);
    let mut scaled = [tiny];
    scale_by_power_of_two(&mut scaled, -k);
    assert!(scaled[0] >= rmin, "{} < {rmin}", scaled[0]);
    let mut one_less = [tiny];
    scale_by_power_of_two(&mut one_less, -(k + 1));
    assert!(one_less[0] < rmin, "{k} was not minimal");
}

#[test]
fn bound_factor_narrows_only_the_upper_end() {
    let (rmin0, rmax0) = homogeneous_safe_range::<f64>(4, 0, 0).unwrap();
    let (rmin, rmax) = homogeneous_safe_range::<f64>(4, 1000, 0).unwrap();
    assert_eq!(rmin, rmin0);
    assert!(rmax < rmax0);
    assert_eq!(gate_exponent(&[100.0_f64], 4, no_factor).unwrap(), None);
    assert!(matches!(
        gate_exponent(&[100.0_f64], 4, |_, _| GateBound::factor(1000)),
        Ok(Some(_))
    ));
}

#[test]
fn an_empty_range_is_a_typed_overflow() {
    // F16: Ω/smlnum = 65504·2⁴ < 2²⁰, so a degree-1 bound factor of 2²⁰ leaves
    // no scaling that keeps the intermediates finite.
    let values = [eunomia::F16::from_f64(1.0)];
    assert!(matches!(
        gate_exponent(&values, 1, |_, _| GateBound::factor(20)),
        Err(LetoError::Overflow { .. })
    ));
}

#[test]
fn an_unfittable_deflation_floor_is_a_typed_overflow() {
    // F16, degree 2, no bound factor: upper end √Ω < 2⁸; a floor needing
    // ‖A‖_max ≥ 2^13·smlnum = 2⁹ cannot be met by any scaling.
    let one = eunomia::F16::from_f64(1.0);
    let floor = |_: &[eunomia::F16], _| GateBound {
        factor_log2: 0,
        floor_log2: 13,
    };
    match gate_exponent(&[one], 2, floor) {
        Err(LetoError::Overflow { reason }) => {
            assert!(reason.contains("deflation floor"), "{reason}");
        }
        other => panic!("expected the deflation-floor overflow, got {other:?}"),
    }
}

#[test]
fn deflation_floor_raises_only_the_lower_end() {
    // Degree 1: the lower end is smlnum = 2⁻⁹⁷⁰, raised to 2⁶·smlnum by a
    // 2⁶·safmin deflation floor so that floor is ε·‖A‖_max at most.
    let (rmin, rmax) = homogeneous_safe_range::<f64>(1, 0, 0).unwrap();
    let (floor_rmin, floor_rmax) = homogeneous_safe_range::<f64>(1, 0, 6).unwrap();
    assert_eq!(rmin, 2.0_f64.powi(-970));
    assert_eq!(floor_rmin, 2.0_f64.powi(-964));
    assert_eq!(floor_rmax, rmax);
    assert!(64.0 * f64::MIN_POSITIVE <= f64::EPSILON * floor_rmin);
}

#[test]
fn window_exponent_rescales_only_outside_the_window() {
    let window = KernelWindow::<f64>::new(2, 2, 1);
    assert_eq!(window.exponent(&[3.0, -1e-300]), 0);
    assert_eq!(window.exponent(&[0.0, -0.0]), 0);
    assert_eq!(window.exponent(&[1.0, -(2.0_f64.powi(600))]), 600);
    assert_eq!(window.exponent(&[1.5 * 2.0_f64.powi(-600)]), -600);
}

#[test]
fn norm_ratio_is_tight_for_a_diagonal_and_bounded_by_the_dimension() {
    assert_eq!(norm_ratio_log2(&[1e300_f64, 0.0, 0.0, 1e-300], 1e300), 0);
    // All-ones 4×4: ‖A‖_F/‖A‖_max = 4 = 2².
    assert_eq!(norm_ratio_log2(&[1.0_f64; 16], 1.0), 2);
    // All-ones 3×3: 3, rounded up to 2².
    assert_eq!(norm_ratio_log2(&[1.0_f64; 9], 1.0), 2);
    // 128² unit entries in F16: ‖A‖_F/‖A‖_max = 2⁷. A recursive sum
    // stagnates at 2048 (2048 + 1 rounds to 2048) and charged r = 6.
    let one = eunomia::F16::from_f64(1.0);
    assert_eq!(norm_ratio_log2(&vec![one; 128 * 128], one), 7);
    // A computed sum just below a power of two may hide a true one above
    // it: 64 entries, one 1 and fifteen (1 − 2⁻⁶), sum 1 + 15·0.96875 =
    // 15.53 inside the F16 band (16·(1 − 2⁻⁴), 16), are charged
    // ⌈log₂ √32⌉ = 3 rather than ⌈log₂ √15.53⌉ = 2.
    let near = eunomia::F16::from_f64(1.0 - 2.0_f64.powi(-6));
    let zero = eunomia::F16::from_f64(0.0);
    let mut band = vec![zero; 64];
    band[0] = one;
    band[1..16].fill(near);
    assert_eq!(norm_ratio_log2(&band, one), 3);
}

#[test]
fn restore_reports_an_unrepresentable_result() {
    let mut values = [1.5_f64, 3.0];
    assert!(matches!(
        restore(&mut values, 1023, "probe"),
        Err(LetoError::Overflow { reason: "probe" })
    ));
    let mut values = [1.5_f64, -3.0];
    restore(&mut values, 4, "probe").expect("in range");
    assert_eq!(values, [24.0, -48.0]);
}

/// `‖values‖_F/‖values‖_max` in `f64`, with its relative rounding bound
/// `γ_{len+1}(ε₆₄)` (each term exact, `len` additions and a square root).
fn exact_ratio<T: RealScalar>(values: &[T]) -> (f64, f64) {
    let largest = values
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.to_f64().abs()));
    let sum: f64 = values.iter().map(|v| (v.to_f64() / largest).powi(2)).sum();
    let slack = (values.len() as f64 + 2.0) * f64::EPSILON;
    (sum.sqrt(), slack)
}

/// `2^l ≤ ‖A‖_F/‖A‖_max ≤ 2^r` on adversarial vectors in every format:
/// log-uniform magnitudes across each format's whole range (subnormals
/// included), constant vectors of every length, and near-power sums.
fn check_ratio_bounds<T: RealScalar>(seed: u64, min_exp: i32, max_exp: i32) {
    let mut rng = crate::Xorshift64::new(seed);
    let mut cases: Vec<Vec<T>> = Vec::new();
    for len in [
        1_usize, 2, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 255, 256, 1000, 4096,
    ] {
        cases.push(vec![T::ONE; len]);
        for _ in 0..6 {
            let span = f64::from(max_exp - min_exp);
            cases.push(
                (0..len)
                    .map(|_| {
                        let e = f64::from(min_exp) + span * rng.next_unit_f64();
                        T::from_f64((rng.next_unit_f64() - 0.5) * e.exp2())
                    })
                    .collect(),
            );
            let base = rng.next_unit_f64();
            cases.push(
                (0..len)
                    .map(|_| T::from_f64(base * (1.0 + 1e-3 * rng.next_unit_f64())))
                    .collect(),
            );
        }
    }
    for values in cases {
        let largest = values
            .iter()
            .fold(T::ZERO, |acc, v| if v.abs() > acc { v.abs() } else { acc });
        if largest == T::ZERO {
            continue;
        }
        let (ratio, slack) = exact_ratio(&values);
        let (l, r) = (
            norm_ratio_floor_log2(&values, largest),
            norm_ratio_log2(&values, largest),
        );
        let len = values.len();
        assert!(
            2.0_f64.powi(l) <= ratio * (1.0 + slack),
            "len {len}: 2^{l} > ratio {ratio}"
        );
        // `r` treats a sum exactly on a power of two as exact (documented):
        // the true ratio may exceed `2^r` there by `√(1 + c)`, `c < ½`.
        assert!(
            ratio * (1.0 - slack) <= 2.0_f64.powi(r) * 1.5_f64.sqrt(),
            "len {len}: ratio {ratio} > 2^{r}·√1.5"
        );
        assert!(l <= r, "len {len}: l {l} > r {r}");
    }
}

#[test]
fn norm_ratio_bounds_hold_on_adversarial_vectors() {
    check_ratio_bounds::<f64>(0xA1, -1074, 1023);
    check_ratio_bounds::<f32>(0xA2, -149, 127);
    check_ratio_bounds::<eunomia::F16>(0xA3, -24, 15);
    check_ratio_bounds::<eunomia::Bf16>(0xA4, -133, 127);
}

/// Tight cases. `[1, 1, 1 − 2⁻¹¹, 1]` in F16: `s = 3.999…`, but the pairwise
/// sum rounds to exactly `4`; only the `(1 − 2c)` margin keeps `l = 0`
/// (`2^1 > √s`). `256²` unit F16 entries: the sum `2¹⁶` exceeds `Ω`, which
/// the running exponent absorbs, giving `r = 8` and `l = 7` rather than the
/// `l = 0` an overflowed sum forced.
#[test]
fn norm_ratio_floor_is_tight_where_rounding_crosses_a_power() {
    use eunomia::F16;
    let one = F16::from_f64(1.0);
    let below = F16::from_f64(0.999_511_718_75);
    let crossing = [one, one, below, one];
    assert_eq!(norm_ratio_floor_log2(&crossing, one), 0);
    let ones = vec![one; 256 * 256];
    assert_eq!(norm_ratio_log2(&ones, one), 8);
    assert_eq!(norm_ratio_floor_log2(&ones, one), 7);
}
