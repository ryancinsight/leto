use super::Arithmetic;
use crate::application::linalg::thresholds::{machine_epsilon, safe_min};
use crate::domain::real::RealScalar;

fn pivot_keys_remain_enclosed_at_exponent_boundaries<T: RealScalar>() {
    let epsilon = machine_epsilon::<T>();
    let sigma = safe_min::<T>().mul(epsilon);
    let arithmetic = Arithmetic::<T>::new().expect("shipped format meets the error bound");
    let values = [
        T::ZERO,
        sigma,
        sigma.scale_binary(1),
        safe_min::<T>(),
        safe_min::<T>().scale_binary(1),
        T::ONE.scale_binary(-1),
        T::ONE,
    ];

    for value in values {
        let Some(bounds) = arithmetic.enclose_squared_sum(value, 1) else {
            panic!("finite boundary value must admit a norm interval: {value:?}");
        };
        let Some(key) = arithmetic.pivot_key(bounds, 1) else {
            panic!("finite boundary value must admit a pivot interval: {value:?}");
        };
        assert!(
            key.contains(value),
            "native key {value:?} outside [{:?}, {:?}] for {}",
            key.lower,
            key.upper,
            core::any::type_name::<T>()
        );
    }
}

fn initial_certificate_contains_exact_binary_sum<T: RealScalar>(length: usize) {
    let half = T::ONE.scale_binary(-1);
    let quarter = T::ONE.scale_binary(-2);
    let three_quarters = half.add(quarter);
    let two = T::ONE.scale_binary(1);
    let values: Vec<T> = (0..length)
        .map(|index| match index % 4 {
            0 => T::ONE,
            1 => half,
            2 => three_quarters,
            _ => two,
        })
        .collect();
    let rounded = values
        .iter()
        .copied()
        .fold(T::ZERO, |sum, value| sum.add(value.mul(value)));
    // Quarter-grid inputs have exactly representable squares, so their f64 sum
    // is exact. Widening here forms an independent oracle; the certificate
    // remains in T.
    let exact = values
        .iter()
        .map(|value| {
            let value = value.to_f64();
            value * value
        })
        .sum::<f64>();
    let bounds = Arithmetic::<T>::new()
        .and_then(|arithmetic| arithmetic.enclose_squared_sum(rounded, values.len()))
        .expect("finite supported-format reduction admits a certificate");

    assert!(
        bounds.lower.to_f64() <= exact && exact <= bounds.upper.to_f64(),
        "exact squared sum {exact} outside [{:?}, {:?}] for {}",
        bounds.lower,
        bounds.upper,
        core::any::type_name::<T>()
    );
}

fn reduction_certificate_is_refused<T: RealScalar>(length: usize) -> bool {
    Arithmetic::<T>::new()
        .and_then(|arithmetic| arithmetic.enclose_squared_sum(T::ONE, length))
        .is_none()
}

fn reduction_bound_rejects_unrepresentable_lengths<T: RealScalar>() {
    let arithmetic = Arithmetic::<T>::new().expect("shipped format meets the error bound");
    let precision_bits = arithmetic
        .rho
        .binary_exponent()
        .and_then(|exponent| exponent.checked_neg())
        .and_then(|bits| u32::try_from(bits).ok())
        .expect("machine epsilon is an exact negative power of two");
    let safe_shift = (precision_bits - 3).min(usize::BITS - 3);
    let invalid_shift = precision_bits - 1;
    let safe_length = 1_usize << safe_shift;

    assert!(
        arithmetic.reduction_error(safe_length).is_some(),
        "2·l·ρ = 1/4 must certify for {}",
        core::any::type_name::<T>()
    );
    if let Some(invalid_length) = 1_usize.checked_shl(invalid_shift) {
        assert!(
            arithmetic.reduction_error(invalid_length).is_none(),
            "a reduction with 2·l·ρ ≥ 1 must not certify for {}",
            core::any::type_name::<T>()
        );
    }
    assert!(arithmetic.reduction_error(usize::MAX).is_none());
}

#[test]
fn pivot_key_intervals_cover_normal_and_subnormal_boundaries() {
    use eunomia::{Bf16, F16};

    pivot_keys_remain_enclosed_at_exponent_boundaries::<f64>();
    pivot_keys_remain_enclosed_at_exponent_boundaries::<f32>();
    pivot_keys_remain_enclosed_at_exponent_boundaries::<F16>();
    pivot_keys_remain_enclosed_at_exponent_boundaries::<Bf16>();
}

#[test]
fn initial_squared_sum_certificates_enclose_independent_binary_oracles() {
    use eunomia::{Bf16, F16};

    // A reduction of `l` terms certifies only when `2·l·ρ < 1/2`
    // (`reduction_error` keeps `γ = 2lρ/(1-2lρ)` below one). Bf16's `ρ = 2⁻⁷`
    // bounds `l` below 32, so it takes 16 terms and refuses 32; the wider
    // formats take 32.
    initial_certificate_contains_exact_binary_sum::<f64>(32);
    initial_certificate_contains_exact_binary_sum::<f32>(32);
    initial_certificate_contains_exact_binary_sum::<F16>(32);
    initial_certificate_contains_exact_binary_sum::<Bf16>(16);
    assert!(
        reduction_certificate_is_refused::<Bf16>(32),
        "a 32-term Bf16 reduction must not certify"
    );
}

#[test]
fn reduction_error_rejects_lengths_outside_the_operation_budget() {
    use eunomia::{Bf16, F16};

    reduction_bound_rejects_unrepresentable_lengths::<f64>();
    reduction_bound_rejects_unrepresentable_lengths::<f32>();
    reduction_bound_rejects_unrepresentable_lengths::<F16>();
    reduction_bound_rejects_unrepresentable_lengths::<Bf16>();
}
