//! Analytic unit directions and the checked Unit constructor's invariant.
use eunomia::{FloatElement, NumericElement, RealField};
use leto::geometry::{Unit, Vector3};

fn basis_direction<T: RealField>(smallest: T) {
    let v = Vector3::new(smallest, T::ZERO, T::ZERO);
    let normalized = v.normalize();
    assert_eq!(normalized.data, [T::ONE, T::ZERO, T::ZERO]);
    assert_eq!(normalized.norm(), T::ONE);
}

fn symmetric_direction<T: RealField>(largest: T) {
    let v = Vector3::new(largest, largest, largest);
    let normalized = v.normalize();
    let three = T::ONE + T::ONE + T::ONE;
    let expected = three.sqrt().recip();
    for actual in normalized.data {
        assert!((actual - expected).abs() <= T::EPSILON * (T::ONE + T::ONE));
    }
    assert!((normalized.norm() - T::ONE).abs() <= T::EPSILON * three);
}

fn checked_unit<T: RealField>(v: Vector3<T>) {
    let unit = Unit::try_new(v, T::ZERO).expect("finite nonzero vector");
    let norm = unit.as_vector().norm();
    assert!(norm.is_finite(), "checked Unit contains a nonfinite norm");
    assert!((norm - T::ONE).abs() <= T::EPSILON * (T::ONE + T::ONE + T::ONE));
}

fn reject_infinite<T: RealField>() {
    assert!(Unit::try_new(Vector3::new(T::INFINITY, T::ONE, T::ZERO), T::ZERO).is_none());
}

fn ordinary_direction<T: RealField>() {
    let three = T::ONE + T::ONE + T::ONE;
    let four = three + T::ONE;
    let five = four + T::ONE;
    let v = Vector3::new(three, four, T::ZERO);
    let normalized = v.normalize();
    for (actual, expected) in normalized
        .data
        .into_iter()
        .zip([three / five, four / five, T::ZERO])
    {
        assert!((actual - expected).abs() <= T::EPSILON);
    }
    let unit = Unit::try_new(v, T::ZERO).expect("positive finite length");
    assert!((unit.as_vector().norm() - T::ONE).abs() <= T::EPSILON);
}

#[test]
fn smallest_basis_native32() {
    basis_direction(f32::from_bits(1));
}

#[test]
fn largest_equal_components_native32() {
    symmetric_direction(f32::MAX);
}

#[test]
fn checked_smallest_unit_native32() {
    checked_unit(Vector3::new(f32::from_bits(1), <f32>::ZERO, <f32>::ZERO));
}

#[test]
fn checked_largest_unit_native32() {
    checked_unit(Vector3::new(f32::MAX, f32::MAX, f32::MAX));
}

#[test]
fn checked_unit_nonfinite_native32() {
    reject_infinite::<f32>();
}

#[test]
fn ordinary_control_native32() {
    ordinary_direction::<f32>();
}

#[test]
fn smallest_basis_native64() {
    basis_direction(f64::from_bits(1));
}

#[test]
fn largest_equal_components_native64() {
    symmetric_direction(f64::MAX);
}

#[test]
fn checked_smallest_unit_native64() {
    checked_unit(Vector3::new(f64::from_bits(1), <f64>::ZERO, <f64>::ZERO));
}

#[test]
fn checked_largest_unit_native64() {
    checked_unit(Vector3::new(f64::MAX, f64::MAX, f64::MAX));
}

#[test]
fn checked_unit_nonfinite_native64() {
    reject_infinite::<f64>();
}

#[test]
fn ordinary_control_native64() {
    ordinary_direction::<f64>();
}

#[test]
fn smallest_basis_wrapped32() {
    basis_direction(eunomia::F32::from_f32(f32::from_bits(1)));
}

#[test]
fn largest_equal_components_wrapped32() {
    symmetric_direction(eunomia::F32::from_f32(f32::MAX));
}

#[test]
fn checked_smallest_unit_wrapped32() {
    checked_unit(Vector3::new(
        eunomia::F32::from_f32(f32::from_bits(1)),
        <eunomia::F32>::ZERO,
        <eunomia::F32>::ZERO,
    ));
}

#[test]
fn checked_largest_unit_wrapped32() {
    checked_unit(Vector3::new(
        eunomia::F32::from_f32(f32::MAX),
        eunomia::F32::from_f32(f32::MAX),
        eunomia::F32::from_f32(f32::MAX),
    ));
}

#[test]
fn checked_unit_nonfinite_wrapped32() {
    reject_infinite::<eunomia::F32>();
}

#[test]
fn ordinary_control_wrapped32() {
    ordinary_direction::<eunomia::F32>();
}

#[test]
fn smallest_basis_wrapped64() {
    basis_direction(eunomia::F64::from_f64(f64::from_bits(1)));
}

#[test]
fn largest_equal_components_wrapped64() {
    symmetric_direction(eunomia::F64::from_f64(f64::MAX));
}

#[test]
fn checked_smallest_unit_wrapped64() {
    checked_unit(Vector3::new(
        eunomia::F64::from_f64(f64::from_bits(1)),
        <eunomia::F64>::ZERO,
        <eunomia::F64>::ZERO,
    ));
}

#[test]
fn checked_largest_unit_wrapped64() {
    checked_unit(Vector3::new(
        eunomia::F64::from_f64(f64::MAX),
        eunomia::F64::from_f64(f64::MAX),
        eunomia::F64::from_f64(f64::MAX),
    ));
}

#[test]
fn checked_unit_nonfinite_wrapped64() {
    reject_infinite::<eunomia::F64>();
}

#[test]
fn ordinary_control_wrapped64() {
    ordinary_direction::<eunomia::F64>();
}
