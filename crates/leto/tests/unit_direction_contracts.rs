//! Checked unit input invariants and the existing IEEE threshold comparison.
use eunomia::RealField;
use leto::geometry::{Unit, Vector};

fn thresholds<T: RealField>() {
    let z = T::ZERO;
    let o = T::ONE;
    let v = Vector::from_array([o, z]);
    for threshold in [z, -o, -T::INFINITY] {
        assert_eq!(
            Unit::try_new(v, threshold)
                .expect("nonzero unit vector exceeds threshold")
                .into_inner(),
            v
        );
    }
    for threshold in [o, o + T::EPSILON, T::INFINITY, T::NAN] {
        assert!(Unit::try_new(v, threshold).is_none());
    }
    assert_eq!(
        Unit::try_new(v, o - T::EPSILON)
            .expect("nonzero unit vector exceeds threshold")
            .into_inner(),
        v
    );
    for threshold in [z, -o, -T::INFINITY] {
        assert!(Unit::try_new(Vector::from_array([z, z]), threshold).is_none());
        assert!(Unit::try_new(Vector::<T, 0>::from_array([]), threshold).is_none());
        for invalid in [T::INFINITY, -T::INFINITY, T::NAN] {
            assert!(Unit::try_new(Vector::from_array([invalid, o]), threshold).is_none());
        }
    }
    assert!(Vector::from_array([z, z])
        .normalize()
        .data
        .iter()
        .all(|v| v.is_nan()));
}

#[test]
fn checked_inputs_and_threshold_comparison() {
    thresholds::<f32>();
    thresholds::<f64>();
    thresholds::<eunomia::F32>();
    thresholds::<eunomia::F64>();
}
