//! Exact finite inverse range regressions and preserved legacy boundaries.
use eunomia::RealField;
use leto::geometry::Quaternion;

fn quaternion<T: RealField>(a: [T; 4]) -> Quaternion<T> {
    Quaternion::new(a[0], a[1], a[2], a[3])
}
fn assert_inverse<T: RealField>(magnitude: T, reciprocal: T) {
    for positive in [true, false] {
        let (value, inverse) = if positive {
            (magnitude, reciprocal)
        } else {
            (-magnitude, -reciprocal)
        };
        for axis in 0..4 {
            let mut input = [T::ZERO; 4];
            input[axis] = value;
            let mut expected = [T::ZERO; 4];
            expected[axis] = if axis == 0 { inverse } else { -inverse };
            let q = quaternion(input);
            let actual = q
                .try_inverse()
                .expect("nonzero finite input has a representable inverse");
            assert_eq!(
                actual,
                quaternion(expected),
                "inverse of finite basis quaternion {q:?}"
            );
            assert_eq!(q * actual, Quaternion::identity());
            assert_eq!(actual * q, Quaternion::identity());
        }
        let q = Quaternion::new(value, value, value, value);
        let quarter = T::from_f64(0.25);
        let scaled = inverse * quarter;
        let actual = q
            .try_inverse()
            .expect("nonzero finite mixed input has a representable inverse");
        assert_eq!(actual, Quaternion::new(scaled, -scaled, -scaled, -scaled));
        assert_eq!(q * actual, Quaternion::identity());
        assert_eq!(actual * q, Quaternion::identity());
    }
}
#[test]
fn large_binary32() {
    assert_inverse(
        f32::from_bits((127 + 80) << 23),
        f32::from_bits((127 - 80) << 23),
    );
}
#[test]
fn small_binary32() {
    assert_inverse(
        f32::from_bits((127 - 80) << 23),
        f32::from_bits((127 + 80) << 23),
    );
}
#[test]
fn large_binary64() {
    assert_inverse(
        f64::from_bits((1023 + 600) << 52),
        f64::from_bits((1023 - 600) << 52),
    );
}
#[test]
fn small_binary64() {
    assert_inverse(
        f64::from_bits((1023 - 600) << 52),
        f64::from_bits((1023 + 600) << 52),
    );
}
#[test]
fn ordinary_binary32() {
    assert_inverse(2.0_f32, 0.5);
    assert_inverse(0.5_f32, 2.0);
}
#[test]
fn ordinary_binary64() {
    assert_inverse(2.0_f64, 0.5);
    assert_inverse(0.5_f64, 2.0);
}
#[test]
fn identity_and_zero_binary32() {
    let q = Quaternion::<f32>::identity();
    assert_eq!(q.try_inverse(), Some(q));
    let zero = Quaternion::new(0.0_f32, 0.0, 0.0, 0.0);
    assert_eq!(zero.norm_squared(), 0.0);
    assert_eq!(zero.try_inverse(), None);
}
#[test]
fn identity_and_zero_binary64() {
    let q = Quaternion::<f64>::identity();
    assert_eq!(q.try_inverse(), Some(q));
    let zero = Quaternion::new(0.0_f64, 0.0, 0.0, 0.0);
    assert_eq!(zero.norm_squared(), 0.0);
    assert_eq!(zero.try_inverse(), None);
}

fn assert_mixed_range<T: RealField>(large: T, small: T, tiny: T, one: T) {
    let input = Quaternion::new(small, tiny, T::ZERO, -T::ZERO);
    // tiny = small^2 * one for these exact power-of-two fixtures.
    let inverse = input
        .try_inverse()
        .expect("mixed finite inverse is representable");
    assert_eq!(inverse.w, large);
    assert_eq!(inverse.x, -one);
    assert_eq!(inverse.y, T::ZERO);
    assert_eq!(inverse.z, T::ZERO);
    assert_eq!(input * inverse, Quaternion::identity());
    assert_eq!(inverse * input, Quaternion::identity());
}
#[test]
fn mixed_subnormal_input_binary32() {
    assert_mixed_range(
        f32::from_bits((127 + 80) << 23),
        f32::from_bits((127 - 80) << 23),
        f32::from_bits(1 << 9),
        f32::from_bits((127 + 20) << 23),
    );
}
#[test]
fn mixed_subnormal_input_binary64() {
    assert_mixed_range(
        f64::from_bits((1023 + 600) << 52),
        f64::from_bits((1023 - 600) << 52),
        f64::from_bits(1 << 24),
        f64::from_bits((1023 + 150) << 52),
    );
}
#[test]
fn subnormal_squared_norm_binary32() {
    assert_inverse(
        f32::from_bits((127 - 65) << 23),
        f32::from_bits((127 + 65) << 23),
    );
}
#[test]
fn subnormal_squared_norm_binary64() {
    assert_inverse(
        f64::from_bits((1023 - 513) << 52),
        f64::from_bits((1023 + 513) << 52),
    );
}
fn quartet<T: RealField>(a: T, expected: T) {
    let q = Quaternion::new(a, a, a, a);
    let inverse = q.try_inverse().expect("quartet inverse is finite");
    assert_eq!(
        inverse,
        Quaternion::new(expected, -expected, -expected, -expected)
    );
    assert_eq!(q * inverse, Quaternion::identity());
    assert_eq!(inverse * q, Quaternion::identity());
}
#[test]
fn tiny_quartet_has_finite_inverse_binary32() {
    quartet(f32::from_bits(1 << 20), f32::from_bits((127 + 127) << 23));
}
#[test]
fn tiny_quartet_has_finite_inverse_binary64() {
    quartet(f64::from_bits(1 << 49), f64::from_bits((1023 + 1023) << 52));
}
#[test]
fn wrapped_large_binary32() {
    use eunomia::FloatElement;
    assert_inverse(
        eunomia::F32::from_f32(f32::from_bits((127 + 80) << 23)),
        eunomia::F32::from_f32(f32::from_bits((127 - 80) << 23)),
    );
}
#[test]
fn wrapped_small_binary32() {
    use eunomia::FloatElement;
    assert_inverse(
        eunomia::F32::from_f32(f32::from_bits((127 - 80) << 23)),
        eunomia::F32::from_f32(f32::from_bits((127 + 80) << 23)),
    );
}
#[test]
fn wrapped_large_binary64() {
    use eunomia::FloatElement;
    assert_inverse(
        eunomia::F64::from_f64(f64::from_bits((1023 + 600) << 52)),
        eunomia::F64::from_f64(f64::from_bits((1023 - 600) << 52)),
    );
}
#[test]
fn wrapped_small_binary64() {
    use eunomia::FloatElement;
    assert_inverse(
        eunomia::F64::from_f64(f64::from_bits((1023 - 600) << 52)),
        eunomia::F64::from_f64(f64::from_bits((1023 + 600) << 52)),
    );
}
fn preserve_nonfinite<T: RealField>() {
    let q = Quaternion::new(T::INFINITY, T::ZERO, T::ONE, -T::ONE);
    let inverse = q
        .try_inverse()
        .expect("legacy nonfinite input returns Some");
    assert!(inverse.w.is_nan());
    assert_eq!(inverse.x, T::ZERO);
    assert_eq!(inverse.y, T::ZERO);
    assert_eq!(inverse.z, T::ZERO);
    let q = Quaternion::new(T::NAN, T::ONE, T::ZERO, T::ZERO);
    let inverse = q.try_inverse().expect("legacy NaN input returns Some");
    assert!([inverse.w, inverse.x, inverse.y, inverse.z]
        .iter()
        .all(|x| x.is_nan()));
}
#[test]
fn nonfinite_boundary_binary32() {
    preserve_nonfinite::<f32>();
}
#[test]
fn nonfinite_boundary_binary64() {
    preserve_nonfinite::<f64>();
}
