//! Eunomia supertrait extensions every scalar must keep.

use super::*;

#[test]
fn scalar_traits_are_eunomia_extensions() {
    assert_scalar_supertrait::<f32>();
    assert_scalar_supertrait::<f64>();
    assert_scalar_supertrait::<eunomia::F16>();
    assert_scalar_supertrait::<eunomia::Bf16>();
    assert_scalar_supertrait::<i32>();
    assert_scalar_supertrait::<u64>();
    assert_scalar_supertrait::<isize>();
    assert_scalar_supertrait::<usize>();

    assert_real_supertrait::<f32>();
    assert_real_supertrait::<f64>();
    assert_real_supertrait::<eunomia::F16>();
    assert_real_supertrait::<eunomia::Bf16>();

    assert_eq!(scalar_count::<f64>(3), Ok(3.0));
    assert_eq!(scalar_count::<isize>(5), Ok(5_isize));
    assert_eq!(scalar_count::<usize>(6), Ok(6_usize));
    assert_eq!(
        scalar_count::<eunomia::F16>(4),
        Ok(eunomia::F16::from_f32(4.0))
    );
    assert_eq!(
        scalar_count::<eunomia::Bf16>(4),
        Ok(eunomia::Bf16::from_f32(4.0))
    );
    assert_eq!(
        scalar_count::<i8>(128),
        Err(eunomia::CountRangeError::new::<i8>(128))
    );
}
