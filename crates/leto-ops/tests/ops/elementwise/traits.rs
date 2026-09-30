//! Super-trait conformance: `Scalar`/`RealScalar` extend the eunomia element traits.

fn assert_scalar_supertrait<T>()
where
    T: leto_ops::Scalar + eunomia::NumericElement,
{
}

fn assert_real_supertrait<T>()
where
    T: leto_ops::RealScalar + eunomia::FloatElement,
{
}

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

    assert_eq!(<f64 as leto_ops::Scalar>::from_usize(3), 3.0);
    assert_eq!(<isize as leto_ops::Scalar>::from_usize(5), 5_isize);
    assert_eq!(<usize as leto_ops::Scalar>::from_usize(6), 6_usize);
    assert_eq!(
        <eunomia::F16 as leto_ops::Scalar>::from_usize(4),
        eunomia::F16::from_f32(4.0)
    );
    assert_eq!(
        <eunomia::Bf16 as leto_ops::Scalar>::from_usize(4),
        eunomia::Bf16::from_f32(4.0)
    );
}

