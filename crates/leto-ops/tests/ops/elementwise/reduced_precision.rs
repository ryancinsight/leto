//! Reduced-precision (F16/Bf16) Hermes routing tests.

use super::*;

/// Reduced-precision Hermes operations round once from an `f32` intermediate
/// on every backend, as the scalar operators do, so elementwise results are
/// bitwise equal; reductions may reorder, so they are held to `n · u` on the
/// summed magnitudes (Higham ASNA 2nd ed. §4.2, naive summation), and min/max
/// are exact selections.
use leto_ops::domain::strategy::{SimdOperations, SimdStrategy};

trait ReducedPrecisionTestScalar:
    leto_ops::Scalar
    + Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
{
    const UNIT_ROUND: f64;

    fn from_test_f32(value: f32) -> Self;
    fn to_test_f32(self) -> f32;
    fn to_test_bits(self) -> u16;
}

impl ReducedPrecisionTestScalar for eunomia::F16 {
    const UNIT_ROUND: f64 = 1.0 / 2048.0;

    fn from_test_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_test_f32(self) -> f32 {
        Self::to_f32(self)
    }

    fn to_test_bits(self) -> u16 {
        Self::to_bits(self)
    }
}

impl ReducedPrecisionTestScalar for eunomia::Bf16 {
    const UNIT_ROUND: f64 = 1.0 / 256.0;

    fn from_test_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_test_f32(self) -> f32 {
        Self::to_f32(self)
    }

    fn to_test_bits(self) -> u16 {
        Self::to_bits(self)
    }
}

fn assert_reduced_precision_slice_operations<T>()
where
    T: ReducedPrecisionTestScalar,
    SimdStrategy: SimdOperations<T>,
{
    let n = 259usize;
    let a: Vec<T> = (0..n)
        .map(|i| T::from_test_f32(((i * 37 % 101) as f32 - 50.0) / 8.0))
        .collect();
    let b: Vec<T> = (0..n)
        .map(|i| T::from_test_f32(((i * 53 % 97) as f32 - 48.0) / 16.0 + 0.25))
        .collect();
    let mut out = vec![T::from_test_f32(0.0); n];
    type ElementwiseOperation<T> = (
        &'static str,
        fn(&[T], &[T], &mut [T]) -> Result<(), &'static str>,
        fn(T, T) -> T,
    );
    let ops: [ElementwiseOperation<T>; 4] = [
        (
            "add",
            <SimdStrategy as SimdOperations<T>>::add_slice,
            |x, y| x + y,
        ),
        (
            "sub",
            <SimdStrategy as SimdOperations<T>>::sub_slice,
            |x, y| x - y,
        ),
        (
            "mul",
            <SimdStrategy as SimdOperations<T>>::mul_slice,
            |x, y| x * y,
        ),
        (
            "div",
            <SimdStrategy as SimdOperations<T>>::div_slice,
            |x, y| x / y,
        ),
    ];
    for (name, op, scalar) in [ops[0], ops[1], ops[2], ops[3]] {
        op(&a, &b, &mut out).unwrap_or_else(|e| panic!("{name}: reduced precision declined: {e}"));
        for (i, ((&x, &y), &got)) in a.iter().zip(&b).zip(&out).enumerate() {
            let want = scalar(x, y);
            assert_eq!(
                T::to_test_bits(got),
                T::to_test_bits(want),
                "{name}[{i}]: hermes {} vs scalar {}",
                T::to_test_f32(got),
                T::to_test_f32(want)
            );
        }
    }
    let sum =
        <SimdStrategy as SimdOperations<T>>::sum_slice(&a).expect("reduced-precision sum routed");
    let dot = <SimdStrategy as SimdOperations<T>>::dot_slice(&a, &b)
        .expect("reduced-precision dot routed");
    let sum_ref: f64 = a.iter().map(|x| f64::from(T::to_test_f32(*x))).sum();
    let dot_ref: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(T::to_test_f32(*x)) * f64::from(T::to_test_f32(*y)))
        .sum();
    let sum_scale: f64 = a.iter().map(|x| f64::from(T::to_test_f32(*x)).abs()).sum();
    let dot_scale: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f64::from(T::to_test_f32(*x)) * f64::from(T::to_test_f32(*y))).abs())
        .sum();
    assert!(
        (f64::from(T::to_test_f32(sum)) - sum_ref).abs() <= (n as f64) * T::UNIT_ROUND * sum_scale,
        "sum {sum:?} vs {sum_ref}"
    );
    assert!(
        (f64::from(T::to_test_f32(dot)) - dot_ref).abs() <= (n as f64) * T::UNIT_ROUND * dot_scale,
        "dot {dot:?} vs {dot_ref}"
    );
    let min =
        <SimdStrategy as SimdOperations<T>>::min_slice(&a).expect("reduced-precision min routed");
    let max =
        <SimdStrategy as SimdOperations<T>>::max_slice(&a).expect("reduced-precision max routed");
    let min_ref =
        a.iter().copied().fold(
            T::from_test_f32(f32::INFINITY),
            |m, x| if x < m { x } else { m },
        );
    let max_ref = a
        .iter()
        .copied()
        .fold(T::from_test_f32(f32::NEG_INFINITY), |m, x| {
            if x > m {
                x
            } else {
                m
            }
        });
    assert_eq!(T::to_test_bits(min), T::to_test_bits(min_ref));
    assert_eq!(T::to_test_bits(max), T::to_test_bits(max_ref));

    // The public op reaches the same route through the `Scalar` impl.
    let arr_a = Array::from_shape_vec([n], a.clone()).unwrap();
    let arr_b = Array::from_shape_vec([n], b.clone()).unwrap();
    let mut via_public = Array::from_shape_vec([n], vec![T::from_test_f32(0.0); n]).unwrap();
    add(&arr_a.view(), &arr_b.view(), &mut via_public.view_mut()).unwrap();
    for (i, ((&x, &y), &got)) in a
        .iter()
        .zip(&b)
        .zip(via_public.storage().as_slice())
        .enumerate()
    {
        assert_eq!(
            T::to_test_bits(got),
            T::to_test_bits(x + y),
            "public add[{i}]"
        );
    }
}

#[test]
fn f16_slice_operations_route_through_hermes_and_match_scalar_semantics() {
    assert_reduced_precision_slice_operations::<eunomia::F16>();
}

#[test]
fn bf16_slice_operations_route_through_hermes_and_match_scalar_semantics() {
    assert_reduced_precision_slice_operations::<eunomia::Bf16>();
}

