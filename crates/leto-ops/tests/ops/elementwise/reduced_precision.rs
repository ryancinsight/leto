//! Reduced-precision scalar elementwise behavior.

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
    let factor = b[7];
    let mut scaled = a.clone();
    <SimdStrategy as SimdOperations<T>>::scale_slice(&mut scaled, factor)
        .unwrap_or_else(|e| panic!("scale: reduced precision declined: {e}"));
    for (i, (&x, &got)) in a.iter().zip(&scaled).enumerate() {
        let want = x * factor;
        assert_eq!(
            T::to_test_bits(got),
            T::to_test_bits(want),
            "scale[{i}]: hermes {} vs scalar {}",
            T::to_test_f32(got),
            T::to_test_f32(want)
        );
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

#[test]
fn scalar_scale_slice_matches_scalar_reference_bitwise() {
    // f32/f64 route through the SIMD strategy; i32 takes the scalar default.
    // Length 259 covers SIMD tails; length 3 covers the sub-lane path.
    for len in [3usize, 259] {
        let data: Vec<f32> = (0..len)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 8.0)
            .collect();
        let mut got = data.clone();
        leto_ops::Scalar::scale_slice(&mut got, 1.5);
        for (i, (&x, &g)) in data.iter().zip(&got).enumerate() {
            assert_eq!(g.to_bits(), (x * 1.5f32).to_bits(), "f32[{len}][{i}]");
        }
        let data: Vec<f64> = (0..len)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 8.0)
            .collect();
        let mut got = data.clone();
        leto_ops::Scalar::scale_slice(&mut got, 1.5);
        for (i, (&x, &g)) in data.iter().zip(&got).enumerate() {
            assert_eq!(g.to_bits(), (x * 1.5f64).to_bits(), "f64[{len}][{i}]");
        }
        let data: Vec<i32> = (0..len as i32).map(|i| (i * 37 % 101) - 50).collect();
        let mut got = data.clone();
        leto_ops::Scalar::scale_slice(&mut got, 3);
        for (i, (&x, &g)) in data.iter().zip(&got).enumerate() {
            assert_eq!(g, x * 3, "i32[{len}][{i}]");
        }
    }
}

/// Values `0.5, 1.0, 1.5, …`, built without a cast so the test carries no
/// precision-losing conversion.
fn ramp(len: usize, step: f64) -> Vec<f64> {
    let mut value = 0.0;
    (0..len)
        .map(|_| {
            value += step;
            value
        })
        .collect()
}

/// Above the element-count gate of the unary paths (`PARALLEL_THRESHOLD`), so
/// `map_into` and `map_inplace` take their parallel dense route.
const PARALLEL_LEN: usize = 1 << 17;

#[test]
fn parallel_unary_dense_routes_match_the_serial_result() {
    let source_values = ramp(PARALLEL_LEN, 0.5);
    let source = Array::from_shape_vec([PARALLEL_LEN], source_values.clone()).unwrap();

    let mut mapped = Array::from_shape_vec([PARALLEL_LEN], vec![0.0_f64; PARALLEL_LEN]).unwrap();
    map_into(&source.view(), &mut mapped.view_mut(), |x| {
        x.mul_add(3.0, -1.0)
    })
    .unwrap();
    for (index, (&got, &x)) in mapped
        .storage()
        .as_slice()
        .iter()
        .zip(&source_values)
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            x.mul_add(3.0, -1.0).to_bits(),
            "map_into[{index}]"
        );
    }

    let mut in_place = Array::from_shape_vec([PARALLEL_LEN], source_values.clone()).unwrap();
    map_inplace(&mut in_place.view_mut(), |x| x.mul_add(-2.0, 0.25)).unwrap();
    for (index, (&got, &x)) in in_place
        .storage()
        .as_slice()
        .iter()
        .zip(&source_values)
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            x.mul_add(-2.0, 0.25).to_bits(),
            "map_inplace[{index}]"
        );
    }
}

#[test]
fn parallel_binary_dense_route_matches_the_serial_result() {
    // The binary gate is a working set past the last-level cache, so the
    // length comes from the probed geometry rather than a fixed count: three
    // operands of `len` doubles, one page group past the capacity.
    let capacity = leto_ops::cached_cache_geometry().l3_bytes();
    let len = capacity / (3 * core::mem::size_of::<f64>()) + (1 << 16);

    let lhs = Array::from_shape_vec([len], ramp(len, 0.5)).unwrap();
    let rhs = Array::from_shape_vec([len], ramp(len, -0.25)).unwrap();
    let mut out = Array::from_shape_vec([len], vec![0.0_f64; len]).unwrap();
    add(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    // Every element of the output is the sum of the two operands at the same
    // position: a dropped, doubled or shifted task run shows up here.
    for (index, ((&got, &x), &y)) in out
        .storage()
        .as_slice()
        .iter()
        .zip(lhs.storage().as_slice())
        .zip(rhs.storage().as_slice())
        .enumerate()
    {
        assert_eq!(got.to_bits(), (x + y).to_bits(), "add[{index}]");
    }
}

/// Rows and columns whose product puts three `f64` operands past the probed
/// last-level cache, which is the gate the strided elementwise paths take.
fn strided_parallel_shape(operands: usize) -> (usize, usize) {
    let capacity = leto_ops::cached_cache_geometry().l3_bytes();
    let elements = capacity / (operands * core::mem::size_of::<f64>()) + (1 << 16);
    let columns = 64;
    (elements.div_ceil(columns), columns)
}

#[test]
fn parallel_strided_binary_map_matches_the_serial_result() {
    // A transposed view walks its last axis by a row stride, which takes the
    // tiled block arm; the output is contiguous in the walked order.
    let (rows, columns) = strided_parallel_shape(3);
    let len = rows * columns;
    let lhs = Array::from_shape_vec([rows, columns], ramp(len, 0.5)).unwrap();
    let rhs = Array::from_shape_vec([rows, columns], ramp(len, -0.25)).unwrap();
    let mut out = Array::from_shape_vec([columns, rows], vec![0.0_f64; len]).unwrap();

    add(
        &lhs.transpose([1, 0]).unwrap(),
        &rhs.transpose([1, 0]).unwrap(),
        &mut out.view_mut(),
    )
    .unwrap();

    // Output element (column, row) is the sum at (row, column) of both inputs:
    // a dropped, doubled or shifted task run shows up as a mismatch here.
    let (lhs_values, rhs_values) = (lhs.storage().as_slice(), rhs.storage().as_slice());
    for (index, &got) in out.storage().as_slice().iter().enumerate() {
        let (column, row) = (index / rows, index % rows);
        let source = row * columns + column;
        assert_eq!(
            got.to_bits(),
            (lhs_values[source] + rhs_values[source]).to_bits(),
            "add[{column}, {row}]"
        );
    }
}

#[test]
fn parallel_strided_unary_map_matches_the_serial_result() {
    let (rows, columns) = strided_parallel_shape(2);
    let len = rows * columns;
    let source = Array::from_shape_vec([rows, columns], ramp(len, 0.5)).unwrap();
    let mut out = Array::from_shape_vec([columns, rows], vec![0.0_f64; len]).unwrap();

    map_into(
        &source.transpose([1, 0]).unwrap(),
        &mut out.view_mut(),
        |x| x.mul_add(3.0, -1.0),
    )
    .unwrap();

    let values = source.storage().as_slice();
    for (index, &got) in out.storage().as_slice().iter().enumerate() {
        let (column, row) = (index / rows, index % rows);
        let x = values[row * columns + column];
        assert_eq!(
            got.to_bits(),
            x.mul_add(3.0, -1.0).to_bits(),
            "map_into[{column}, {row}]"
        );
    }
}

#[test]
fn parallel_row_walk_arms_match_the_serial_result() {
    // A padded source sliced back to the output width keeps its last-axis
    // stride at one element while its row stride differs from the output's,
    // which is the row-walk arm rather than the tiled block arm.
    const PAD: usize = 3;
    let (rows, columns) = strided_parallel_shape(3);
    let padded_columns = columns + PAD;
    let padded_len = rows * padded_columns;
    let len = rows * columns;

    let lhs = Array::from_shape_vec([rows, padded_columns], ramp(padded_len, 0.5)).unwrap();
    let rhs = Array::from_shape_vec([rows, padded_columns], ramp(padded_len, -0.25)).unwrap();
    let lhs_view = lhs.slice(&[(0, rows, 1), (0, columns, 1)]).unwrap();
    let rhs_view = rhs.slice(&[(0, rows, 1), (0, columns, 1)]).unwrap();

    let mut out = Array::from_shape_vec([rows, columns], vec![0.0_f64; len]).unwrap();
    add(&lhs_view, &rhs_view, &mut out.view_mut()).unwrap();

    let (lhs_values, rhs_values) = (lhs.storage().as_slice(), rhs.storage().as_slice());
    for (index, &got) in out.storage().as_slice().iter().enumerate() {
        let (row, column) = (index / columns, index % columns);
        let source = row * padded_columns + column;
        assert_eq!(
            got.to_bits(),
            (lhs_values[source] + rhs_values[source]).to_bits(),
            "add[{row}, {column}]"
        );
    }

    // The unary row walk, on the same shape.
    let mut mapped = Array::from_shape_vec([rows, columns], vec![0.0_f64; len]).unwrap();
    map_into(&lhs_view, &mut mapped.view_mut(), |x| x.mul_add(3.0, -1.0)).unwrap();
    for (index, &got) in mapped.storage().as_slice().iter().enumerate() {
        let (row, column) = (index / columns, index % columns);
        let x = lhs_values[row * padded_columns + column];
        assert_eq!(
            got.to_bits(),
            x.mul_add(3.0, -1.0).to_bits(),
            "map_into[{row}, {column}]"
        );
    }
}
