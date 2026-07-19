//! Value-semantic + ndarray-differential tests for elementwise operators (ADR 0004).

use leto::{Array2, Storage};
use ndarray::Array2 as NdArray2;

fn assert_reduced_precision_scalar_arithmetic<T>()
where
    T: leto::ScalarOperand
        + eunomia::FloatElement
        + core::fmt::Debug
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    let value = |input| <T as eunomia::FloatElement>::from_f32(input);
    let array = Array2::from_shape_vec([1, 2], vec![value(1.0), value(2.0)]).unwrap();

    assert_eq!(
        (&array + value(2.0)).storage().as_slice(),
        &[value(3.0), value(4.0)]
    );
    assert_eq!(
        (&array * value(2.0)).storage().as_slice(),
        &[value(2.0), value(4.0)]
    );
}

fn seq(len: usize, scale: f64, bias: f64) -> Vec<f64> {
    (0..len).map(|i| i as f64 * scale + bias).collect()
}

fn leto_pair() -> (Array2<f64>, Array2<f64>) {
    let a = Array2::from_shape_vec([3, 4], seq(12, 0.5, 1.0)).unwrap();
    let b = Array2::from_shape_vec([3, 4], seq(12, 0.25, 2.0)).unwrap();
    (a, b)
}

fn nd_pair() -> (NdArray2<f64>, NdArray2<f64>) {
    let a = NdArray2::from_shape_vec((3, 4), seq(12, 0.5, 1.0)).unwrap();
    let b = NdArray2::from_shape_vec((3, 4), seq(12, 0.25, 2.0)).unwrap();
    (a, b)
}

#[test]
fn array_array_operators_match_ndarray() {
    let (la, lb) = leto_pair();
    let (na, nb) = nd_pair();

    assert_eq!(
        (&la + &lb).storage().as_slice(),
        (&na + &nb).as_slice().unwrap()
    );
    assert_eq!(
        (&la - &lb).storage().as_slice(),
        (&na - &nb).as_slice().unwrap()
    );
    assert_eq!(
        (&la * &lb).storage().as_slice(),
        (&na * &nb).as_slice().unwrap()
    );
    assert_eq!(
        (&la / &lb).storage().as_slice(),
        (&na / &nb).as_slice().unwrap()
    );
}

#[test]
fn mul_is_elementwise_not_matmul() {
    // `*` is the Hadamard product: shape is preserved and each element is the
    // product of the corresponding inputs (ndarray semantics, ADR 0004).
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = &a * &b;
    assert_eq!(c.shape(), [2, 2]);
    assert_eq!(c.storage().as_slice(), &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn scalar_operators_match_ndarray() {
    let (la, _) = leto_pair();
    let (na, _) = nd_pair();

    assert_eq!(
        (&la + 2.0).storage().as_slice(),
        (&na + 2.0).as_slice().unwrap()
    );
    assert_eq!(
        (&la - 1.5).storage().as_slice(),
        (&na - 1.5).as_slice().unwrap()
    );
    assert_eq!(
        (&la * 3.0).storage().as_slice(),
        (&na * 3.0).as_slice().unwrap()
    );
    assert_eq!(
        (&la / 2.0).storage().as_slice(),
        (&na / 2.0).as_slice().unwrap()
    );
}

#[test]
fn eunomia_reduced_precision_types_are_scalar_operands() {
    assert_reduced_precision_scalar_arithmetic::<eunomia::F16>();
    assert_reduced_precision_scalar_arithmetic::<eunomia::Bf16>();
}

#[test]
fn neg_matches_ndarray() {
    let (la, _) = leto_pair();
    let (na, _) = nd_pair();
    assert_eq!((-&la).storage().as_slice(), (-&na).as_slice().unwrap());
}

#[test]
fn operators_compose() {
    // `-(&a * 2.0 + &b)` exercises chaining of scalar, binary, and unary ops.
    let (la, lb) = leto_pair();
    let (na, nb) = nd_pair();
    let leto = -&(&(&la * 2.0) + &lb);
    let nd = -&(&(&na * 2.0) + &nb);
    assert_eq!(leto.storage().as_slice(), nd.as_slice().unwrap());
}

#[test]
fn integer_arrays_support_operators() {
    // Operators are generic over the element type, not f64-specific.
    let a = Array2::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let b = Array2::from_shape_vec([2, 2], vec![10_i32, 20, 30, 40]).unwrap();
    assert_eq!((&a + &b).storage().as_slice(), &[11, 22, 33, 44]);
    assert_eq!((&a * 2).storage().as_slice(), &[2, 4, 6, 8]);
}

#[test]
#[should_panic(expected = "equal shapes")]
fn shape_mismatch_panics() {
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let _ = &a + &b;
}
