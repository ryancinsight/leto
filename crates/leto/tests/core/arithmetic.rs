//! Value-semantic tests for elementwise operators (ADR 0004).

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array2, Storage};

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

/// Reference elementwise add using plain Rust loops (replaces external oracle).
fn ref_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn ref_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn ref_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn ref_div(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

fn ref_scalar_add(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x + s).collect()
}

fn ref_scalar_sub(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x - s).collect()
}

fn ref_scalar_mul(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn ref_scalar_div(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x / s).collect()
}

fn ref_neg(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| -x).collect()
}

#[test]
fn array_array_operators_match_reference() {
    let (la, lb) = leto_pair();
    let a_slice = la.storage().as_slice();
    let b_slice = lb.storage().as_slice();

    assert_eq!((&la + &lb).storage().as_slice(), ref_add(a_slice, b_slice));
    assert_eq!((&la - &lb).storage().as_slice(), ref_sub(a_slice, b_slice));
    assert_eq!((&la * &lb).storage().as_slice(), ref_mul(a_slice, b_slice));
    assert_eq!((&la / &lb).storage().as_slice(), ref_div(a_slice, b_slice));
}

#[test]
fn mul_is_elementwise_not_matmul() {
    //  is the Hadamard product: shape is preserved and each element is the
    // product of the corresponding inputs (ADR 0004).
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = &a * &b;
    assert_eq!(c.shape(), [2, 2]);
    assert_eq!(c.storage().as_slice(), &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn scalar_operators_match_reference() {
    let (la, _) = leto_pair();
    let a_slice = la.storage().as_slice();

    assert_eq!(
        (&la + 2.0).storage().as_slice(),
        ref_scalar_add(a_slice, 2.0)
    );
    assert_eq!(
        (&la - 1.5).storage().as_slice(),
        ref_scalar_sub(a_slice, 1.5)
    );
    assert_eq!(
        (&la * 3.0).storage().as_slice(),
        ref_scalar_mul(a_slice, 3.0)
    );
    assert_eq!(
        (&la / 2.0).storage().as_slice(),
        ref_scalar_div(a_slice, 2.0)
    );
}

#[test]
fn eunomia_reduced_precision_types_are_scalar_operands() {
    assert_reduced_precision_scalar_arithmetic::<eunomia::F16>();
    assert_reduced_precision_scalar_arithmetic::<eunomia::Bf16>();
}

#[test]
fn neg_matches_reference() {
    let (la, _) = leto_pair();
    assert_eq!(
        (-&la).storage().as_slice(),
        ref_neg(la.storage().as_slice())
    );
}

#[test]
fn operators_compose() {
    //  exercises chaining of scalar, binary, and unary ops.
    let (la, lb) = leto_pair();
    let a_slice = la.storage().as_slice();
    let b_slice = lb.storage().as_slice();
    let leto = -&(&(&la * 2.0) + &lb);
    let expected = ref_neg(&ref_add(&ref_scalar_mul(a_slice, 2.0), b_slice));
    assert_eq!(leto.storage().as_slice(), &expected);
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

#[test]
#[should_panic(expected = "equal shapes")]
fn owned_lhs_shape_mismatch_panics() {
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let _ = a + &b;
}

#[test]
fn owned_lhs_operators_match_borrowed_lhs() {
    // The owned receiver writes into its own allocation; the observable value
    // must equal the allocating borrowed form element for element.
    let (la, lb) = leto_pair();
    let a_slice = la.storage().as_slice().to_vec();
    let b_slice = lb.storage().as_slice().to_vec();

    assert_eq!(
        (la.clone() + &lb).storage().as_slice(),
        ref_add(&a_slice, &b_slice)
    );
    assert_eq!(
        (la.clone() - &lb).storage().as_slice(),
        ref_sub(&a_slice, &b_slice)
    );
    assert_eq!(
        (la.clone() * &lb).storage().as_slice(),
        ref_mul(&a_slice, &b_slice)
    );
    assert_eq!(
        (la.clone() / &lb).storage().as_slice(),
        ref_div(&a_slice, &b_slice)
    );

    // Both operands owned: the left allocation is kept, the right dropped.
    assert_eq!(
        (la.clone() + lb.clone()).storage().as_slice(),
        ref_add(&a_slice, &b_slice)
    );
}

#[test]
fn owned_lhs_scalar_and_neg_match_borrowed_lhs() {
    let (la, _) = leto_pair();
    let a_slice = la.storage().as_slice().to_vec();

    assert_eq!(
        (la.clone() + 2.0).storage().as_slice(),
        ref_scalar_add(&a_slice, 2.0)
    );
    assert_eq!(
        (la.clone() - 1.5).storage().as_slice(),
        ref_scalar_sub(&a_slice, 1.5)
    );
    assert_eq!(
        (la.clone() * 3.0).storage().as_slice(),
        ref_scalar_mul(&a_slice, 3.0)
    );
    assert_eq!(
        (la.clone() / 2.0).storage().as_slice(),
        ref_scalar_div(&a_slice, 2.0)
    );
    assert_eq!((-la).storage().as_slice(), ref_neg(&a_slice));
}

#[test]
fn owned_lhs_reuses_the_left_allocation() {
    // The buffer-reuse contract, observed directly: the owned operator returns
    // the same heap allocation it consumed, so a chain adds no allocation.
    let (la, lb) = leto_pair();
    let before = la.storage().as_slice().as_ptr();
    let sum = la + &lb;
    assert_eq!(
        sum.storage().as_slice().as_ptr(),
        before,
        "owned lhs must write into its existing allocation"
    );
}

#[test]
fn owned_lhs_accepts_a_strided_right_operand() {
    // Right operand is a transposed (non-C-contiguous) source, so the owned
    // route walks the logical-order element iterator rather than a slice.
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Array2::from_shape_vec([3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let b_t = b.transpose([1, 0]).unwrap();
    let expected: Vec<f64> = a
        .storage()
        .as_slice()
        .iter()
        .zip(b_t.iter())
        .map(|(x, y)| x + y)
        .collect();
    let owned = a + &b_t.to_contiguous();
    assert_eq!(owned.storage().as_slice(), &expected);
}

#[test]
fn chained_operators_compose_without_intermediate_refs() {
    // The owned receiver makes the natural chain syntax type-check: the middle
    // term is an owned temporary reused in place rather than re-borrowed.
    let (la, lb) = leto_pair();
    let lc = Array2::from_shape_vec([3, 4], seq(12, 0.125, 0.5)).unwrap();
    let a_slice = la.storage().as_slice().to_vec();
    let b_slice = lb.storage().as_slice().to_vec();
    let c_slice = lc.storage().as_slice().to_vec();

    let chained = &la + &lb + &lc;
    let expected = ref_add(&ref_add(&a_slice, &b_slice), &c_slice);
    assert_eq!(chained.storage().as_slice(), &expected);
}
