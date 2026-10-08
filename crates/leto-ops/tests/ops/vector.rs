#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, SliceArg, Storage};
use leto_ops::cross_into;

/// The hephaestus cross-product conformance fixture, verbatim: this is the
/// 1:1 parity proof — the CPU selection must reproduce the GPU contract's
/// oracle lane for lane.
#[test]
fn cross_matches_gpu_contract_fixture() {
    let a = Array::from_shape_vec([6], vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let b = Array::from_shape_vec([6], vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let mut out = Array::from_shape_vec([6], vec![0.0f32; 6]).unwrap();
    cross_into(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn cross_combines_each_triple_in_kernel_order() {
    // (2,3,4)x(5,6,7) = (3*7-4*6, 4*5-2*7, 2*6-3*5) = (-3,6,-3).
    let a = Array::from_shape_vec([3], vec![2.0f64, 3.0, 4.0]).unwrap();
    let b = Array::from_shape_vec([3], vec![5.0f64, 6.0, 7.0]).unwrap();
    let mut out = Array::from_shape_vec([3], vec![0.0f64; 3]).unwrap();
    cross_into(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[-3.0, 6.0, -3.0]);
}

#[test]
fn cross_serves_strided_views() {
    // Step-2 lanes carry the contract fixture; the gaps must stay untouched.
    let a = Array::from_shape_vec(
        [12],
        vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let b = Array::from_shape_vec(
        [12],
        vec![
            0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
    .unwrap();
    let a_strided = a
        .view()
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();
    let b_strided = b
        .view()
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();
    assert_eq!(a_strided.shape(), [6]);
    let mut backing = Array::from_shape_vec([12], vec![0.0f32; 12]).unwrap();
    let mut out = backing
        .slice_with_mut::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();
    cross_into(&a_strided, &b_strided, &mut out).unwrap();
    assert_eq!(
        backing.storage().as_slice(),
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn cross_rejects_invalid_lengths() {
    let a = Array::from_shape_vec([6], vec![0.0f32; 6]).unwrap();
    let short = Array::from_shape_vec([3], vec![0.0f32; 3]).unwrap();
    let mut out = Array::from_shape_vec([6], vec![0.0f32; 6]).unwrap();
    assert!(cross_into(&a.view(), &short.view(), &mut out.view_mut()).is_err());
    assert!(cross_into(&short.view(), &short.view(), &mut out.view_mut()).is_err());
    let ragged = Array::from_shape_vec([4], vec![0.0f32; 4]).unwrap();
    let mut ragged_out = Array::from_shape_vec([4], vec![0.0f32; 4]).unwrap();
    assert!(cross_into(&ragged.view(), &ragged.view(), &mut ragged_out.view_mut()).is_err());
}

#[test]
fn cross_triples_are_independent() {
    // Each output triple reads only its own input triples: no cross-triple
    // contamination, and safe callers cannot alias `out` with an operand
    // anyway (the `&`/`&mut` view signature forbids it at compile time).
    // (1,0,0)x(0,1,0)=(0,0,1); (0,0,2)x(0,3,0)=(-6,0,0);
    // (-1,2,-3)x(4,-5,6)=(-3,-6,-3).
    let a =
        Array::from_shape_vec([9], vec![1.0f64, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 2.0, -3.0]).unwrap();
    let b =
        Array::from_shape_vec([9], vec![0.0f64, 1.0, 0.0, 0.0, 3.0, 0.0, 4.0, -5.0, 6.0]).unwrap();
    let mut out = Array::from_shape_vec([9], vec![0.0f64; 9]).unwrap();
    cross_into(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[0.0, 0.0, 1.0, -6.0, 0.0, 0.0, -3.0, -6.0, -3.0]
    );
}
