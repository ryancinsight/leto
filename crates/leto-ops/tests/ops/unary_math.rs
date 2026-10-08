#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    bessel_k0, dot, hamming_distance, j0, j1, jaccard_distance, l2_normalize_into, map_inplace,
    scalar_map, scalar_map_into, sinc, unary_map, unary_map_into, AbsOp, AcosOp, AcoshOp, AddOp,
    AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CoshOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp,
    J0Op, J1Op, K0Op, Log10Op, Log1pOp, Log2Op, MulOp, NegOp, PowfOp, RoundOp, SignOp, SincOp,
    SinhOp, SqrtOp, TanOp, TanhOp, TruncOp,
};

const EPS: f64 = 1e-12;

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() <= EPS, "actual {a} expected {e}");
    }
}

#[test]
fn test_unary_map_allocating_sqrt() {
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![1.0f64, 4.0, 9.0, 16.0])).unwrap();

    let out = unary_map(SqrtOp, &array.view()).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_unary_map_into_neg_and_abs() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![-1.0f64, 2.0, -3.0, 4.0])).unwrap();

    let mut neg_out = Array::new(layout, VecStorage::fill(4, 0.0f64)).unwrap();
    unary_map_into(NegOp, &array.view(), &mut neg_out.view_mut()).unwrap();
    assert_eq!(neg_out.storage().as_slice(), &[1.0, -2.0, 3.0, -4.0]);

    let abs_out = unary_map(AbsOp, &array.view()).unwrap();
    assert_eq!(abs_out.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_unary_map_exp_matches_reference() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let values = vec![0.0f64, 1.0, -1.0];
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(ExpOp, &array.view()).unwrap();
    let expected: Vec<f64> = values.iter().map(|x| x.exp()).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_powf_op_carries_exponent() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0])).unwrap();

    let out = unary_map(PowfOp { exponent: 2.0 }, &array.view()).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 4.0, 9.0]);
}

#[test]
fn test_map_inplace_mutates_in_place() {
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let mut array = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0])).unwrap();

    map_inplace(&mut array.view_mut(), |x| x * 10.0).unwrap();
    assert_eq!(array.storage().as_slice(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_map_inplace_on_transposed_view() {
    // Transposed (F-order) view: contiguous in memory order, so the fast path
    // applies, and every logical element is touched exactly once.
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let mut array = Array::new(
        layout,
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();

    {
        let mut transposed = array.view_mut().transpose_mut([1, 0]).unwrap();
        map_inplace(&mut transposed, |x| x + 100.0).unwrap();
    }
    assert_eq!(
        array.storage().as_slice(),
        &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    );
}

#[test]
fn test_scalar_map_add_and_mul() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0])).unwrap();

    let added = scalar_map::<AddOp, _, 1>(&array.view(), 10.0).unwrap();
    assert_eq!(added.storage().as_slice(), &[11.0, 12.0, 13.0]);

    let mut out = Array::new(layout, VecStorage::fill(3, 0.0f64)).unwrap();
    scalar_map_into::<MulOp, _, 1>(&array.view(), 2.0, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_dot_contiguous_and_strided() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![5.0f64, 6.0, 7.0, 8.0])).unwrap();

    // 1*5 + 2*6 + 3*7 + 4*8 = 70
    assert_eq!(dot(&a.view(), &b.view()).unwrap(), 70.0);

    // Strided: a row of a transposed 2x2 matrix.
    let m = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    let transposed = m.transpose([1, 0]).unwrap();
    let col0 = transposed
        .slice_with::<1>(&[leto::SliceArg::Index(0), leto::SliceArg::All])
        .unwrap();
    // column 0 of [[1,2],[3,4]] is [1,3]
    let ones = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::new(vec![1.0f64, 1.0]),
    )
    .unwrap();
    assert_eq!(dot(&col0, &ones.view()).unwrap(), 4.0);
}

#[test]
fn test_dot_shape_mismatch_rejected() {
    let a = Array::new(
        Layout::c_contiguous([3]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0, 3.0]),
    )
    .unwrap();
    let b = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0]),
    )
    .unwrap();
    assert!(dot(&a.view(), &b.view()).is_err());
}

#[test]
fn test_l2_normalize() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![3.0f64, 0.0, 4.0])).unwrap();
    let mut out = Array::new(layout, VecStorage::fill(3, 0.0f64)).unwrap();
    l2_normalize_into(&array.view(), &mut out.view_mut(), 0.0).unwrap();
    assert_close_slice(out.storage().as_slice(), &[0.6, 0.0, 0.8]);
}

#[test]
fn test_jaccard_distance() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(
        layout,
        VecStorage::new(vec![0b1100u32, 0b1010, 0b1111, 0b0000]),
    )
    .unwrap();
    let b = Array::new(
        layout,
        VecStorage::new(vec![0b1010u32, 0b1100, 0b1111, 0b0000]),
    )
    .unwrap();
    let dist = jaccard_distance(&a.view(), &b.view()).unwrap();
    assert!((dist - 0.4).abs() <= EPS);
}

#[test]
fn test_hamming_distance() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(
        layout,
        VecStorage::new(vec![0b1100u32, 0b1010, 0b1111, 0b0000]),
    )
    .unwrap();
    let b = Array::new(
        layout,
        VecStorage::new(vec![0b1010u32, 0b1100, 0b1111, 0b0000]),
    )
    .unwrap();
    let dist = hamming_distance(&a.view(), &b.view()).unwrap();
    assert_eq!(dist, 4);
}

#[test]
fn test_sinc_op_zero_is_exactly_one() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![0.0f64, -0.0, 1.0])).unwrap();

    let mut out = Array::new(layout, VecStorage::fill(3, 0.0f64)).unwrap();
    unary_map_into(SincOp, &array.view(), &mut out.view_mut()).unwrap();
    let got = out.storage().as_slice();
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1], 1.0);
    assert!((got[2] - 1.0f64.sin()).abs() <= EPS);
}

#[test]
fn test_sinc_op_matches_scalar_oracle() {
    let layout = Layout::c_contiguous([5]).unwrap();
    let values = vec![0.5f64, 1.0, -1.0, 2.0, core::f64::consts::PI];
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(SincOp, &array.view()).unwrap();
    let expected: Vec<f64> = values.iter().map(|&x| sinc(x)).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_sinc_op_f32_and_nan() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![0.0f32, 1.0, -2.0, f32::NAN])).unwrap();

    let out = unary_map(SincOp, &array.view()).unwrap();
    let got = out.storage().as_slice();
    assert_eq!(got[0], 1.0);
    assert!((got[1] - 1.0f32.sin()).abs() <= 1e-6);
    assert!((got[2] - (-2.0f32).sin() / -2.0).abs() <= 1e-6);
    assert!(got[3].is_nan());
}

#[test]
fn test_bessel_ops_match_scalar_oracle_both_branches() {
    // Values span the rational branch, the |x| = 8 crossover, the Hankel
    // branch, and negative arguments (J0 even, J1 odd).
    let values = vec![0.0f64, 0.5, 1.0, 2.0, 5.0, 7.999, 8.0, 9.0, 12.0, -3.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let j0_out = unary_map(J0Op, &array.view()).unwrap();
    let expected_j0: Vec<f64> = values.iter().map(|&x| j0(x)).collect();
    assert_close_slice(j0_out.storage().as_slice(), &expected_j0);

    let mut j1_out = Array::new(layout, VecStorage::fill(values.len(), 0.0f64)).unwrap();
    unary_map_into(J1Op, &array.view(), &mut j1_out.view_mut()).unwrap();
    let expected_j1: Vec<f64> = values.iter().map(|&x| j1(x)).collect();
    assert_close_slice(j1_out.storage().as_slice(), &expected_j1);
}

#[test]
fn test_bessel_ops_f32_within_lane_precision() {
    let values = vec![0.0f32, 1.0, 4.0, 9.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let j0_out = unary_map(J0Op, &array.view()).unwrap();
    for (got, &x) in j0_out.storage().as_slice().iter().zip(&values) {
        let expected = j0(x as f64) as f32;
        assert!(
            (got - expected).abs() <= 1e-5,
            "j0({x}): got {got} expected {expected}"
        );
    }
    let j1_out = unary_map(J1Op, &array.view()).unwrap();
    for (got, &x) in j1_out.storage().as_slice().iter().zip(&values) {
        let expected = j1(x as f64) as f32;
        assert!(
            (got - expected).abs() <= 1e-5,
            "j1({x}): got {got} expected {expected}"
        );
    }
}

#[test]
fn test_k0_op_matches_scalar_oracle_both_branches() {
    // Values span the 9.8.5 branch, the x = 2 crossover, and the 9.8.6
    // branch; the reference values are DLMF 10.32 / A&S Table 9.8 via the
    // scalar oracle.
    let values = vec![0.1f64, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(K0Op, &array.view()).unwrap();
    let expected: Vec<f64> = values.iter().map(|&x| bessel_k0(x)).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_k0_op_rejects_nonpositive_and_nonfinite() {
    let values = vec![
        0.0f64,
        -0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values)).unwrap();

    let mut out = Array::new(layout, VecStorage::fill(6, 0.0f64)).unwrap();
    unary_map_into(K0Op, &array.view(), &mut out.view_mut()).unwrap();
    for got in out.storage().as_slice() {
        assert!(got.is_nan(), "K0 outside (0, +inf) must be NaN");
    }
}

#[test]
fn test_k0_op_f32_within_lane_precision() {
    let values = vec![0.1f32, 1.0, 2.0, 5.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(K0Op, &array.view()).unwrap();
    for (got, &x) in out.storage().as_slice().iter().zip(&values) {
        let expected = bessel_k0(x as f64) as f32;
        assert!(
            (got - expected).abs() <= 1e-5,
            "k0({x}): got {got} expected {expected}"
        );
    }
}

/// PARITY-9 sweep: every method-routed math marker evaluates its scalar
/// oracle. One row per marker with domain-safe probes (inverse trig inside
/// (-1, 1), acosh >= 1, logs > 0, atanh inside (-1, 1)); expm1/log1p probe
/// near zero where the fused forms keep precision the naive forms lose.
#[test]
fn test_unary_math_markers_match_scalar_oracles() {
    macro_rules! check {
        ($op:expr, $oracle:expr, [$($x:expr),+]) => {{
            let oracle: fn(f64) -> f64 = $oracle;
            let points = [$($x),+];
            let layout = Layout::c_contiguous([points.len()]).unwrap();
            let array = Array::new(layout, VecStorage::new(points.to_vec())).unwrap();
            let out = unary_map($op, &array.view()).unwrap();
            for (lane, (&got, &x)) in
                out.storage().as_slice().iter().zip(&points).enumerate()
            {
                let expected = oracle(x);
                assert!(
                    (got - expected).abs() <= EPS,
                    "{} lane {lane} (x = {x}): got {got}, oracle {expected}",
                    stringify!($op)
                );
            }
        }};
    }

    check!(TanOp, f64::tan, [0.0f64, 0.5, -0.5, 1.0, 3.0]);
    check!(AsinOp, f64::asin, [-0.9f64, -0.5, 0.0, 0.5, 0.9]);
    check!(AcosOp, f64::acos, [-0.9f64, -0.5, 0.0, 0.5, 0.9]);
    check!(AtanOp, f64::atan, [-3.0f64, -1.0, 0.0, 1.0, 3.0]);
    check!(SinhOp, f64::sinh, [-2.0f64, -1.0, 0.0, 1.0, 2.0]);
    check!(CoshOp, f64::cosh, [-2.0f64, -1.0, 0.0, 1.0, 2.0]);
    check!(TanhOp, f64::tanh, [-2.0f64, -1.0, 0.0, 1.0, 2.0]);
    check!(Log2Op, f64::log2, [0.25f64, 0.5, 1.0, 2.0, 8.0]);
    check!(Log10Op, f64::log10, [0.25f64, 0.5, 1.0, 2.0, 8.0]);
    check!(Exp2Op, f64::exp2, [-2.0f64, -1.0, 0.0, 1.0, 3.0]);
    check!(AtanhOp, f64::atanh, [-0.9f64, -0.5, 0.0, 0.5, 0.9]);
    check!(AsinhOp, f64::asinh, [-3.0f64, -1.0, 0.0, 1.0, 3.0]);
    check!(AcoshOp, f64::acosh, [1.0f64, 1.5, 2.0, 3.0, 5.0]);
    check!(Expm1Op, f64::exp_m1, [-1.0f64, -1e-10, 0.0, 1e-10, 1.0]);
    check!(Log1pOp, f64::ln_1p, [-0.5f64, -1e-10, 0.0, 1e-10, 2.0]);
    check!(FloorOp, f64::floor, [-2.7f64, -0.5, 0.5, 2.5, 2.7]);
    check!(CeilOp, f64::ceil, [-2.7f64, -0.5, 0.5, 2.5, 2.7]);
    check!(RoundOp, f64::round, [-2.7f64, -2.5, 0.5, 2.5, 2.7]);
    check!(TruncOp, f64::trunc, [-2.7f64, -0.5, 0.5, 2.5, 2.7]);
    check!(
        ExpNegOp,
        (|x: f64| (-x).exp()) as fn(f64) -> f64,
        [-1.0f64, 0.0, 1.0, 2.0]
    );
}

/// Sign follows ADR 0061 (hephaestus `SignOp`): `0` for `±0` and NaN,
/// `±1` otherwise — not `signum`, which signs zero and propagates NaN.
#[test]
fn test_sign_op_matches_hephaestus_special_cases() {
    let points = [3.5f64, -3.5, 0.0, -0.0, f64::NAN];
    let layout = Layout::c_contiguous([points.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(points.to_vec())).unwrap();

    let out = unary_map(SignOp, &array.view()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[1.0, -1.0, 0.0, 0.0, 0.0],
        "sign special cases"
    );
}

/// Trivial per-element ops stay bandwidth-bound (no parallel slowdown on
/// cache-resident data), matching the abs/neg precedent. Pinned at compile
/// time: a routing change that flips a flag breaks the build, not a test.
#[test]
fn test_trivial_math_markers_are_not_compute_bound() {
    use leto_ops::UnaryOp;
    const _: () = assert!(!<FloorOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<CeilOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<RoundOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<TruncOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<SignOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<TanOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<Expm1Op as UnaryOp<f64>>::COMPUTE_BOUND);
}
