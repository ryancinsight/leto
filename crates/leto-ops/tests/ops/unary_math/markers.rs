#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::maps::EPS;
use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    unary_map, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CoshOp, EluGradOp, EluOp,
    Exp2Op, ExpNegOp, Expm1Op, FloorOp, GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp,
    HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, Log10Op, Log1pOp, Log2Op,
    MishGradOp, MishOp, ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp, SiluGradOp,
    SiluOp, SinhOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, TanOp, TanhGradOp,
    TanhOp, TruncOp,
};

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

/// PARITY-10 sweep: every activation marker evaluates its reference value.
/// Constants are textbook values (normal CDF/PDF at 1, σ(1), ln 2, exact
/// rationals) or spec-formula evaluations, never the implementation's own
/// output. `SigmoidGradOp`/`TanhGradOp` rows pass forward outputs, per ADR
/// 0061 Decision 7.
#[test]
fn test_activation_markers_match_reference_values() {
    macro_rules! check_vals {
        ($op:expr, [$(($x:expr, $e:expr)),+]) => {{
            let points = [$($x),+];
            let expected = [$($e),+];
            let layout = Layout::c_contiguous([points.len()]).unwrap();
            let array = Array::new(layout, VecStorage::new(points.to_vec())).unwrap();
            let out = unary_map($op, &array.view()).unwrap();
            for (lane, ((&got, &x), &e)) in out
                .storage()
                .as_slice()
                .iter()
                .zip(&points)
                .zip(&expected)
                .enumerate()
            {
                assert!(
                    (got - e).abs() <= EPS,
                    "{} lane {lane} (x = {x}): got {got}, expected {e}",
                    stringify!($op)
                );
            }
        }};
    }

    check_vals!(ReluOp, [(-2.0f64, 0.0), (0.0, 0.0), (3.0, 3.0)]);
    check_vals!(ReluGradOp, [(-2.0f64, 0.0), (0.0, 0.0), (3.0, 1.0)]);
    check_vals!(
        EluOp,
        [(0.0f64, 0.0), (2.0, 2.0), (-1.0, -0.6321205588285577)]
    );
    check_vals!(
        EluGradOp,
        [(0.0f64, 1.0), (2.0, 1.0), (-1.0, 0.36787944117144233)]
    );
    check_vals!(SigmoidOp, [(0.0f64, 0.5), (1.0, 0.7310585786300049)]);
    check_vals!(SigmoidGradOp, [(0.5f64, 0.25), (0.0, 0.0), (1.0, 0.0)]);
    check_vals!(
        TanhGradOp,
        [(0.0f64, 1.0), (0.5, 0.75), (1.0, 0.0), (-1.0, 0.0)]
    );
    check_vals!(SoftplusOp, [(0.0f64, std::f64::consts::LN_2)]);
    check_vals!(SoftplusGradOp, [(0.0f64, 0.5), (1.0, 0.7310585786300049)]);
    check_vals!(SoftsignOp, [(0.0f64, 0.0), (2.0, 0.6666666666666666)]);
    check_vals!(SoftsignGradOp, [(0.0f64, 1.0), (2.0, 0.1111111111111111)]);
    check_vals!(
        HardsigmoidOp,
        [
            (0.0f64, 0.5),
            (3.0, 1.0),
            (-3.0, 0.0),
            (10.0, 1.0),
            (-10.0, 0.0)
        ]
    );
    check_vals!(
        HardsigmoidGradOp,
        [(0.0f64, 0.16666666666666666), (3.0, 0.0), (-3.0, 0.0)]
    );
    check_vals!(HardswishOp, [(0.0f64, 0.0), (3.0, 3.0), (10.0, 10.0)]);
    check_vals!(HardswishGradOp, [(0.0f64, 0.5), (3.0, 1.0), (-3.0, 0.0)]);
    check_vals!(
        GeluOp,
        [
            (0.0f64, 0.0),
            (1.0, 0.8413447460685428),
            (-1.0, -0.15865525393145707)
        ]
    );
    check_vals!(GeluGradOp, [(0.0f64, 0.5), (1.0, 1.0833154705876862)]);
    check_vals!(GeluTanhOp, [(0.0f64, 0.0), (1.0, 0.8411919906082768)]);
    check_vals!(GeluTanhGradOp, [(0.0f64, 0.5), (1.0, 1.0829640838457826)]);
    check_vals!(SiluOp, [(0.0f64, 0.0), (1.0, 0.7310585786300049)]);
    check_vals!(SiluGradOp, [(0.0f64, 0.5), (1.0, 0.9276705118714869)]);
    check_vals!(MishOp, [(0.0f64, 0.0), (1.0, 0.8650983882673103)]);
    check_vals!(MishGradOp, [(0.0f64, 0.6), (1.0, 1.0490362200997922)]);
}

/// Every activation marker propagates NaN rather than resolving it to a
/// branch value (the explicit `is_nan` guards; unordered comparisons alone
/// would silently pick the else arm).
#[test]
fn test_activation_markers_propagate_nan() {
    macro_rules! check_nan {
        ($op:expr) => {{
            let layout = Layout::c_contiguous([1]).unwrap();
            let array = Array::new(layout, VecStorage::new(vec![f64::NAN])).unwrap();
            let out = unary_map($op, &array.view()).unwrap();
            assert!(
                out.storage().as_slice()[0].is_nan(),
                "{} must propagate NaN",
                stringify!($op)
            );
        }};
    }

    check_nan!(ReluOp);
    check_nan!(ReluGradOp);
    check_nan!(EluOp);
    check_nan!(EluGradOp);
    check_nan!(SigmoidOp);
    check_nan!(SigmoidGradOp);
    check_nan!(TanhOp);
    check_nan!(TanhGradOp);
    check_nan!(SoftplusOp);
    check_nan!(SoftplusGradOp);
    check_nan!(SoftsignOp);
    check_nan!(SoftsignGradOp);
    check_nan!(HardsigmoidOp);
    check_nan!(HardsigmoidGradOp);
    check_nan!(HardswishOp);
    check_nan!(HardswishGradOp);
    check_nan!(GeluOp);
    check_nan!(GeluGradOp);
    check_nan!(GeluTanhOp);
    check_nan!(GeluTanhGradOp);
    check_nan!(SiluOp);
    check_nan!(SiluGradOp);
    check_nan!(MishOp);
    check_nan!(MishGradOp);
}

/// Analytic limits at infinities, bit-exact (all are exactly representable,
/// including the `-0.0` negative-tail limits, which `to_bits` distinguishes
/// from `+0.0`).
#[test]
fn test_activation_markers_match_analytic_limits() {
    macro_rules! check_lim {
        ($op:expr, [$(($x:expr, $e:expr)),+]) => {{
            let points = [$($x),+];
            let expected: Vec<f64> = vec![$($e),+];
            let layout = Layout::c_contiguous([points.len()]).unwrap();
            let array = Array::new(layout, VecStorage::new(points.to_vec())).unwrap();
            let out = unary_map($op, &array.view()).unwrap();
            for (lane, ((&got, &x), &e)) in out
                .storage()
                .as_slice()
                .iter()
                .zip(&points)
                .zip(&expected)
                .enumerate()
            {
                assert_eq!(
                    got.to_bits(),
                    e.to_bits(),
                    "{} lane {lane} (x = {x}): got {got}, expected {e}",
                    stringify!($op)
                );
            }
        }};
    }

    check_lim!(
        ReluOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(ReluGradOp, [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]);
    check_lim!(
        EluOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, -1.0)]
    );
    check_lim!(EluGradOp, [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]);
    check_lim!(SigmoidOp, [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]);
    check_lim!(
        SoftplusOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        SoftplusGradOp,
        [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        SoftsignOp,
        [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, -1.0)]
    );
    check_lim!(
        SoftsignGradOp,
        [(f64::INFINITY, 0.0), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        HardsigmoidOp,
        [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        HardsigmoidGradOp,
        [(f64::INFINITY, 0.0), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        HardswishOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, -0.0)]
    );
    check_lim!(
        HardswishGradOp,
        [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        GeluOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, -0.0)]
    );
    check_lim!(GeluGradOp, [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]);
    check_lim!(
        GeluTanhOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, -0.0)]
    );
    check_lim!(
        GeluTanhGradOp,
        [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]
    );
    check_lim!(
        SiluOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, -0.0)]
    );
    check_lim!(SiluGradOp, [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]);
    check_lim!(
        MishOp,
        [(f64::INFINITY, f64::INFINITY), (f64::NEG_INFINITY, -0.0)]
    );
    check_lim!(MishGradOp, [(f64::INFINITY, 1.0), (f64::NEG_INFINITY, 0.0)]);
}

/// Each gradient marker equals its forward marker's central difference,
/// independent of either formula's transcription. Points straddle every
/// gradient crossover (SiLU 2.63, GELU-tanh w = 2.47 near x = 1.42, Mish
/// sp = 1.55 near x = 1.31) and skip the piecewise kinks (ReLU 0,
/// hard-sigmoid/swish ±3). `TanhGradOp` takes forward outputs, so it is
/// checked against `sech²` via an independent `cosh` route instead.
#[test]
fn test_activation_grads_match_finite_differences() {
    use leto_ops::UnaryOp;
    let h = 1e-6f64;
    macro_rules! check_pair {
        ($fwd:expr, $grad:expr, [$($x:expr),+]) => {{
            let fwd = $fwd;
            let grad = $grad;
            for &x in [$($x),+].iter() {
                let numeric = (fwd.apply(x + h) - fwd.apply(x - h)) / (2.0 * h);
                let analytic = grad.apply(x);
                assert!(
                    (numeric - analytic).abs() <= 1e-6,
                    "{} vs {} at x = {x}: numeric {numeric}, analytic {analytic}",
                    stringify!($grad),
                    stringify!($fwd),
                );
            }
        }};
    }

    check_pair!(EluOp, EluGradOp, [-2.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]);
    check_pair!(SoftplusOp, SoftplusGradOp, [-3.0f64, -1.0, 0.0, 1.0, 3.0]);
    check_pair!(SoftsignOp, SoftsignGradOp, [-3.0f64, -1.0, 0.0, 1.0, 3.0]);
    check_pair!(
        SiluOp,
        SiluGradOp,
        [-3.0f64, -1.0, 0.0, 1.0, 2.5, 2.6, 2.7, 3.0]
    );
    check_pair!(
        MishOp,
        MishGradOp,
        [-3.0f64, -1.0, 0.0, 1.0, 1.2, 1.3, 1.4, 3.0]
    );
    check_pair!(GeluOp, GeluGradOp, [-3.0f64, -1.0, 0.0, 1.0, 3.0]);
    check_pair!(
        GeluTanhOp,
        GeluTanhGradOp,
        [-3.0f64, -1.0, 0.0, 1.0, 1.3, 1.4, 1.5, 3.0]
    );
    check_pair!(ReluOp, ReluGradOp, [-3.0f64, -1.0, 1.0, 3.0]);
    check_pair!(
        HardsigmoidOp,
        HardsigmoidGradOp,
        [-5.0f64, -1.0, 0.0, 1.0, 5.0]
    );
    check_pair!(HardswishOp, HardswishGradOp, [-5.0f64, -1.0, 0.0, 1.0, 5.0]);

    for x in [0.1f64, 0.5, 1.0, 2.0, 4.0] {
        let y = x.tanh();
        let got = TanhGradOp.apply(y);
        let expected = 1.0 / (x.cosh() * x.cosh());
        assert!(
            (got - expected).abs() <= EPS,
            "tanh grad at y = tanh({x}) = {y}: got {got}, sech² {expected}"
        );
    }
}

/// Arithmetic-only activation markers stay bandwidth-bound; transcendental
/// ones stay compute-bound. Pinned at compile time with the math markers.
#[test]
fn test_activation_compute_bound_flags() {
    use leto_ops::UnaryOp;
    const _: () = assert!(!<ReluOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<ReluGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<SigmoidGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<TanhGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<SoftsignOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<SoftsignGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<HardsigmoidOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<HardsigmoidGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<HardswishOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(!<HardswishGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<SigmoidOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<GeluOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<GeluTanhGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<MishOp as UnaryOp<f64>>::COMPUTE_BOUND);
    const _: () = assert!(<SiluGradOp as UnaryOp<f64>>::COMPUTE_BOUND);
}
