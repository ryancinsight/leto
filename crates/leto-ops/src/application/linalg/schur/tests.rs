//! Kernel-tier scale equivariance of the Francis iteration and the 2×2
//! standardization: with the local products formed through a power-of-two
//! window, `f(2ᵏ·H) = 2ᵏ·f(H)` holds bit for bit — including far outside the
//! matrix-tier gate, where an unscaled first column, reflector norm, or
//! discriminant would overflow (`2⁶⁰⁰`, products near `2¹²⁰⁰`) or underflow
//! (`2⁻⁶⁰⁰`).

use super::{francis, standardize};

/// An unreduced 4×4 Hessenberg matrix with every entry in `(−2, 2)`, exact
/// in `f64`.
const HESSENBERG: [f64; 16] = [
    1.5, 0.5, -0.25, 1.0, //
    1.0, 1.25, 0.5, -0.75, //
    0.0, 0.625, -1.0, 0.5, //
    0.0, 0.0, 0.875, 1.75,
];

fn identity(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 })
        .collect()
}

fn scaled(values: &[f64], exponent: i32) -> Vec<f64> {
    values.iter().map(|v| v * 2.0_f64.powi(exponent)).collect()
}

#[test]
fn francis_iteration_is_power_of_two_equivariant_beyond_the_gate() {
    let n = 4;
    let mut unit = HESSENBERG.to_vec();
    let mut unit_q = identity(n);
    francis::run::<f64, true>(&mut unit, &mut unit_q, n).expect("unit scale converges");
    for exponent in [600, -600] {
        let mut h = scaled(&HESSENBERG, exponent);
        let mut q = identity(n);
        francis::run::<f64, true>(&mut h, &mut q, n)
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        assert_eq!(scaled(&h, -exponent), unit, "2^{exponent}: T");
        assert_eq!(q, unit_q, "2^{exponent}: Q");
    }
}

#[test]
fn standardization_is_power_of_two_equivariant_beyond_the_gate() {
    // Real eigenvalues: disc = (a − d)² + 4bc = 0.0625 + 2 > 0.
    let block = [1.5, 1.0, 0.5, 1.25];
    let mut unit = block.to_vec();
    let mut unit_q = identity(2);
    standardize::standardize(&mut unit, &mut unit_q, 2);
    assert_eq!(unit[2], 0.0, "unit block triangularized");
    for exponent in [600, -600] {
        let mut t = scaled(&block, exponent);
        let mut q = identity(2);
        standardize::standardize(&mut t, &mut q, 2);
        assert_eq!(scaled(&t, -exponent), unit, "2^{exponent}: T");
        assert_eq!(q, unit_q, "2^{exponent}: Q");
    }
}

#[test]
fn gate_keeps_the_deflation_floor_below_epsilon_times_the_norm() {
    use crate::application::linalg::scaling;
    use eunomia::{FloatElement, NumericElement, F16};
    // F16, n = 64: at most 64 deflations at safmin, jointly 8·safmin = 2⁻¹¹,
    // must stay within ε·‖A‖_F. Entries 0.3 and 0.225 give
    // ‖A‖_F/‖A‖_max = 1.25 (upper exponent r = 1, lower l = 0): the floor end
    // is 2^(3 − l)·smlnum = 0.5, so 0.3 (ε·‖A‖_F ≈ 2⁻¹¹·⁴) must be moved up.
    // Crediting r instead of l would leave it at the root end 0.25, unmoved.
    let n = 64;
    let mut values = vec![F16::from_f64(0.0); n * n];
    values[0] = F16::from_f64(0.3);
    values[1] = F16::from_f64(0.225);
    let frobenius = |scale: i32| {
        values
            .iter()
            .map(|v| v.scale_binary(scale).to_f64().powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let (safmin, eps) = (2.0_f64.powi(-14), 2.0_f64.powi(-10));
    let joint_floor = (n as f64).sqrt() * safmin;
    assert!(joint_floor > eps * frobenius(0), "the input must violate");
    let exponent = scaling::gate_exponent(&values, 2, super::francis_bound(n))
        .expect("the range is non-empty")
        .expect("0.3 is below the floor end");
    assert!(joint_floor <= eps * frobenius(-exponent), "{exponent}");
    // `0.3·ones(64)`: `l = 5` credits `‖A‖_F = 64·‖A‖_max` and the input is
    // left in place; without the credit the floor end `0.5` would move it.
    let ones = vec![F16::from_f64(0.3); n * n];
    let unmoved = scaling::gate_exponent(&ones, 2, super::francis_bound(n)).expect("non-empty");
    assert_eq!(unmoved, None);
}

/// `dlahqr`'s bulge start (loop 50). With `h₀₀, h₁₁ ≈ 10⁻³` the deflation
/// pre-check for `h₁₀ ≈ 1.1·10⁻¹⁷` is `ulp·(|h₀₀| + |h₁₁|) ≈ 3.8·10⁻¹⁹`, so
/// row 0 is not split off, but the bulge-start test (`ulp·|v₁|·(|h₀₀| +
/// |h₁₁| + |h₂₂|)`, with `|h₂₂| = 0.375`) finds `h₁₀` negligible and starts
/// every step at row 1. Column 0 is then touched only through
/// `h₁₀·(1 − τ)`, so `h₀₀` survives the run exactly; starting at row 0
/// rotates it (found by a seeded search over 4,000 Hessenberg matrices).
#[test]
fn bulge_starts_below_a_negligible_subdiagonal() {
    let n = 4;
    let mut h = vec![
        0.001_110_045_906_518_637_5,
        -0.875,
        -0.125,
        0.875,
        1.086_723_651_168_370_5e-17,
        0.000_575_212_256_743_316_8,
        -0.375,
        -0.625,
        0.0,
        0.375,
        -0.375,
        0.125,
        0.0,
        0.0,
        0.875,
        -0.625,
    ];
    let original = h[0];
    let mut q = identity(n);
    francis::run::<f64, true>(&mut h, &mut q, n).expect("converges");
    assert_eq!(h[0].to_bits(), original.to_bits(), "{h:?}");
}
