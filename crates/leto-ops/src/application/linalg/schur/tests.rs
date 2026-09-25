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
