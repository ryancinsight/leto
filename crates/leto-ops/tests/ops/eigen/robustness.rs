#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::super::backward_error::symmetric_certificate;
use super::super::format::{epsilon, scale, Format};
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi};

/// Jacobi factors a diagonal input exactly: its gate's upper end is the
/// overflow threshold over `2^(1+r)` with `r = 0` for a diagonal
/// (`‖A‖_F = ‖A‖_max` to rounding), so no power-of-two move can underflow the
/// small entry, and a zero off-diagonal takes no rotation. Checked for both
/// entry points on the reported pairs and on 294 diagonals per format: seven
/// large binades up to one below the largest, six small ones from the
/// subnormal range up, seven significands each.
fn check_diagonal_is_exact<T: Format>(pairs: &[(f64, f64)]) {
    let mut cases: Vec<(f64, f64)> = pairs.to_vec();
    let small_exponents = [
        T::MIN_EXPONENT - T::PRECISION + 3,
        T::MIN_EXPONENT - T::PRECISION / 2,
        T::MIN_EXPONENT - 1,
        T::MIN_EXPONENT,
        T::MIN_EXPONENT + 1,
        T::MIN_EXPONENT + 5,
    ];
    for big_exponent in (T::MAX_EXPONENT - 7)..T::MAX_EXPONENT {
        for &small_exponent in &small_exponents {
            for k in 1..8 {
                let significand = 1.0 + f64::from(k) / 8.0;
                cases.push((
                    scale(significand, big_exponent),
                    scale(significand, small_exponent),
                ));
            }
        }
    }
    for (big, small) in cases {
        let (big, small) = (T::from_f64(big), T::from_f64(small));
        let matrix = Array2::from_shape_vec([2, 2], vec![big, T::ZERO, T::ZERO, small]).unwrap();
        let expected = [small.to_f64(), big.to_f64()];
        let values = symmetric_eigenvalues_jacobi(&matrix.view()).unwrap();
        let full = symmetric_eigen_jacobi(&matrix.view()).unwrap();
        for computed in [values, full.eigenvalues] {
            assert_eq!(
                [computed[0].to_f64(), computed[1].to_f64()],
                expected,
                "diag({}, {})",
                expected[1],
                expected[0]
            );
        }
    }
}

#[test]
fn symmetric_jacobi_factors_a_diagonal_exactly_in_every_format() {
    check_diagonal_is_exact::<f64>(&[(1e300, 1e-300)]);
    check_diagonal_is_exact::<f32>(&[(1e38, 1e-38)]);
    check_diagonal_is_exact::<Bf16>(&[(1e38, 1e-38)]);
    check_diagonal_is_exact::<F16>(&[(32768.0, 1.0009765625)]);
}

/// `M·(J₄ − 2I)` (`J₄` the all-ones 4×4) has eigenvalues `{−2M, −2M, −2M,
/// 2M}` and `‖A‖_F = 4M = 2²·‖A‖_max`, so its Jacobi gate bound is
/// `2^(1+2)` and its upper end `Ω/8`. At `M = 0.3·Ω` every entry is `≤ Ω/2`
/// and the eigenvalues `±0.6·Ω` are representable, but the rotations'
/// `a_qq − a_pp` between the emerging `+2M` and `−2M` reaches `4M = 1.2·Ω`:
/// only the Frobenius-ratio term `r = 2` of the bound moves the input down
/// (without it the solver returned `−M` four times as a wrong `Ok`). The
/// minimal move is exactly two binades (`M/2 = 0.15·Ω > Ω/8 ≥ M/4`), so the
/// result must be, bit for bit, `4×` the solver's result on `M/4·(J₄ − 2I)`,
/// an input it factors unscaled — and that result is certified against the
/// exact spectrum from its own eigenbasis (`symmetric_certificate`).
fn check_near_overflow_uses_the_norm_ratio<T: Format>() {
    let largest = T::ONE.scale_binary(T::MAX_EXPONENT).to_f64() * (2.0 - epsilon::<T>());
    let m = T::from_f64(largest * 0.3);
    let build = |m: T| {
        let mut values = vec![m; 16];
        for i in 0..4 {
            values[i * 5] = m.neg();
        }
        Array2::from_shape_vec([4, 4], values).unwrap()
    };
    let quarter = m.scale_binary(-2);
    let (matrix, reduced) = (build(m), build(quarter));
    let values = symmetric_eigenvalues_jacobi(&matrix.view())
        .unwrap_or_else(|error| panic!("eigenvalues: {error}"));
    let full = symmetric_eigen_jacobi(&matrix.view())
        .unwrap_or_else(|error| panic!("decomposition: {error}"));
    let reduced_full = symmetric_eigen_jacobi(&reduced.view()).unwrap();
    let expected: Vec<T> = reduced_full
        .eigenvalues
        .iter()
        .map(|v| v.scale_binary(2))
        .collect();
    assert_eq!(values, expected);
    assert_eq!(full.eigenvalues, expected);

    let image: Vec<f64> = reduced
        .storage()
        .as_slice()
        .iter()
        .map(|v| v.to_f64())
        .collect();
    let lambdas: Vec<f64> = reduced_full
        .eigenvalues
        .iter()
        .map(|v| v.to_f64())
        .collect();
    let basis: Vec<f64> = reduced_full
        .eigenvectors
        .storage()
        .as_slice()
        .iter()
        .map(|v| v.to_f64())
        .collect();
    let bound = symmetric_certificate(&image, &lambdas, &basis, 4);
    let two_q = 2.0 * quarter.to_f64();
    for (value, exact) in lambdas.iter().zip([-two_q, -two_q, -two_q, two_q]) {
        assert!(
            (value - exact).abs() <= bound,
            "{value:e} vs {exact:e}, bound {bound:e}"
        );
    }
}

#[test]
fn symmetric_jacobi_near_overflow_scales_by_the_norm_ratio() {
    check_near_overflow_uses_the_norm_ratio::<f64>();
    check_near_overflow_uses_the_norm_ratio::<f32>();
    check_near_overflow_uses_the_norm_ratio::<F16>();
    check_near_overflow_uses_the_norm_ratio::<Bf16>();
}
