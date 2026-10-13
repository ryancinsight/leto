//! Scale-range SVD checks.

use super::super::backward_error::informative;
use super::super::format::{scale, Format};
use super::super::{a_posteriori, backward_error};
use super::{exponents, general_singular_values, sample, GENERAL, N};
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{singular_values, svd_decompose, take_dbdsqr_counts};

fn check_singular_values<T: Format>() {
    let (reference, reference_error) = general_singular_values();
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&GENERAL, exponent);
        let values = singular_values(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let values_counts = take_dbdsqr_counts();
        let full = svd_decompose(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let full_counts = take_dbdsqr_counts();
        // A posteriori, on the returned factors (Σ̂ in units of s).
        let sigmas: Vec<f64> = full
            .singular_values
            .iter()
            .map(|v| scale(v.to_f64(), -exponent))
            .collect();
        let as_f64 = |m: &Array2<T>| -> Vec<f64> {
            m.storage().as_slice().iter().map(|v| v.to_f64()).collect()
        };
        let (u, v) = (
            as_f64(&full.left_singular_vectors),
            as_f64(&full.right_singular_vectors),
        );
        let mut diagonal = vec![0.0; N * N];
        for (i, sigma) in sigmas.iter().enumerate() {
            diagonal[i * N + i] = *sigma;
        }
        let residual = a_posteriori::residual(&sample.unit, &u, &diagonal, &v, N, N, N);
        let radius = a_posteriori::svd_certificate(
            residual,
            a_posteriori::gram_defect(&u, N, N),
            a_posteriori::gram_defect(&v, N, N),
            &sigmas,
        );
        let certified = sample.input_error + radius + reference_error;
        for (sigma, expected) in sigmas.iter().zip(&reference) {
            assert!(
                (sigma - expected).abs() <= certified,
                "2^{exponent}: σ {sigma} vs {expected}, certified {certified:e}"
            );
        }
        // A priori, both entry points, where informative: deterministic at
        // caps, else probabilistic at the larger instrumented counts (the
        // bound increases in both, so it covers both runs).
        let sweeps = values_counts.0.max(full_counts.0);
        let chases = values_counts.1.max(full_counts.1);
        let Some((eta, tag)) = informative(backward_error::svd(N, N, sample.epsilon))
            .map(|eta| (eta, "a-priori"))
            .or_else(|| {
                informative(backward_error::svd_tilde(
                    N,
                    N,
                    sample.epsilon,
                    sweeps,
                    chases,
                ))
                .map(|eta| (eta, "probabilistic"))
            })
        else {
            continue;
        };
        let bound = sample.input_error + eta * sample.norm + reference_error + sample.grid;
        for sigmas in [values, full.singular_values] {
            for (sigma, expected) in sigmas.iter().zip(&reference) {
                let sigma = scale(sigma.to_f64(), -exponent);
                assert!(
                    (sigma - expected).abs() <= bound,
                    "2^{exponent}: [{tag} {sweeps}+{chases}] σ {sigma} vs {expected}, bound {bound:e}"
                );
            }
        }
    }
}

#[test]
fn singular_values_are_correct_across_each_exponent_range() {
    check_singular_values::<f64>();
    check_singular_values::<f32>();
    check_singular_values::<F16>();
    check_singular_values::<Bf16>();
}
