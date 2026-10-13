#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

//! Scale-range pivoted-QR checks.

use super::super::backward_error;
use super::super::backward_error::informative;
use super::super::format::{scale, Format};
use super::{exponents, general_singular_values, sample, GENERAL, N};
use eunomia::{Bf16, F16};
use leto::Storage;
use leto_ops::col_piv_qr;

fn check_pivoted_qr<T: Format>() {
    let sigma_min = *general_singular_values().0.last().unwrap();
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&GENERAL, exponent);
        let decomposition = col_piv_qr(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        if sample.input_error < sigma_min / 2.0 {
            assert_eq!(decomposition.rank(), N, "2^{exponent}: rank");
        }
        let q = decomposition.q();
        let r = decomposition.r();
        let (q, r) = (q.storage().as_slice(), r.storage().as_slice());
        let permutation = decomposition.permutation();
        let (mut residual, mut rounding) = (0.0, 0.0);
        for i in 0..N {
            for (k, &column) in permutation.iter().enumerate() {
                // R in units of s before multiplying, so no f64 product is subnormal.
                let terms =
                    (0..N).map(|j| q[i * N + j].to_f64() * scale(r[j * N + k].to_f64(), -exponent));
                let qr: f64 = terms.clone().sum();
                let magnitude: f64 = terms.map(f64::abs).sum();
                residual += (sample.unit[i * N + column] - qr).powi(2);
                rounding +=
                    (backward_error::gamma(N as f64 + 1.0, f64::EPSILON) * magnitude).powi(2);
            }
        }
        // A priori only (pivoted QR returns no certificate of its own), where
        // informative: deterministic at structural counts, else probabilistic
        // (which also covers Bf16 here).
        let Some((eta, tag)) = informative(backward_error::col_piv_qr(N, N, sample.epsilon))
            .map(|eta| (eta, "a-priori"))
            .or_else(|| {
                informative(backward_error::col_piv_qr_tilde(N, N, sample.epsilon))
                    .map(|eta| (eta, "probabilistic"))
            })
        else {
            continue;
        };
        let bound = eta * sample.norm + N as f64 * sample.grid + rounding.sqrt();
        assert!(
            residual.sqrt() <= bound,
            "2^{exponent}: [{tag}] ‖ÂP − QR‖ {} > {bound:e}",
            residual.sqrt()
        );
    }
}

#[test]
fn pivoted_qr_is_correct_across_each_exponent_range() {
    check_pivoted_qr::<f64>();
    check_pivoted_qr::<f32>();
    check_pivoted_qr::<F16>();
    check_pivoted_qr::<Bf16>();
}
