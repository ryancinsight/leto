#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

//! Scale-range eigenvalue and Schur checks.

use super::super::a_posteriori::BlockEigenvalue;
use super::super::backward_error::informative;
use super::super::format::{scale, Format};
use super::super::spectral_condition::bauer_fike_factor;
use super::super::{a_posteriori, backward_error};
use super::{exponents, frobenius, sample, sorted_unit, GENERAL, N};
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{eigenvalues, schur, take_francis_steps};

/// `S·diag(1, 2, 4)·S⁻¹` with `S = [[1,1,0],[0,1,1],[1,0,1]]`,
/// `S⁻¹ = ½[[1,−1,1],[1,1,−1],[−1,1,1]]`: eigenvalues `{1, 2, 4}`, entries
/// multiples of `½` below 3 in magnitude.
const SIMILAR: [f64; 9] = [1.5, 0.5, -0.5, -1.0, 3.0, 1.0, -1.5, 1.5, 2.5];

const SIMILAR_EIGENVALUES: [f64; 3] = [1.0, 2.0, 4.0];

/// `‖S‖_F·‖S⁻¹‖_F = √6 · 1.5`.
const SIMILAR_CONDITION: f64 = 3.675;

/// The spectrum a sweep checks: an eigenvalue-bearing base matrix, its
/// eigenvalues ascending, and the Bauer–Fike factor and reference error in
/// units of `s`.
struct Spectrum {
    matrix: [f64; 9],
    eigenvalues: [f64; 3],
    condition: f64,
    reference_error: f64,
}

fn similar_spectrum() -> Spectrum {
    Spectrum {
        matrix: SIMILAR,
        eigenvalues: SIMILAR_EIGENVALUES,
        condition: SIMILAR_CONDITION,
        reference_error: 0.0,
    }
}

/// `GENERAL`'s eigenvalues from the `f64` Schur form at unit scale, with the
/// Bauer–Fike factor from its eigenvectors — computed independently of
/// `schur` (`spectral_condition`).
fn general_spectrum() -> Spectrum {
    let reference = schur(
        &Array2::from_shape_vec([N, N], GENERAL.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap()
    .eigenvalues();
    let mut eigenvalues = [0.0; 3];
    for (slot, z) in eigenvalues.iter_mut().zip(&reference) {
        assert_eq!(z.im, 0.0, "GENERAL has a real spectrum");
        *slot = z.re;
    }
    eigenvalues.sort_by(f64::total_cmp);
    let condition = bauer_fike_factor(&GENERAL, &eigenvalues.map(|re| (re, 0.0)));
    Spectrum {
        matrix: GENERAL,
        eigenvalues,
        condition,
        reference_error: condition * backward_error::francis(N, f64::EPSILON) * frobenius(&GENERAL),
    }
}

fn check_eigenvalues<T: Format>(spectrum: &Spectrum) {
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&spectrum.matrix, exponent);
        let decomposition =
            schur(&sample.matrix.view()).unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let schur_steps = take_francis_steps();
        let spectrum_only = eigenvalues(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let values_steps = take_francis_steps();
        // A posteriori: every eigenvalue of T̂ is one of Â − E = A + (Â − A) − E,
        // within κ·(δ + ‖E‖₂) of the spectrum of A (Bauer–Fike on A).
        let q: Vec<f64> = decomposition
            .q()
            .storage()
            .as_slice()
            .iter()
            .map(|v| v.to_f64())
            .collect();
        let t: Vec<f64> = decomposition
            .t()
            .storage()
            .as_slice()
            .iter()
            .map(|v| scale(v.to_f64(), -exponent))
            .collect();
        let defect = a_posteriori::gram_defect(&q, N, N);
        let residual = a_posteriori::residual(&sample.unit, &q, &t, &q, N, N, N);
        let radius = a_posteriori::schur_certificate(residual, defect, &t);
        for BlockEigenvalue { re, im, error } in a_posteriori::quasi_triangular_eigenvalues(&t, N) {
            let nearest = spectrum
                .eigenvalues
                .iter()
                .map(|e| (re - e).hypot(im))
                .fold(f64::INFINITY, f64::min);
            let certified = spectrum.condition * (sample.input_error + radius)
                + spectrum.reference_error
                + error;
            assert!(
                nearest <= certified,
                "2^{exponent}: λ(T̂) {re}+{im}i is {nearest:e} from the spectrum, certified {certified:e}"
            );
        }
        // A priori, both entry points, where informative: deterministic at
        // caps, else probabilistic at the larger instrumented step count.
        let steps = schur_steps.max(values_steps);
        let Some((eta, tag)) = informative(backward_error::francis(N, sample.epsilon))
            .map(|eta| (eta, "a-priori"))
            .or_else(|| {
                informative(backward_error::francis_tilde(N, sample.epsilon, steps))
                    .map(|eta| (eta, "probabilistic"))
            })
        else {
            continue;
        };
        let bound = spectrum.condition * (sample.input_error + eta * sample.norm)
            + spectrum.reference_error
            + sample.grid;
        for values in [spectrum_only, decomposition.eigenvalues()] {
            for ((re, im), expected) in sorted_unit(&values, exponent)
                .into_iter()
                .zip(spectrum.eigenvalues)
            {
                assert!(
                    (re - expected).abs() <= bound && im.abs() <= bound,
                    "2^{exponent}: [{tag} {steps}] λ {re}+{im}i vs {expected}, bound {bound:e}"
                );
            }
        }
    }
}

#[test]
fn eigenvalues_and_schur_are_correct_across_each_exponent_range() {
    // GENERAL covers the f32 band 2⁻³¹..2⁻²³ where the unscaled Francis step
    // used to stall; SIMILAR the known closed-form spectrum.
    for spectrum in [similar_spectrum(), general_spectrum()] {
        check_eigenvalues::<f64>(&spectrum);
        check_eigenvalues::<f32>(&spectrum);
        check_eigenvalues::<F16>(&spectrum);
        check_eigenvalues::<Bf16>(&spectrum);
    }
}
