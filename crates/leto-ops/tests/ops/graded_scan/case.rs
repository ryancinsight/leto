//! One scan case and its a-posteriori and a-priori checks on the image `Â`.

use super::super::a_posteriori::{self, BlockEigenvalue};
use super::super::backward_error::{self, informative};
use super::super::format::{epsilon, scale, Format};
use leto::{Array2, Storage};
use leto_ops::{schur, svd_decompose};

pub(super) fn frobenius(values: &[f64]) -> f64 {
    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if largest == 0.0 {
        return 0.0;
    }
    largest
        * values
            .iter()
            .map(|v| (v / largest).powi(2))
            .sum::<f64>()
            .sqrt()
}

/// The image of a `T` array, in units of `2^e`.
pub(super) fn image_of<T: Format>(values: &Array2<T>, exponent: i32) -> Vec<f64> {
    values
        .storage()
        .as_slice()
        .iter()
        .map(|v| scale(v.to_f64(), -exponent))
        .collect()
}

/// Distance from `(re, im)` to the nearest point of `spectrum`.
pub(super) fn nearest(spectrum: &[(f64, f64)], re: f64, im: f64) -> f64 {
    spectrum
        .iter()
        .map(|(r, i)| (re - r).hypot(im - i))
        .fold(f64::INFINITY, f64::min)
}

/// One scan case: the `T` matrix, the image `Â` of what it holds in units of
/// `2^e`, and half the subnormal spacing in the same units.
pub(super) struct Case<'a, T> {
    pub(super) label: &'a str,
    pub(super) matrix: &'a Array2<T>,
    pub(super) image: &'a [f64],
    pub(super) n: usize,
    pub(super) exponent: i32,
    pub(super) grid: f64,
}

impl<T: Format> Case<'_, T> {
    pub(super) fn norm(&self) -> f64 {
        frobenius(self.image)
    }

    /// `svd_decompose`: the a-posteriori Weyl certificate against the `f64`
    /// reference `σ(Â)` (within `reference_error`), and the a-priori residual
    /// and orthogonality where informative. Returns `σ̂` in units of `2^e`.
    pub(super) fn svd_factors(&self, reference: &[f64], reference_error: f64) -> Vec<f64> {
        let (label, n, eps) = (self.label, self.n, epsilon::<T>());
        let full = svd_decompose(&self.matrix.view())
            .unwrap_or_else(|error| panic!("{label}: svd_decompose: {error}"));
        let sigmas: Vec<f64> = full
            .singular_values
            .iter()
            .map(|s| scale(s.to_f64(), -self.exponent))
            .collect();
        // Û and V̂ are scale-free; Σ̂ is in units of 2^e.
        let u = image_of(&full.left_singular_vectors, 0);
        let v = image_of(&full.right_singular_vectors, 0);
        let mut diagonal = vec![0.0; n * n];
        for (i, s) in sigmas.iter().enumerate() {
            diagonal[i * n + i] = *s;
        }
        // Restoring Σ̂ rounds each value once onto T's grid.
        let residual = a_posteriori::residual(self.image, &u, &diagonal, &v, n, n, n)
            + (n as f64).sqrt() * self.grid;
        let (left, right) = (
            a_posteriori::gram_defect(&u, n, n),
            a_posteriori::gram_defect(&v, n, n),
        );
        let (eta_r, eta_o) = backward_error::svd_factors(n, n, eps);
        if let Some(eta) = informative(eta_r) {
            let bound = eta * self.norm();
            assert!(
                residual <= bound,
                "{label}: ‖Â − ÛΣ̂V̂ᵀ‖_F {residual:e} > {bound:e}"
            );
        }
        if let Some(eta) = informative(eta_o) {
            assert!(
                left <= eta && right <= eta,
                "{label}: ‖ÛᵀÛ − I‖ {left:e}, ‖V̂ᵀV̂ − I‖ {right:e} > {eta:e}"
            );
        }
        let bound = a_posteriori::svd_certificate(residual, left, right, &sigmas) + reference_error;
        for (s, r) in sigmas.iter().zip(reference) {
            assert!(
                (s - r).abs() <= bound,
                "{label}: σ {s:e} vs {r:e}, certified {bound:e} (residual {residual:e}, δ_U {left:e}, δ_V {right:e})"
            );
        }
        sigmas
    }

    /// `schur`: the read-off of each returned eigenvalue from its own `T̂`
    /// block, the a-posteriori Bauer–Fike certificate against the `f64`
    /// reference spectrum (`condition` is `None` when `Â` has no eigenvector
    /// basis to bound), and the a-priori residual and orthogonality where
    /// informative.
    pub(super) fn schur_factors(
        &self,
        reference: &[(f64, f64)],
        condition: Option<f64>,
        reference_error: f64,
    ) {
        let (label, n, eps) = (self.label, self.n, epsilon::<T>());
        let decomposition =
            schur(&self.matrix.view()).unwrap_or_else(|error| panic!("{label}: schur: {error}"));
        let q = image_of(&decomposition.q(), 0);
        let t = image_of(&decomposition.t(), self.exponent);
        let defect = a_posteriori::gram_defect(&q, n, n);
        // Restoring T̂ rounds each entry once; ‖Q̂‖₂² ≤ 1 + δ.
        let residual = a_posteriori::residual(self.image, &q, &t, &q, n, n, n)
            + (1.0 + defect) * n as f64 * self.grid;
        let (eta_r, eta_o) = backward_error::schur_factors(n, eps);
        if let Some(eta) = informative(eta_r) {
            let bound = eta * self.norm();
            assert!(
                residual <= bound,
                "{label}: ‖Â − Q̂T̂Q̂ᵀ‖_F {residual:e} > {bound:e}"
            );
        }
        if let Some(eta) = informative(eta_o) {
            assert!(defect <= eta, "{label}: ‖Q̂ᵀQ̂ − I‖ {defect:e} > {eta:e}");
        }
        let blocks = a_posteriori::quasi_triangular_eigenvalues(&t, n);
        // The returned eigenvalues read off the same blocks in T: real parts
        // exact up to restoring, imaginary parts `√|b|·√|c|` within γ₃(ε(T)).
        for (z, &BlockEigenvalue { re, im, error }) in
            decomposition.eigenvalues().iter().zip(&blocks)
        {
            let (got_re, got_im) = (
                scale(z.re.to_f64(), -self.exponent),
                scale(z.im.to_f64(), -self.exponent),
            );
            let read_off = backward_error::gamma(3.0, eps) * im.abs() + error + self.grid;
            assert!(
                (got_re - re).abs() <= self.grid && (got_im - im).abs() <= read_off,
                "{label}: eigenvalue {got_re:e}+{got_im:e}i read off as {re:e}+{im:e}i"
            );
        }
        let Some(condition) = condition else {
            return;
        };
        let radius = a_posteriori::schur_certificate(residual, defect, &t);
        for BlockEigenvalue { re, im, error } in blocks {
            let distance = nearest(reference, re, im);
            let bound = condition * radius + reference_error + error;
            assert!(
                distance <= bound,
                "{label}: λ(T̂) {re:e}+{im:e}i is {distance:e} from the spectrum, certified {bound:e} (κ {condition:e}, residual {residual:e}, δ {defect:e})"
            );
        }
    }
}
