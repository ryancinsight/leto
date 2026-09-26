//! A-posteriori oracles: the residuals of the factors a routine returned,
//! measured in `f64` with their own evaluation rounding bounded, and the
//! eigenvalue and singular-value bounds those measured residuals certify.
//!
//! Unlike the a-priori bounds of `backward_error.rs`, nothing here counts the
//! routine's operations: the certificates hold for whatever factors come
//! back, so they are never vacuous. What they certify is the returned values
//! *given* the returned factors; that the factors themselves are accurate is
//! the separate claim `residual ≤ a-priori bound`, assertable only where the
//! a-priori bound is below `1` (see `graded_scan.rs`).
//!
//! # Derivations
//!
//! Evaluation in `f64` (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed., 2002, §3.5): a product of two matrices with inner
//! dimension `m` has `|fl(XY) − XY| ≤ γ_m|X||Y|`; nesting a second product of
//! inner dimension `m′` and a final subtraction gives
//! `γ_{m+m′+1}·(|A| + |X||Y||Z|)` entrywise, bounded in the Frobenius norm.
//!
//! Orthogonal factors: for `Q̂` (`r × c`, `r ≥ c`) with
//! `δ = ‖Q̂ᵀQ̂ − I‖₂ < 1`, the polar decomposition `Q̂ = U·P` (`U` with
//! orthonormal columns, `P = (Q̂ᵀQ̂)^½`) has `‖P − I‖₂ ≤ ‖P² − I‖₂ = δ` (every
//! eigenvalue `p ≥ 0` of `P` has `|p − 1| ≤ |p − 1|·(p + 1)`) and
//! `‖P‖₂ ≤ 1 + δ`.
//!
//! - Schur: with `R = Â − Q̂T̂Q̂ᵀ`, `Â − U·T̂·Uᵀ = R + U(PT̂P − T̂)Uᵀ` and
//!   `PT̂P − T̂ = (P − I)T̂P + T̂(P − I)`, so `E = Â − UT̂Uᵀ` has
//!   `‖E‖₂ ≤ ‖R‖_F + δ(2 + δ)‖T̂‖_F`. `UT̂Uᵀ` is exactly similar to `T̂`, so
//!   every eigenvalue of `T̂` is an eigenvalue of `Â − E`, within
//!   `κ(V)·‖E‖₂` of an eigenvalue of `Â = VΛV⁻¹` (Bauer–Fike).
//! - SVD: with `R = Â − ÛΣ̂V̂ᵀ`, `Û = U_U·P_U`, `V̂ = U_V·P_V`,
//!   `Â − U_UΣ̂U_Vᵀ = R + U_U(P_UΣ̂P_V − Σ̂)U_Vᵀ` and
//!   `P_UΣ̂P_V − Σ̂ = (P_U − I)Σ̂P_V + Σ̂(P_V − I)`, so the sorted `σ̂ᵢ` (the
//!   exact singular values of `U_UΣ̂U_Vᵀ`) are within
//!   `‖R‖_F + (δ_U + δ_V + δ_U·δ_V)‖Σ̂‖₂` of the sorted `σᵢ(Â)` (Weyl).

use super::backward_error::gamma;

const EPS: f64 = f64::EPSILON;

fn frobenius(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}

/// The nonzero positions of each row of a row-major `rows × cols` matrix,
/// so the products below skip exact zeros (exact, and it keeps the
/// structured large-order checks within the test budget).
fn row_support(values: &[f64], rows: usize, cols: usize) -> Vec<Vec<usize>> {
    (0..rows)
        .map(|r| (0..cols).filter(|&c| values[r * cols + c] != 0.0).collect())
        .collect()
}

/// An upper bound on `‖Q̂ᵀQ̂ − I‖₂` for the row-major `rows × cols` `q`: the
/// computed Frobenius defect plus its evaluation rounding
/// `γ_{rows+1}·‖|Q̂|ᵀ|Q̂| + I‖_F`.
pub fn gram_defect(q: &[f64], rows: usize, cols: usize) -> f64 {
    let mut dot = vec![0.0; cols * cols];
    let mut magnitude = vec![0.0; cols * cols];
    for (r, support) in row_support(q, rows, cols).iter().enumerate() {
        for &i in support {
            for &j in support {
                let term = q[r * cols + i] * q[r * cols + j];
                dot[i * cols + j] += term;
                magnitude[i * cols + j] += term.abs();
            }
        }
    }
    let (mut defect, mut rounding) = (0.0_f64, 0.0_f64);
    for i in 0..cols {
        for j in 0..cols {
            let identity = if i == j { 1.0 } else { 0.0 };
            defect += (dot[i * cols + j] - identity).powi(2);
            rounding += (magnitude[i * cols + j] + identity).powi(2);
        }
    }
    defect.sqrt() + gamma(rows as f64 + 1.0, EPS) * rounding.sqrt()
}

/// An upper bound on `‖Â − X·M·Yᵀ‖_F` for row-major `a` (`rows × cols`), `x`
/// (`rows × inner`), `m` (`inner × inner`) and `y` (`cols × inner`),
/// evaluated as `X·(M·Yᵀ)` with its rounding bounded (the order of the
/// summation does not enter `γ`).
pub fn residual(
    a: &[f64],
    x: &[f64],
    m: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    inner: usize,
) -> f64 {
    // `M·Yᵀ` and `|M||Y|ᵀ`, `inner × cols`.
    let mut my = vec![0.0; inner * cols];
    let mut my_abs = vec![0.0; inner * cols];
    let y_support = row_support(y, cols, inner);
    for i in 0..inner {
        for (j, support) in y_support.iter().enumerate() {
            for &k in support {
                let term = m[i * inner + k] * y[j * inner + k];
                my[i * cols + j] += term;
                my_abs[i * cols + j] += term.abs();
            }
        }
    }
    let mut product = vec![0.0; rows * cols];
    let mut product_abs = vec![0.0; rows * cols];
    for (i, support) in row_support(x, rows, inner).iter().enumerate() {
        for &k in support {
            let xik = x[i * inner + k];
            for j in 0..cols {
                product[i * cols + j] += xik * my[k * cols + j];
                product_abs[i * cols + j] += (xik * my_abs[k * cols + j]).abs();
            }
        }
    }
    let computed = (0..rows * cols).map(|t| a[t] - product[t]);
    let magnitude = (0..rows * cols).map(|t| a[t].abs() + product_abs[t]);
    // The magnitudes are themselves evaluated with the same `γ`; `(1 + γ)`
    // covers their rounding.
    let g = gamma(2.0 * inner as f64 + 1.0, EPS);
    frobenius(computed) + g * (1.0 + g) * frobenius(magnitude)
}

/// `‖E‖₂` of the Schur certificate: `residual + δ(2 + δ)‖T̂‖_F`.
pub fn schur_certificate(residual: f64, defect: f64, t: &[f64]) -> f64 {
    assert!(
        defect < 1.0,
        "Schur vectors are not near-orthonormal: {defect}"
    );
    residual + defect * (2.0 + defect) * frobenius(t.iter().copied())
}

/// The Weyl radius of the SVD certificate:
/// `residual + (δ_U + δ_V + δ_U·δ_V)‖Σ̂‖₂`.
pub fn svd_certificate(residual: f64, left: f64, right: f64, sigmas: &[f64]) -> f64 {
    assert!(
        left < 1.0 && right < 1.0,
        "singular vectors are not near-orthonormal: {left}, {right}"
    );
    let largest = sigmas.iter().fold(0.0_f64, |acc, s| acc.max(s.abs()));
    residual + (left + right + left * right) * largest
}

/// An eigenvalue of a diagonal block of `T̂`, read in `f64`, with a bound on
/// that evaluation's error.
#[derive(Clone, Copy)]
pub struct BlockEigenvalue {
    pub re: f64,
    pub im: f64,
    pub error: f64,
}

/// The eigenvalues of a quasi-triangular `T̂` in `f64`, each with a bound on
/// its `f64` evaluation error: `1 × 1` blocks exactly; a `2 × 2` block must be
/// in `dlanv2` standard form (`a = d`, `b·c < 0`) — the Schur contract — and
/// gives `a ± i·√|b|·√|c|`, within `γ₃·|Im|`. Restoring `T̂` onto the grid can
/// underflow a standard block's `b` to zero; the block is then triangular and
/// gives `d`, `a` exactly, in the order `dlanv2` returns them.
pub fn quasi_triangular_eigenvalues(t: &[f64], n: usize) -> Vec<BlockEigenvalue> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1) * n + i] != 0.0 {
            let (a, b, c, d) = (
                t[i * n + i],
                t[i * n + i + 1],
                t[(i + 1) * n + i],
                t[(i + 1) * n + i + 1],
            );
            if b == 0.0 {
                for re in [d, a] {
                    out.push(BlockEigenvalue {
                        re,
                        im: 0.0,
                        error: 0.0,
                    });
                }
                i += 2;
                continue;
            }
            assert!(
                a == d && b * c < 0.0,
                "2×2 block at {i} is not in standard form: [[{a:e}, {b:e}], [{c:e}, {d:e}]]"
            );
            let im = b.abs().sqrt() * c.abs().sqrt();
            let error = gamma(3.0, EPS) * im;
            out.push(BlockEigenvalue { re: a, im, error });
            out.push(BlockEigenvalue {
                re: a,
                im: -im,
                error,
            });
            i += 2;
        } else {
            out.push(BlockEigenvalue {
                re: t[i * n + i],
                im: 0.0,
                error: 0.0,
            });
            i += 1;
        }
    }
    out
}
