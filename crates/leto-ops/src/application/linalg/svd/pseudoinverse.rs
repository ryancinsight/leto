//! Moore-Penrose pseudoinverse via the rank-revealing SVD.

use super::{default_tolerance, svd_rank_revealing_with_tolerance};
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, Result};

/// Moore-Penrose pseudoinverse `A⁺` (shape `n × m`), rank-revealing.
///
/// From the SVD `A = U Σ Vᵀ`, the unique pseudoinverse is `A⁺ = V Σ⁺ Uᵀ` where
/// `Σ⁺` reciprocates the nonzero singular values and transposes:
/// `A⁺ = Σ_{σᵢ > τ·σ_max} σᵢ⁻¹ vᵢ uᵢᵀ`. The relative threshold `τ·σ_max` drops
/// singular values below the noise floor, so **rank-deficient inputs are
/// handled** (unlike a normal-equations or full-rank-only inverse).
///
/// # Theorem (Moore-Penrose conditions)
/// `A⁺` defined above satisfies the four defining identities; in particular
/// `A A⁺ A = A` and `A⁺ A A⁺ = A⁺`. *Proof:* substitute `A = U Σ Vᵀ`,
/// `A⁺ = V Σ⁺ Uᵀ` and use `UᵀU = VᵀV = I` on the retained columns:
/// `A A⁺ A = U Σ (Σ⁺ Σ) Vᵀ = U Σ Vᵀ = A`, since `Σ⁺Σ` is the identity on the
/// nonzero singular directions. ∎
///
/// Numerically sound: built on the one-sided Jacobi SVD, it never forms `AᵀA`
/// and so does not square the condition number.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty or non-finite input.
pub fn pinv<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    let tolerance = default_tolerance::<T>();
    let svd = svd_rank_revealing_with_tolerance(matrix, tolerance)?;

    let sigma_max = svd
        .singular_values
        .iter()
        .copied()
        .fold(T::ZERO, |m, s| if s > m { s } else { m });
    let cutoff = sigma_max.mul(tolerance);

    // Reciprocate retained singular values; below-cutoff directions contribute 0.
    let inv_sigma: Vec<T> = svd
        .singular_values
        .iter()
        .map(|&sigma| {
            if sigma > cutoff {
                T::ONE.div(sigma)
            } else {
                T::ZERO
            }
        })
        .collect();

    let u = &svd.left_singular_vectors; // [rows, k]
    let v = &svd.right_singular_vectors; // [cols, k]

    // A⁺[i][j] = Σ_t V[i][t] · (σₜ⁻¹) · U[j][t]
    let mut values = vec![T::ZERO; cols * rows];
    for i in 0..cols {
        for j in 0..rows {
            let mut acc = T::ZERO;
            for (t, &inv_s) in inv_sigma.iter().enumerate() {
                let v_it = *v.get([i, t])?;
                let u_jt = *u.get([j, t])?;
                acc = acc.add(v_it.mul(inv_s).mul(u_jt));
            }
            values[i * rows + j] = acc;
        }
    }
    Ok(Array2::from_shape_vec([cols, rows], values)
        .expect("pseudoinverse shape [cols, rows] matches storage"))
}
