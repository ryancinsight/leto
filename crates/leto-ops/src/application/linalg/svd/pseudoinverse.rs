//! Moore-Penrose pseudoinverse via the SVD.

use super::svd_decompose;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, Result, Storage};

/// Relative rank cutoff: singular values below `1e-12 · σ_max` are treated as
/// noise and their directions dropped from `A⁺`. Relative rather than absolute
/// so the cutoff scales with the matrix; `1e-12` sits well above `f64` rounding
/// of the decomposition (`O(ε·σ_max)`, `ε ≈ 2.2e-16`) and below any singular
/// value a caller would consider structurally present.
fn rank_cutoff_ratio<T: RealScalar>() -> T {
    T::ONE.div(T::from_usize(1_000_000_000_000))
}

/// Moore-Penrose pseudoinverse `A⁺` (shape `n × m`), rank-revealing.
///
/// From the SVD `A = U Σ Vᵀ`, the unique pseudoinverse is `A⁺ = V Σ⁺ Uᵀ` where
/// `Σ⁺` reciprocates the nonzero singular values and transposes:
/// `A⁺ = Σ_{σᵢ > τ·σ_max} σᵢ⁻¹ vᵢ uᵢᵀ`. The relative threshold `τ·σ_max` drops
/// singular values below the noise floor, so **rank-deficient inputs are
/// handled** (unlike a normal-equations or full-rank-only inverse).
///
/// Dropping a direction here is safe precisely because
/// [`svd_decompose`](super::svd_decompose) materializes an orthonormal `U`
/// column for every singular value including the zero ones: the retained
/// directions form an orthonormal basis of the numerical range, so
/// `Σ⁺Σ` is the identity on them and the Moore-Penrose identities close.
///
/// # Theorem (Moore-Penrose conditions)
/// `A⁺` defined above satisfies the four defining identities; in particular
/// `A A⁺ A = A` and `A⁺ A A⁺ = A⁺`. *Proof:* substitute `A = U Σ Vᵀ`,
/// `A⁺ = V Σ⁺ Uᵀ` and use `UᵀU = VᵀV = I` on the retained columns:
/// `A A⁺ A = U Σ (Σ⁺ Σ) Vᵀ = U Σ Vᵀ = A`, since `Σ⁺Σ` is the identity on the
/// nonzero singular directions. ∎
///
/// Numerically sound: built on the bidiagonal-QR SVD, it never forms `AᵀA` and
/// so does not square the condition number.
///
/// # Errors
/// [`leto::LetoError`] on empty or non-finite input.
pub fn pinv<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    let svd = svd_decompose(matrix)?;

    let sigma_max = svd
        .singular_values
        .iter()
        .copied()
        .fold(T::ZERO, |m, s| if s > m { s } else { m });
    let cutoff = sigma_max.mul(rank_cutoff_ratio::<T>());

    // Reciprocate retained singular values; below-cutoff directions contribute 0.
    let k = svd.singular_values.len();
    let mut inv_sigma_stack = [T::ZERO; 128];
    let mut inv_sigma_vec = Vec::new();
    let inv_sigma = if k <= 128 {
        &mut inv_sigma_stack[..k]
    } else {
        inv_sigma_vec.resize(k, T::ZERO);
        &mut inv_sigma_vec[..]
    };

    for (i, inv_sig_i) in inv_sigma.iter_mut().enumerate().take(k) {
        let sigma = svd.singular_values[i];
        *inv_sig_i = if sigma > cutoff {
            T::ONE.div(sigma)
        } else {
            T::ZERO
        };
    }

    let u = &svd.left_singular_vectors; // [rows, k]
    let v = &svd.right_singular_vectors; // [cols, k]

    let u_slice = u.storage().as_slice();
    let v_slice = v.storage().as_slice();
    let u_cols = u.shape()[1];
    let v_cols = v.shape()[1];

    // A⁺[i][j] = Σ_t V[i][t] · (σₜ⁻¹) · U[j][t]
    let mut values = vec![T::ZERO; cols * rows];
    for i in 0..cols {
        for j in 0..rows {
            let mut acc = T::ZERO;
            for (t, &inv_s) in inv_sigma.iter().enumerate() {
                let v_it = v_slice[i * v_cols + t];
                let u_jt = u_slice[j * u_cols + t];
                acc = acc.add(v_it.mul(inv_s).mul(u_jt));
            }
            values[i * rows + j] = acc;
        }
    }
    Ok(Array2::from_shape_vec([cols, rows], values)
        .expect("pseudoinverse shape [cols, rows] matches storage"))
}
