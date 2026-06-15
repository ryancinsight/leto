//! Kronecker (tensor) product of two matrices.

use crate::domain::scalar::Scalar;
use leto::{Array2, ArrayView2, Result};

/// Kronecker product `A ⊗ B`.
///
/// For `A ∈ Tᵐˣⁿ` and `B ∈ Tᵖˣ۹` the result is the `mp × nq` block matrix whose
/// `(i, j)` block is the scalar multiple `aᵢⱼ · B`:
///
/// ```text
/// (A ⊗ B)[i·p + k][j·q + l] = aᵢⱼ · bₖₗ.
/// ```
///
/// # Theorem (mixed-product property)
/// `(A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)` whenever the ordinary products `AC` and `BD`
/// are defined. *Proof:* the `((i,k),(j,l))` entry of the left side is
/// `Σₚ Σᵣ aᵢₚ bₖᵣ cₚⱼ dᵣₗ = (Σₚ aᵢₚ cₚⱼ)(Σᵣ bₖᵣ dᵣₗ) = (AC)ᵢⱼ (BD)ₖₗ`,
/// which is the `((i,k),(j,l))` entry of the right side. ∎
///
/// Corollaries (used as oracle-independent test invariants):
/// - **Transpose:** `(A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ`.
/// - **Trace (square `A, B`):** `tr(A ⊗ B) = tr(A) · tr(B)`.
///
/// Output is C-contiguous, filled in a single pass in the native precision of
/// `T` with no per-block temporary allocation.
#[inline]
pub fn kron<T: Scalar>(a: &ArrayView2<'_, T>, b: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [a_rows, a_cols] = a.shape();
    let [b_rows, b_cols] = b.shape();
    let rows = a_rows * b_rows;
    let cols = a_cols * b_cols;
    let mut values = vec![T::ZERO; rows * cols];

    for i in 0..a_rows {
        for j in 0..a_cols {
            let scale = *a.get([i, j])?;
            let row_base = i * b_rows;
            let col_base = j * b_cols;
            for k in 0..b_rows {
                let out_row = row_base + k;
                for l in 0..b_cols {
                    values[out_row * cols + col_base + l] = scale.mul(*b.get([k, l])?);
                }
            }
        }
    }

    Ok(Array2::from_shape_vec([rows, cols], values)
        .expect("Kronecker product shape [a_rows·b_rows, a_cols·b_cols] matches storage"))
}
