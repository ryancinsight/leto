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

    // Cache b in a contiguous vector to eliminate 2D bounds checks and layout offset calculations
    // in the hot innermost loops.
    let mut b_cached = Vec::with_capacity(b_rows * b_cols);
    for k in 0..b_rows {
        for l in 0..b_cols {
            b_cached.push(*b.get([k, l])?);
        }
    }

    let a_data = a.data();
    let a_strides = a.strides();
    let a_offset = a.offset() as isize;

    for i in 0..a_rows {
        let a_row_off = a_offset + i as isize * a_strides[0];
        let row_base = i * b_rows;
        for j in 0..a_cols {
            let a_off = a_row_off + j as isize * a_strides[1];
            // SAFETY: matrix index bounds are validated on construction.
            let scale = unsafe { *a_data.get_unchecked(a_off as usize) };
            if scale == T::ZERO {
                continue;
            }
            let col_base = j * b_cols;
            for k in 0..b_rows {
                let out_row = row_base + k;
                let out_row_off = out_row * cols + col_base;
                let b_row_off = k * b_cols;
                for l in 0..b_cols {
                    // SAFETY: both b_cached and values are pre-allocated and indexes are guaranteed in-bounds.
                    unsafe {
                        let b_val = *b_cached.get_unchecked(b_row_off + l);
                        *values.get_unchecked_mut(out_row_off + l) = scale.mul(b_val);
                    }
                }
            }
        }
    }

    Ok(Array2::from_shape_vec([rows, cols], values)
        .expect("Kronecker product shape [a_rows·b_rows, a_cols·b_cols] matches storage"))
}
