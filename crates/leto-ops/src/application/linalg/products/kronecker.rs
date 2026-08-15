//! Kronecker (tensor) product of two matrices.

use crate::domain::scalar::Scalar;
use leto::{Array2, ArrayView2, Result};

/// Kronecker product `A ⊗ B`.
///
/// ```
/// use leto::{Array2, Storage};
/// use leto_ops::kron;
///
/// let a = Array2::from_shape_vec([1, 2], vec![2_i32, 3]).unwrap();
/// let b = Array2::from_shape_vec([2, 1], vec![5_i32, 7]).unwrap();
///
/// let product = kron(&a.view(), &b.view()).unwrap();
/// assert_eq!(product.shape(), [2, 2]);
/// assert_eq!(product.storage().as_slice(), &[10, 15, 14, 21]);
/// ```
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
    // Establishes the storage proof the strided branch's `get_unchecked` on
    // `a_data` relies on. `b` is read through the checked `get`/`as_slice`
    // paths, but is validated too so both operands carry the same contract.
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    let rows = a_rows * b_rows;
    let cols = a_cols * b_cols;
    let mut values = vec![T::ZERO; rows * cols];

    // Cache b in a contiguous vector to eliminate 2D bounds checks and layout offset calculations
    // in the hot innermost loops.
    let mut b_cached = Vec::with_capacity(b_rows * b_cols);
    if let Some(slice) = b.as_slice() {
        b_cached.extend_from_slice(&slice[..b_rows * b_cols]);
    } else {
        for k in 0..b_rows {
            for l in 0..b_cols {
                b_cached.push(*b.get([k, l])?);
            }
        }
    }

    if let Some(a_slice) = a.as_slice() {
        for i in 0..a_rows {
            let row_base = i * b_rows;
            for j in 0..a_cols {
                let scale = a_slice[i * a_cols + j];
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
    } else {
        let a_data = a.data();
        let a_strides = a.strides();
        let a_offset = a.offset() as isize;

        for i in 0..a_rows {
            let a_row_off = a_offset + i as isize * a_strides[0];
            let row_base = i * b_rows;
            for j in 0..a_cols {
                let a_off = a_row_off + j as isize * a_strides[1];
                // SAFETY: `validate_storage_len` above proved every physical
                // offset this layout addresses lies inside `a_data`.
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
    }

    Ok(Array2::from_shape_vec([rows, cols], values)
        .expect("Kronecker product shape [a_rows·b_rows, a_cols·b_cols] matches storage"))
}
