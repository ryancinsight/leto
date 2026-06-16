//! Householder QR factorization kernel (`A →` packed reflectors + `R`).

use super::QrDecomposition;
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result, Storage};

/// Compute the Householder QR factorization of an `m × n` matrix, `m ≥ n`.
///
/// The input may be strided/transposed; it is copied once into row-major
/// working storage. Underdetermined shapes (`m < n`), non-finite values, and
/// exactly-zero pivot-column norms are rejected with distinct error reasons.
/// The zero-norm rejection is an exact contract: near rank-deficiency leaves
/// a tiny floating-point residue rather than an exact zero and manifests as
/// ill-conditioning of the solve — detecting it requires column pivoting or
/// an SVD, which this unpivoted factorization deliberately does not do.
pub fn qr_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<QrDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows < cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![cols, cols],
        });
    }

    // One bulk row-major copy into working storage, then a single tight
    // finiteness scan — replacing `rows*cols` bounds-checked element gets. The
    // factorization mutates `a` in place.
    let contiguous = matrix.to_contiguous();
    let a_slice = contiguous.storage().as_slice();
    if !a_slice.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "QR input contains a non-finite value".to_string(),
        });
    }
    let mut a = a_slice.to_vec();

    let mut heads = vec![T::ZERO; cols];
    let mut betas = vec![T::ZERO; cols];
    // Reused row-vector scratch `w = vᵀ·A_trailing` for the rank-1 reflector
    // update (see the trailing-column apply); one allocation, not one per column.
    let mut w = vec![T::ZERO; cols];

    for k in 0..cols {
        // ‖x‖ for the pivot column below (and including) the diagonal.
        let mut norm_sq = T::ZERO;
        for r in k..rows {
            let x = a[r * cols + k];
            norm_sq = norm_sq.add(x.mul(x));
        }
        let norm = norm_sq.sqrt();
        if norm == T::ZERO {
            return Err(LetoError::StorageError {
                reason: format!("QR pivot column {k} has zero norm: matrix is rank-deficient"),
            });
        }

        // alpha = -sign(x₀)·‖x‖ for cancellation-free head computation.
        let pivot = a[k * cols + k];
        let alpha = if pivot > T::ZERO {
            T::ZERO.sub(norm)
        } else {
            norm
        };
        let head = pivot.sub(alpha);

        // vᵀv = head² + Σ tail²  (tail entries stay in place below the diagonal).
        let mut v_norm_sq = head.mul(head);
        for r in (k + 1)..rows {
            let x = a[r * cols + k];
            v_norm_sq = v_norm_sq.add(x.mul(x));
        }
        let beta = T::ONE.add(T::ONE).div(v_norm_sq);

        // Apply H = I − β·v·vᵀ to the trailing columns as a row-oriented rank-1
        // update: w = vᵀ·A[:, k+1..], then A[:, k+1..] −= β·v·w. Accumulating and
        // applying along contiguous matrix rows (row-major) gives cache-friendly,
        // auto-vectorizable inner loops, versus the strided column traversal. The
        // per-`w[c]` summation order (row k, then rows k+1..m ascending) is
        // identical to the column-oriented form, so results are bitwise-identical.
        let trail = cols - (k + 1);
        let w = &mut w[..trail];
        // w ← head · (row k of the trailing block).
        let row_k = &a[k * cols + (k + 1)..k * cols + cols];
        for (wc, &akc) in w.iter_mut().zip(row_k) {
            *wc = head.mul(akc);
        }
        // w += v[r] · (row r) for r = k+1..m  (v[r] is the in-place reflector tail).
        for r in (k + 1)..rows {
            let vr = a[r * cols + k];
            let row_r = &a[r * cols + (k + 1)..r * cols + cols];
            for (wc, &arc) in w.iter_mut().zip(row_r) {
                *wc = wc.add(vr.mul(arc));
            }
        }
        // Scale in place: w[c] ← β·w[c] = bs[c]. Scaling here (then multiplying by
        // head / v[r]) reproduces the column-oriented grouping (β·s)·head exactly;
        // FP multiplication is commutative but not associative, so the order matters.
        for wc in w.iter_mut() {
            *wc = beta.mul(*wc);
        }
        // Row k: A[k, c] −= bs[c]·head.
        let row_k = &mut a[k * cols + (k + 1)..k * cols + cols];
        for (akc, &wc) in row_k.iter_mut().zip(w.iter()) {
            *akc = akc.sub(wc.mul(head));
        }
        // Rows r>k: A[r, c] −= bs[c]·v[r].
        for r in (k + 1)..rows {
            let vr = a[r * cols + k];
            let row_r = &mut a[r * cols + (k + 1)..r * cols + cols];
            for (arc, &wc) in row_r.iter_mut().zip(w.iter()) {
                *arc = arc.sub(wc.mul(vr));
            }
        }

        a[k * cols + k] = alpha; // R diagonal; v's tail remains below.
        heads[k] = head;
        betas[k] = beta;
    }

    Ok(QrDecomposition {
        packed: a,
        heads,
        betas,
        rows,
        cols,
    })
}
