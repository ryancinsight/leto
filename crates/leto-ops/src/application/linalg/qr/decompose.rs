//! Householder QR factorization kernel (`A →` packed reflectors + `R`).
//!
//! Panel-blocked (LAPACK `dgeqrf` structure): columns are reduced in panels of
//! [`BLOCK_WIDTH`]; within a panel each reflector is applied unblocked to the
//! remaining panel columns, then the panel's reflectors are applied to the
//! trailing block in one BLAS-3 sweep via the compact-WY
//! [`reflector_block`](crate::application::linalg::reflector_block). For
//! `cols ≤ BLOCK_WIDTH` there is a single panel and no trailing block, so the
//! path is byte-for-byte the original unblocked factorization — small matrices
//! pay nothing.

use super::QrDecomposition;
use crate::application::linalg::reflector_block::apply_block_left;
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result, Storage};

/// Panel width for blocked QR. 32 matches the GEMM tile and the `axpy` crossover.
const BLOCK_WIDTH: usize = 32;

/// Minimum row count for the blocked path to pay. The compact-WY trailing GEMM
/// only amortizes the panel extraction/transpose/allocation overhead at scale
/// (measured A/B crossover ≈ 200 rows: 256² QR 1.51 → 1.29 ms blocked, but 128²
/// 175 → 223 µs — a regression). Below it the factorization runs as a single
/// full-width panel, byte-for-byte the original unblocked sweep, so small and
/// medium matrices pay nothing.
const BLOCK_MIN_ROWS: usize = 256;

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

    // Block only when the matrix is large enough that the trailing GEMM amortizes
    // the panel overhead; otherwise a single full-width panel reproduces the exact
    // unblocked sweep (no block apply, no numeric change).
    let panel_width = if rows >= BLOCK_MIN_ROWS {
        BLOCK_WIDTH
    } else {
        cols.max(1)
    };
    let mut panel_start = 0;
    while panel_start < cols {
        let panel_end = (panel_start + panel_width).min(cols);

        for k in panel_start..panel_end {
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

            // Apply H = I − β·v·vᵀ to the remaining *panel* columns [k+1, panel_end)
            // as a row-oriented rank-1 update: w = vᵀ·A[:, k+1..panel_end], then
            // A[:, k+1..panel_end] −= β·v·w. The trailing columns [panel_end, cols)
            // are deferred to the panel's single compact-WY block update below.
            let trail = panel_end - (k + 1);
            let w = &mut w[..trail];
            let row_k = &a[k * cols + (k + 1)..k * cols + panel_end];
            for (wc, &akc) in w.iter_mut().zip(row_k) {
                *wc = head.mul(akc);
            }
            for r in (k + 1)..rows {
                let vr = a[r * cols + k];
                let row_r = &a[r * cols + (k + 1)..r * cols + panel_end];
                for (wc, &arc) in w.iter_mut().zip(row_r) {
                    *wc = wc.add(vr.mul(arc));
                }
            }
            for wc in w.iter_mut() {
                *wc = beta.mul(*wc);
            }
            let row_k = &mut a[k * cols + (k + 1)..k * cols + panel_end];
            for (akc, &wc) in row_k.iter_mut().zip(w.iter()) {
                *akc = akc.sub(wc.mul(head));
            }
            for r in (k + 1)..rows {
                let vr = a[r * cols + k];
                let row_r = &mut a[r * cols + (k + 1)..r * cols + panel_end];
                for (arc, &wc) in row_r.iter_mut().zip(w.iter()) {
                    *arc = arc.sub(wc.mul(vr));
                }
            }

            a[k * cols + k] = alpha; // R diagonal; v's tail remains below.
            heads[k] = head;
            betas[k] = beta;
        }

        // Compact-WY block update of the trailing columns [panel_end, cols) by the
        // panel's reflectors (rows [panel_start, rows)). The reflectors are zero
        // above their pivot, so the transform is confined to rows ≥ panel_start.
        // `a[k][k]` currently holds the R diagonal (alpha); the reflector head is
        // in `heads[k]`, so the extracted panel `V` uses the head on the diagonal
        // and the in-place tail below — exactly the stored reflector.
        let nb = panel_end - panel_start;
        let ntrail = cols - panel_end;
        if ntrail > 0 {
            let m_sub = rows - panel_start;
            // V (m_sub × nb): column j is reflector (panel_start + j).
            let mut v = vec![T::ZERO; m_sub * nb];
            let mut panel_betas = vec![T::ZERO; nb];
            for j in 0..nb {
                let col = panel_start + j;
                panel_betas[j] = betas[col];
                v[j * nb + j] = heads[col]; // diagonal head (local row j == col)
                for r in (col + 1)..rows {
                    v[(r - panel_start) * nb + j] = a[r * cols + col];
                }
            }
            // C (m_sub × ntrail): trailing columns extracted contiguously.
            let mut c_block = vec![T::ZERO; m_sub * ntrail];
            for r in panel_start..rows {
                for (jc, slot) in c_block
                    [(r - panel_start) * ntrail..(r - panel_start) * ntrail + ntrail]
                    .iter_mut()
                    .enumerate()
                {
                    *slot = a[r * cols + panel_end + jc];
                }
            }
            apply_block_left(&v, &panel_betas, &mut c_block, m_sub, ntrail, nb);
            // Write the updated trailing block back.
            for r in panel_start..rows {
                for jc in 0..ntrail {
                    a[r * cols + panel_end + jc] = c_block[(r - panel_start) * ntrail + jc];
                }
            }
        }

        panel_start = panel_end;
    }

    Ok(QrDecomposition {
        packed: a,
        heads,
        betas,
        rows,
        cols,
    })
}
