//! QR least-squares solve: apply `Qᵀ` via the stored reflectors, back-substitute `R`.

use super::{qr_decompose, QrDecomposition};
use crate::domain::real::RealScalar;
use leto::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, LetoError, Result};

impl<T: RealScalar> QrDecomposition<T> {
    /// Solve `min ‖A·x − rhs‖₂` (least squares; exact solve when `m = n`).
    ///
    /// Applies the stored reflectors to `rhs` (computing `Qᵀ·rhs` without
    /// materializing `Q`), then back-substitutes against `R`.
    /// Solve `min ‖A·x − rhs‖₂` directly into a caller-owned view `out`.
    #[allow(clippy::needless_range_loop)]
    pub fn solve_least_squares_into(
        &self,
        rhs: &ArrayView1<'_, T>,
        out: &mut ArrayViewMut1<'_, T>,
    ) -> Result<()> {
        let (m, n) = (self.rows, self.cols);
        if rhs.shape() != [m] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![m],
            });
        }
        if out.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: out.shape().to_vec(),
                rhs: vec![n],
            });
        }

        let mut y_stack = [T::ZERO; 128];
        let mut y_vec = Vec::new();
        let y = if m <= 128 {
            if let Some(slice) = rhs.as_slice() {
                y_stack[..m].copy_from_slice(&slice[..m]);
            } else {
                for k in 0..m {
                    y_stack[k] = *rhs.get([k])?;
                }
            }
            &mut y_stack[..m]
        } else {
            if let Some(slice) = rhs.as_slice() {
                y_vec = slice.to_vec();
            } else {
                y_vec.reserve_exact(m);
                for k in 0..m {
                    y_vec.push(*rhs.get([k])?);
                }
            }
            &mut y_vec[..]
        };

        // y ← Qᵀ·y, one reflector at a time.
        for k in 0..n {
            let mut s = self.heads[k].mul(y[k]);
            for (offset, &value) in y[(k + 1)..m].iter().enumerate() {
                s = s.add(self.packed[(k + 1 + offset) * n + k].mul(value));
            }
            let bs = self.betas[k].mul(s);
            y[k] = y[k].sub(bs.mul(self.heads[k]));
            for (offset, slot) in y[(k + 1)..m].iter_mut().enumerate() {
                let update = bs.mul(self.packed[(k + 1 + offset) * n + k]);
                *slot = slot.sub(update);
            }
        }

        // Back-substitute R·x = y[..n].
        for r in (0..n).rev() {
            let mut acc = y[r];
            for (offset, &solved) in y[(r + 1)..n].iter().enumerate() {
                acc = acc.sub(self.packed[r * n + r + 1 + offset].mul(solved));
            }
            y[r] = acc.div(self.packed[r * n + r]);
        }

        // Write x to out.
        if let Some(out_slice) = out.as_mut_slice() {
            out_slice.copy_from_slice(&y[..n]);
        } else {
            for k in 0..n {
                *out.get_mut([k])? = y[k];
            }
        }
        Ok(())
    }

    /// Solve `min ‖A·x − rhs‖₂` (least squares; exact solve when `m = n`).
    ///
    /// Applies the stored reflectors to `rhs` (computing `Qᵀ·rhs` without
    /// materializing `Q`), then back-substitutes against `R`.
    pub fn solve_least_squares(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.cols;
        let mut out = Array1::from_elem([n], T::ZERO);
        self.solve_least_squares_into(rhs, &mut out.view_mut())?;
        Ok(out)
    }
}

/// Convenience: factor and solve `min ‖A·x − rhs‖₂` in one call.
#[inline]
pub fn solve_least_squares<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    qr_decompose(matrix)?.solve_least_squares(rhs)
}
