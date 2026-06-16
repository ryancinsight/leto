//! QR least-squares solve: apply `Qᵀ` via the stored reflectors, back-substitute `R`.

use super::{qr_decompose, QrDecomposition};
use crate::domain::real::RealScalar;
use leto::{Array1, ArrayView1, ArrayView2, LetoError, Result};

impl<T: RealScalar> QrDecomposition<T> {
    /// Solve `min ‖A·x − rhs‖₂` (least squares; exact solve when `m = n`).
    ///
    /// Applies the stored reflectors to `rhs` (computing `Qᵀ·rhs` without
    /// materializing `Q`), then back-substitutes against `R`.
    pub fn solve_least_squares(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let (m, n) = (self.rows, self.cols);
        if rhs.shape() != [m] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![m],
            });
        }

        let mut y = Vec::with_capacity(m);
        for k in 0..m {
            y.push(*rhs.get([k])?);
        }

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
        let mut x = y;
        x.truncate(n);
        for r in (0..n).rev() {
            let mut acc = x[r];
            for (offset, &solved) in x[(r + 1)..n].iter().enumerate() {
                acc = acc.sub(self.packed[r * n + r + 1 + offset].mul(solved));
            }
            x[r] = acc.div(self.packed[r * n + r]);
        }
        Array1::from_shape_vec([n], x)
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
