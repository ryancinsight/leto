//! 2-D and 3-D spatial interpolation — SSOT for the Atlas simulation stack.
//!
//! These functions operate in **fractional index space**: the caller is
//! responsible for converting physical coordinates to fractional array indices
//! (`ix = x / dx`, etc.) before calling.  Keeping the math pure (no grid
//! metadata) makes the implementations generic, testable in isolation, and
//! reusable across kwavers, CFDrs, helios, and ritk.
//!
//! ## Degenerate axes
//!
//! Both `bilinear_index_space` and `trilinear_index_space` handle **degenerate
//! axes** (extent 1) gracefully: when a dimension has only one sample the
//! corresponding fractional weight is forced to zero so the interpolant
//! collapses to the single available value without accessing an out-of-bounds
//! index.  This is important for quasi-1D / quasi-2D grids.
//!
//! ## Reference
//!
//! Numerical Recipes in C, §3.6 (bilinear and trilinear interpolation).

use eunomia::{FloatElement, NumericElement};
use leto::{Array2, Array3, ArrayView2, ArrayView3};

// ── Bilinear (2-D) ────────────────────────────────────────────────────────────

/// Bilinear interpolation at fractional array indices `(ix, iy)`.
///
/// `ix` and `iy` are fractional zero-based indices into the first and second
/// dimensions of `field` respectively.  Out-of-bounds indices are clamped to
/// the boundary; the return value is always defined on the closed domain
/// `[0, nx−1] × [0, ny−1]`.
///
/// # Degenerate axes
///
/// When a dimension has extent 1 the corresponding weight is clamped to 0,
/// so the axis does not contribute and the function reduces to nearest-sample.
///
/// # Examples
///
/// ```rust
/// use leto::Array2;
/// use leto_ops::bilinear_index_space;
///
/// let mut a = Array2::<f64>::zeros([2, 2]);
/// a[[0, 0]] = 1.0; a[[1, 0]] = 2.0;
/// a[[0, 1]] = 3.0; a[[1, 1]] = 4.0;
/// // Centre of the quad → average of all four corners.
/// let v = bilinear_index_space(a.view(), 0.5, 0.5);
/// assert!((v - 2.5_f64).abs() < 1e-14);
/// ```
#[must_use]
pub fn bilinear_index_space<T>(field: ArrayView2<T>, ix: T, iy: T) -> T
where
    T: FloatElement + NumericElement + Copy,
{
    let [nx, ny] = field.shape();
    let (i, dxf) = axis_index_and_frac::<T>(ix, nx);
    let (j, dyf) = axis_index_and_frac::<T>(iy, ny);
    let i1 = if nx <= 1 { i } else { i + 1 };
    let j1 = if ny <= 1 { j } else { j + 1 };

    let c00 = field[[i, j]];
    let c10 = field[[i1, j]];
    let c01 = field[[i, j1]];
    let c11 = field[[i1, j1]];

    let one = T::from_f64(1.0);
    let row0 = lerp(c00, c10, dxf, one);
    let row1 = lerp(c01, c11, dxf, one);
    lerp(row0, row1, dyf, one)
}

/// Convenience wrapper accepting an `&Array2<T>` directly.
#[must_use]
#[inline]
pub fn bilinear<T>(field: &Array2<T>, ix: T, iy: T) -> T
where
    T: FloatElement + NumericElement + Copy,
{
    bilinear_index_space(field.view(), ix, iy)
}

// ── Trilinear (3-D) ───────────────────────────────────────────────────────────

/// Trilinear interpolation at fractional array indices `(ix, iy, iz)`.
///
/// `ix`, `iy`, `iz` are fractional zero-based indices into the three dimensions
/// of `field`.  Out-of-bounds indices are clamped; degenerate axes (extent 1)
/// collapse to the single sample without an out-of-bounds access.
///
/// The eight-corner formula is:
///
/// ```text
/// f = (1−tx)(1−ty)(1−tz) f[i,j,k]   + tx(1−ty)(1−tz) f[i+1,j,k]
///   + (1−tx)ty(1−tz) f[i,j+1,k]     + tx ty(1−tz) f[i+1,j+1,k]
///   + (1−tx)(1−ty)tz f[i,j,k+1]     + tx(1−ty)tz f[i+1,j,k+1]
///   + (1−tx)ty tz f[i,j+1,k+1]      + tx ty tz f[i+1,j+1,k+1]
/// ```
///
/// # Examples
///
/// ```rust
/// use leto::Array3;
/// use leto_ops::trilinear_index_space;
///
/// // Constant field → any query point returns the constant.
/// let a = Array3::<f64>::from_elem([4, 4, 4], 7.0);
/// assert!((trilinear_index_space(a.view(), 1.7, 0.3, 2.5) - 7.0_f64).abs() < 1e-14);
/// ```
#[must_use]
pub fn trilinear_index_space<T>(field: ArrayView3<T>, ix: T, iy: T, iz: T) -> T
where
    T: FloatElement + NumericElement + Copy,
{
    let [nx, ny, nz] = field.shape();
    let (i, dxf) = axis_index_and_frac::<T>(ix, nx);
    let (j, dyf) = axis_index_and_frac::<T>(iy, ny);
    let (k, dzf) = axis_index_and_frac::<T>(iz, nz);
    let i1 = if nx <= 1 { i } else { i + 1 };
    let j1 = if ny <= 1 { j } else { j + 1 };
    let k1 = if nz <= 1 { k } else { k + 1 };

    let c000 = field[[i, j, k]];
    let c100 = field[[i1, j, k]];
    let c010 = field[[i, j1, k]];
    let c110 = field[[i1, j1, k]];
    let c001 = field[[i, j, k1]];
    let c101 = field[[i1, j, k1]];
    let c011 = field[[i, j1, k1]];
    let c111 = field[[i1, j1, k1]];

    let one = T::from_f64(1.0);
    // Interpolate along x.
    let c00 = lerp(c000, c100, dxf, one);
    let c10 = lerp(c010, c110, dxf, one);
    let c01 = lerp(c001, c101, dxf, one);
    let c11 = lerp(c011, c111, dxf, one);
    // Interpolate along y.
    let c0 = lerp(c00, c10, dyf, one);
    let c1 = lerp(c01, c11, dyf, one);
    // Interpolate along z.
    lerp(c0, c1, dzf, one)
}

/// Convenience wrapper accepting an `&Array3<T>` directly.
#[must_use]
#[inline]
pub fn trilinear<T>(field: &Array3<T>, ix: T, iy: T, iz: T) -> T
where
    T: FloatElement + NumericElement + Copy,
{
    trilinear_index_space(field.view(), ix, iy, iz)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute `(floor_index, fractional_offset)` for axis of length `n`.
///
/// Degenerate axis (n ≤ 1): always returns `(0, 0.0)` so the caller's
/// `idx + 1` stays at 0 and the axis weight is zero.
#[inline]
fn axis_index_and_frac<T: FloatElement + NumericElement + Copy>(v: T, n: usize) -> (usize, T) {
    if n <= 1 {
        return (0, T::from_f64(0.0));
    }
    let max_idx = n - 2; // largest floor index that keeps idx + 1 in bounds
    let clamped = v.to_f64().clamp(0.0, (n - 1) as f64);
    let idx = (clamped.floor() as usize).min(max_idx);
    let frac = T::from_f64((clamped - idx as f64).clamp(0.0, 1.0));
    (idx, frac)
}

/// Linear blend: `(1 − t) · a + t · b`.
#[inline]
fn lerp<T: FloatElement + NumericElement + Copy>(a: T, b: T, t: T, one: T) -> T {
    a * (one - t) + b * t
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_exact_at_corners() {
        let mut f = Array2::<f64>::zeros([2, 2]);
        f[[0, 0]] = 1.0;
        f[[1, 0]] = 2.0;
        f[[0, 1]] = 3.0;
        f[[1, 1]] = 4.0;
        assert!((bilinear_index_space(f.view(), 0.0, 0.0) - 1.0).abs() < 1e-14);
        assert!((bilinear_index_space(f.view(), 1.0, 0.0) - 2.0).abs() < 1e-14);
        assert!((bilinear_index_space(f.view(), 0.0, 1.0) - 3.0).abs() < 1e-14);
        assert!((bilinear_index_space(f.view(), 1.0, 1.0) - 4.0).abs() < 1e-14);
    }

    #[test]
    fn bilinear_mid_quad_is_average() {
        let mut f = Array2::<f64>::zeros([2, 2]);
        f[[0, 0]] = 1.0;
        f[[1, 0]] = 2.0;
        f[[0, 1]] = 3.0;
        f[[1, 1]] = 4.0;
        assert!((bilinear_index_space(f.view(), 0.5, 0.5) - 2.5).abs() < 1e-14);
    }

    #[test]
    fn bilinear_degenerate_y_axis() {
        let mut f = Array2::<f64>::zeros([4, 1]);
        for i in 0..4 {
            f[[i, 0]] = i as f64;
        }
        // Mid-cell in x between samples 1 and 2 (idx=1, frac=0.5) → 1.5
        assert!((bilinear_index_space(f.view(), 1.5, 0.0) - 1.5).abs() < 1e-14);
    }

    #[test]
    fn trilinear_constant_field() {
        let f = Array3::<f64>::from_elem([5, 4, 3], 7.0);
        assert!((trilinear_index_space(f.view(), 2.3, 1.7, 0.5) - 7.0).abs() < 1e-14);
    }

    #[test]
    fn trilinear_linear_field_exact() {
        // f[i,j,k] = i + 2j + 3k  →  trilinear at (1.5,2.5,0.5) = 1.5 + 5.0 + 1.5 = 8.0
        let mut f = Array3::<f64>::zeros([5, 5, 5]);
        for i in 0..5usize {
            for j in 0..5usize {
                for k in 0..5usize {
                    f[[i, j, k]] = i as f64 + 2.0 * j as f64 + 3.0 * k as f64;
                }
            }
        }
        assert!((trilinear_index_space(f.view(), 1.5, 2.5, 0.5) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn trilinear_degenerate_y_z_axes() {
        // Quasi-1D: nz = ny = 1.
        let mut f = Array3::<f64>::zeros([8, 1, 1]);
        for i in 0..8usize {
            f[[i, 0, 0]] = i as f64;
        }
        for i in 0..7usize {
            let v = trilinear_index_space(f.view(), i as f64 + 0.5, 0.0, 0.0);
            assert!((v - (i as f64 + 0.5)).abs() < 1e-12, "i={i} got {v}");
        }
    }

    #[test]
    fn trilinear_clamped_oob() {
        // Out-of-bounds index clamped to nearest edge.
        let f = Array3::<f64>::from_elem([4, 4, 4], 1.0);
        let v = trilinear_index_space(f.view(), -2.0, 100.0, 3.5_f64);
        assert!((v - 1.0).abs() < 1e-14);
    }
}
