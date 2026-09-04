//! Arbitrary-even-order staggered gradient/divergence pair for a Yee leapfrog.
//!
//! # What this is for
//!
//! A velocity-pressure leapfrog needs two operators that are **negative
//! adjoints**, `D = −Gᵀ`, or it has no conserved energy. The second-order
//! [`FiniteDifference3DScheme::StaggeredForward`] and
//! [`FiniteDifference3DScheme::StaggeredBackward`] kernels supply that pair at
//! their fixed order; this supplies it at any even order, which is what a
//! high-accuracy ultrasound or seismic FDTD wants: eighth order in space is
//! standard, and the point of the higher order is fewer points per wavelength
//! for the same phase error, not a smaller residual on a fixed grid.
//!
//! [`FiniteDifference3DScheme::StaggeredForward`]: super::FiniteDifference3DScheme::StaggeredForward
//! [`FiniteDifference3DScheme::StaggeredBackward`]: super::FiniteDifference3DScheme::StaggeredBackward
//!
//! # The pair
//!
//! With `N = order/2` tap pairs and the staggered coefficients `c_n` from
//! [`staggered_first_derivative_coefficients`], the gradient maps cell-centred
//! `p` to face-centred `u` (face `i+½` stored at index `i`), and the divergence
//! maps back:
//!
//! ```text
//!   (G p)_{i+1/2} = (1/Δ) Σ_{n=1..N} c_n ( p_{i+n} − p_{i−n+1} )
//!   (D u)_i       = −(Gᵀ u)_i
//! ```
//!
//! # Why the divergence scatters
//!
//! The divergence is `−Gᵀ` applied directly, which is why it scatters where the
//! gradient gathers: each face sends `∓c_n` of its value to the two cells the
//! gradient drew from, reflected indices included. Writing the transpose out as
//! its own stencil would mean re-deriving the wall closure and hoping it
//! matches; scattering makes `D = −Gᵀ` true by construction, so energy
//! conservation does not depend on getting a boundary case right.
//!
//! # Boundaries
//!
//! Taps outside the grid are reflected about the wall, giving `∂p/∂n = 0`.
//! Cell centres sit at `(i+½)Δ`, so the walls fall between cells and no cell is
//! its own reflection.
//!
//! [`staggered_first_derivative_coefficients`]: super::staggered_first_derivative_coefficients

mod kernels;
#[cfg(test)]
mod tests;

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array3, ArrayView3, LetoError, Result};

use super::coefficients::{staggered_first_derivative_coefficients, TapCoefficients};

/// The spatial axis a gradient or divergence differentiates along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The outermost (slowest-varying) axis of a row-major `Array3`.
    X,
    /// The middle axis.
    Y,
    /// The innermost (contiguous) axis.
    Z,
}

impl Axis {
    /// Position of this axis in a `[nx, ny, nz]` shape.
    #[must_use]
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Staggered gradient/divergence pair at accuracy order `2N`.
///
/// Both operators are grid-shaped in and out: the face `i+½` is stored at index
/// `i`, so a velocity field and a pressure field share one allocation shape.
///
/// # Examples
///
/// ```
/// use leto::Array3;
/// use leto_ops::{Axis, StaggeredLeapfrog3D};
///
/// let op = StaggeredLeapfrog3D::<f64>::new(4, 1.0, 1.0, 1.0).unwrap();
/// assert_eq!(op.order(), 4);
/// assert_eq!(op.halo_width(), 2);
///
/// // A constant field has zero gradient everywhere, walls included.
/// let field = Array3::from_elem([6, 6, 6], 2.5);
/// let mut dst = Array3::zeros([6, 6, 6]);
/// op.gradient_into(Axis::Z, field.view(), &mut dst).unwrap();
/// assert!(dst.as_slice().unwrap().iter().all(|&v| v == 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaggeredLeapfrog3D<T> {
    coefficients: TapCoefficients<T>,
    spacing: [T; 3],
}

impl<T: RealField + FloatElement + Copy> StaggeredLeapfrog3D<T> {
    /// Build the pair for an even accuracy `order` on a grid of the given
    /// spacings.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] for an odd or zero order, an order
    /// beyond the coefficient derivation's verified range
    /// ([`MAX_HALF_ORDER`](super::coefficients::MAX_HALF_ORDER) tap pairs), and
    /// non-positive spacings.
    pub fn new(order: usize, dx: T, dy: T, dz: T) -> Result<Self> {
        if order == 0 || !order.is_multiple_of(2) {
            return Err(LetoError::InvalidInput(format!(
                "StaggeredLeapfrog3D needs an even, non-zero order, got {order}"
            )));
        }
        let zero = <T as NumericElement>::ZERO;
        if dx <= zero || dy <= zero || dz <= zero {
            return Err(LetoError::InvalidInput(
                "StaggeredLeapfrog3D: dx, dy, dz must all be strictly positive".into(),
            ));
        }
        Ok(Self {
            coefficients: staggered_first_derivative_coefficients(order / 2)?,
            spacing: [dx, dy, dz],
        })
    }

    /// Accuracy order `2N`.
    #[must_use]
    #[inline]
    pub fn order(&self) -> usize {
        self.coefficients.order()
    }

    /// Half-width of the stencil in cells — the halo a domain decomposition
    /// must exchange for this order.
    #[must_use]
    #[inline]
    pub fn halo_width(&self) -> usize {
        self.coefficients.half_order()
    }

    /// The derived tap coefficients.
    #[must_use]
    #[inline]
    pub fn coefficients(&self) -> &TapCoefficients<T> {
        &self.coefficients
    }

    /// Per-axis grid spacing `(dx, dy, dz)`.
    #[must_use]
    #[inline]
    pub fn spacing(&self) -> (T, T, T) {
        (self.spacing[0], self.spacing[1], self.spacing[2])
    }

    /// Courant limit as a multiple of `Δx/c`, for `dimensions` spatial axes.
    ///
    /// # Derivation
    ///
    /// The staggered symbol along one axis is `S(θ) = 2 Σ_n c_n sin((n−½)θ)`,
    /// whose magnitude is bounded by `S_max = 2 Σ_n |c_n|`. Leapfrog stability
    /// needs `(c·Δt/2)·|k_eff| ≤ 1` with `|k_eff| = S_max·√D/Δx` in `D`
    /// dimensions, so
    ///
    /// ```text
    ///   Δt ≤ 2Δx / (c · S_max · √D) = Δx / (c · √D · Σ_n |c_n|)
    /// ```
    ///
    /// At order 2 the sum is 1 and this recovers the familiar `1/√3` in 3-D.
    ///
    /// # Why this differs from the collocated limit
    ///
    /// The collocated central-difference table (`1/√3`, `1/√15`, `1/√27`) is a
    /// different scheme's limit. The two agree at order 2 and diverge
    /// immediately after: at order 4 the staggered limit is 0.495 against the
    /// collocated 0.258. Using the collocated number for a staggered run costs
    /// roughly half the achievable step for no accuracy gain.
    #[must_use]
    pub fn cfl_limit(&self, dimensions: usize) -> T {
        let dimensions = T::from_f64(dimensions as f64);
        (dimensions.sqrt() * self.coefficients.absolute_sum()).recip()
    }

    /// Gradient along `axis`: cell-centred `field` to face-centred `dst`, with
    /// face `i+½` stored at index `i`. Both are grid-shaped.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when `dst` does not have the same
    /// shape as `field`.
    pub fn gradient_into(
        &self,
        axis: Axis,
        field: ArrayView3<'_, T>,
        dst: &mut Array3<T>,
    ) -> Result<()> {
        let shape = field.shape();
        assert_grid_shaped(dst.shape(), shape, "gradient")?;
        kernels::gradient(self, axis, field, dst, shape);
        Ok(())
    }

    /// Divergence along `axis`: face-centred `field` back to cell-centred
    /// `dst`. This is `−Gᵀ` applied directly.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when `dst` does not have the same
    /// shape as `field`.
    pub fn divergence_into(
        &self,
        axis: Axis,
        field: ArrayView3<'_, T>,
        dst: &mut Array3<T>,
    ) -> Result<()> {
        let shape = field.shape();
        assert_grid_shaped(dst.shape(), shape, "divergence")?;
        kernels::divergence(self, axis, field, dst, shape);
        Ok(())
    }

    /// Axis index, extent along it, and the reciprocal spacing.
    fn axis_geometry(&self, axis: Axis, shape: [usize; 3]) -> (usize, isize, T) {
        let index = axis.index();
        let extent = isize::try_from(shape[index]).unwrap_or(isize::MAX);
        (index, extent, self.spacing[index].recip())
    }
}

/// Both operators keep the grid shape; a mismatch is a caller error, reported
/// rather than left to a debug-only assertion that would corrupt in release.
#[inline]
fn assert_grid_shaped(actual: [usize; 3], expected: [usize; 3], what: &str) -> Result<()> {
    if actual != expected {
        return Err(LetoError::InvalidInput(format!(
            "StaggeredLeapfrog3D {what}: dst shape {actual:?} does not match field shape {expected:?}"
        )));
    }
    Ok(())
}
