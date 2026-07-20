//! Typed contracts shared by CPU and accelerator finite-difference stencils.

use aequitas::systems::si::{quantities::Length, units::Meter};
use eunomia::{FloatElement, NumericElement};
use thiserror::Error;

/// Boundary condition applied by a Cartesian finite-difference stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Homogeneous Dirichlet: `u = 0` on boundaries.
    Dirichlet,
    /// Homogeneous Neumann: `∂u/∂n = 0` on boundaries.
    Neumann,
    /// Periodic wrapping across domain boundaries.
    Periodic,
}

impl From<BoundaryCondition> for u32 {
    #[inline]
    fn from(value: BoundaryCondition) -> Self {
        match value {
            BoundaryCondition::Dirichlet => 0,
            BoundaryCondition::Neumann => 1,
            BoundaryCondition::Periodic => 2,
        }
    }
}

/// Sign convention for a Laplacian operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianPolarity {
    /// The differential operator `∇²`.
    Laplacian,
    /// The positive-semidefinite operator `-∇²`.
    NegativeLaplacian,
}

/// Typed Laplacian contract failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LaplacianError {
    /// An axis is too short for the boundary stencil.
    #[error("Laplacian grid axes must contain at least two points: nx={nx}, ny={ny}")]
    GridTooSmall {
        /// Number of x-axis points.
        nx: usize,
        /// Number of y-axis points.
        ny: usize,
    },
    /// The flattened grid size does not fit in `usize`.
    #[error("Laplacian grid size overflows usize: nx={nx}, ny={ny}")]
    GridSizeOverflow {
        /// Number of x-axis points.
        nx: usize,
        /// Number of y-axis points.
        ny: usize,
    },
    /// A physical grid spacing is non-finite or non-positive.
    #[error("Laplacian spacing must be finite and positive: dx={dx}, dy={dy}")]
    InvalidSpacing {
        /// Debug representation of x spacing in meters.
        dx: String,
        /// Debug representation of y spacing in meters.
        dy: String,
    },
    /// The input array length does not match the flattened grid.
    #[error("Laplacian input length mismatch: expected {expected}, actual {actual}")]
    InputLength {
        /// Required flattened length.
        expected: usize,
        /// Supplied input length.
        actual: usize,
    },
    /// The output array length does not match the flattened grid.
    #[error("Laplacian output length mismatch: expected {expected}, actual {actual}")]
    OutputLength {
        /// Required flattened length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
}

/// Validated two-dimensional Cartesian Laplacian contract.
///
/// Spacing crosses the boundary as [`Length`] and is reduced once to inverse
/// squared SI spacing in the native precision of `T`. The same contract drives
/// Leto CPU evaluation and Hephaestus accelerator dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Laplacian2D<T> {
    nx: usize,
    ny: usize,
    len: usize,
    inverse_spacing_squared: [T; 2],
    boundary: BoundaryCondition,
    polarity: LaplacianPolarity,
}

impl<T> Laplacian2D<T>
where
    T: FloatElement,
{
    /// Validate a uniform endpoint-inclusive Cartesian grid.
    ///
    /// # Errors
    ///
    /// Returns [`LaplacianError`] when an axis has fewer than two points, the
    /// flattened size overflows, or either spacing is non-finite or non-positive.
    pub fn new(
        nx: usize,
        ny: usize,
        dx: Length<T>,
        dy: Length<T>,
        boundary: BoundaryCondition,
    ) -> Result<Self, LaplacianError> {
        if nx < 2 || ny < 2 {
            return Err(LaplacianError::GridTooSmall { nx, ny });
        }
        let len = nx
            .checked_mul(ny)
            .ok_or(LaplacianError::GridSizeOverflow { nx, ny })?;
        let dx_m = dx.in_unit::<Meter>();
        let dy_m = dy.in_unit::<Meter>();
        if !dx_m.is_finite()
            || dx_m <= <T as NumericElement>::ZERO
            || !dy_m.is_finite()
            || dy_m <= <T as NumericElement>::ZERO
        {
            return Err(LaplacianError::InvalidSpacing {
                dx: format!("{dx_m:?} m"),
                dy: format!("{dy_m:?} m"),
            });
        }

        Ok(Self {
            nx,
            ny,
            len,
            inverse_spacing_squared: [dx_m.recip().powi(2), dy_m.recip().powi(2)],
            boundary,
            polarity: LaplacianPolarity::Laplacian,
        })
    }

    /// Select the operator sign convention.
    #[must_use]
    pub const fn with_polarity(mut self, polarity: LaplacianPolarity) -> Self {
        self.polarity = polarity;
        self
    }

    /// Number of x-axis points.
    #[must_use]
    pub const fn nx(&self) -> usize {
        self.nx
    }

    /// Number of y-axis points.
    #[must_use]
    pub const fn ny(&self) -> usize {
        self.ny
    }

    /// Required flattened array length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the flattened grid is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Boundary condition encoded by this stencil.
    #[must_use]
    pub const fn boundary(&self) -> BoundaryCondition {
        self.boundary
    }

    /// Operator sign convention.
    #[must_use]
    pub const fn polarity(&self) -> LaplacianPolarity {
        self.polarity
    }

    /// Signed inverse squared spacing for x and y.
    #[must_use]
    pub fn signed_inverse_spacing_squared(&self) -> [T; 2] {
        match self.polarity {
            LaplacianPolarity::Laplacian => self.inverse_spacing_squared,
            LaplacianPolarity::NegativeLaplacian => [
                <T as NumericElement>::ZERO - self.inverse_spacing_squared[0],
                <T as NumericElement>::ZERO - self.inverse_spacing_squared[1],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_positive_spacing() {
        let error = Laplacian2D::new(
            4,
            4,
            Length::from_unit::<Meter>(0.0f32),
            Length::from_unit::<Meter>(1.0f32),
            BoundaryCondition::Dirichlet,
        )
        .expect_err("zero spacing must be rejected");

        assert!(matches!(error, LaplacianError::InvalidSpacing { .. }));
    }
}
