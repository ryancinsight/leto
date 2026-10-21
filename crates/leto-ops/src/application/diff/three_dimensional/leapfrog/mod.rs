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
mod map;
mod operator;
#[cfg(test)]
mod tests;

use operator::assert_grid_shaped;
pub use operator::{Axis, StaggeredLeapfrog3D};
