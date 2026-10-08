//! Three-dimensional finite-difference tests, split by concern.
//!
//! [`exactness`] pins the central-scheme polynomial exactness and boundary
//! fallback, [`staggered`] the face-centered operators and shape rejections,
//! [`contracts`] the dispersion ordering and input contracts, and
//! [`adjoint`] the transpose identities.

mod adjoint;
mod contracts;
mod exactness;
mod staggered;
