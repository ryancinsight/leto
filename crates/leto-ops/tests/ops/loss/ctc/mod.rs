//! CTC loss tests, split by concern.
//!
//! [`fixtures`] holds the shared fixture helpers; [`forward`] the analytical
//! loss and normalization oracles; [`limits`] the boundary and degenerate-
//! extent contracts; [`validation`] the typed rejection surface; and
//! [`gradient`] the central-difference differentiability proof.

mod fixtures;
mod forward;
mod gradient;
mod limits;
mod validation;
