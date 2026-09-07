//! Connectionist temporal classification over borrowed log-probability views.
//!
//! [`CtcState`] retains the forward and suffix recurrences in the selected
//! scalar precision. Backward differentiates independent log-probabilities;
//! composing it with log-softmax yields the usual logit gradient.

mod backward;
mod error;
mod forward;
mod state;
mod weight;

pub use error::CtcError;
pub use state::CtcState;
