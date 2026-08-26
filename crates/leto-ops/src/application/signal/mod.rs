//! Signal-processing mathematical primitives.
//!
//! SSOT for generic DSP utilities in the Atlas simulation stack.
//! All implementations are `f64`-based, pure functions, no external deps.
//!
//! ## Modules
//!
//! - [Window functions](crate::application::signal::window): spectral
//!   apodization coefficients (Hann, Hamming, Blackman, Tukey).
//! - [Phase](crate::application::signal::phase): angle arithmetic on the circle
//!   (`wrap_to_pi`).

pub mod phase;
pub mod window;

pub use phase::wrap_to_pi;
pub use window::{blackman, hamming, hann, tukey};
