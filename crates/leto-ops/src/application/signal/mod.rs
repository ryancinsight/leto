//! Signal-processing mathematical primitives.
//!
//! SSOT for generic DSP utilities in the Atlas simulation stack.
//! All implementations are `f64`-based, pure functions, no external deps.
//!
//! ## Modules
//!
//! - [`window`]: Spectral apodization/window coefficients (Hann, Hamming, Blackman, Tukey).
//! - [`phase`]: Phase-angle arithmetic on the circle (`wrap_to_pi`).

pub mod phase;
pub mod window;

pub use phase::wrap_to_pi;
pub use window::{blackman, hamming, hann, tukey};
