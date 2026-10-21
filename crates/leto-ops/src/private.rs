//! Crate-private sealing point: `Sealed` and the `OpMarker` supertrait.
//!
//! `pub(crate)` visibility is what makes the sealing real: downstream
//! crates can name neither this module nor `Sealed`, so the op-marker
//! traits bounded by [`OpMarker`] admit only this crate's own ZSTs.

/// Sealed marker trait for internal operation markers.
pub trait Sealed {}

/// Root bound for all zero-sized operation-marker traits.
pub trait OpMarker: Sealed + Copy + Send + Sync + 'static {}
impl<T: Sealed + Copy + Send + Sync + 'static> OpMarker for T {}
