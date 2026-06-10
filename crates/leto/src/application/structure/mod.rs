//! Structural array operations that compose or partition arrays along axes.
//!
//! `concat` and `pad` allocate C-contiguous owned output and are
//! rank-preserving (rank `N` in, rank `N` out); `split` returns zero-copy
//! rank-`N` subviews; `stack` is rank-increasing (`N` in, `N + 1` out) via the
//! `InsertAxis` rank helper.

mod concat;
mod pad;
mod split;
mod stack;

pub use concat::concat;
pub use pad::{pad, PadWidth};
pub use split::split;
pub use stack::stack;
