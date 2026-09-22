//! Fourth-order central first derivative behind
//! [`FiniteDifference3DScheme::CentralFourthOrder`](super::FiniteDifference3DScheme).
//!
//! A coordinate `c` on an axis of `n` points takes one stencil:
//!
//! | Position | Stencil | Order |
//! |---|---|---|
//! | `2 ≤ c ≤ n − 3` | `(−f[c+2] + 8f[c+1] − 8f[c−1] + f[c−2]) / 12Δ` | 4 |
//! | `c ∈ {1, n − 2}` | `(f[c+1] − f[c−1]) / 2Δ` | 2 |
//! | `c = 0` | `(f[1] − f[0]) / Δ` | 1 |
//! | `c = n − 1` | `(f[n−1] − f[n−2]) / Δ` | 1 |
//! | `n = 1` | `0` | — |
//!
//! The first matching row wins, so an axis shorter than five points takes only
//! the lower-order closures (a two-point axis is one-sided at both ends), and a
//! singleton axis has no variation along it. Each stencil is written as one
//! arithmetic chain shared by both traversals, so they agree bit for bit.
//!
//! C-contiguous fields sweep whole lanes, a task per run of x-planes sized by
//! the bytes a plane moves; any other layout walks the logical indices on the
//! calling thread. [`stencil`] holds the shared coefficient math, [`dense`]
//! and [`strided`] the two traversals, and [`dispatch`] the entry points that
//! choose between them by field contiguity.

mod dense;
mod dispatch;
mod stencil;
mod strided;
#[cfg(test)]
mod tests;

pub(super) use dispatch::{
    central4_divergence_into, central4_into, central4_map_into, central4_map_triple_into,
};
// Not re-exported past this module: only the `parallel`-gated test in
// `tests` reads it, via `super::ELEMENTS_PER_UNIT`, to size its
// parallel-floor fixture.
#[cfg(all(test, feature = "parallel"))]
use stencil::ELEMENTS_PER_UNIT;
