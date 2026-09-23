//! Sweep kernels behind [`StaggeredLeapfrog3D`].
//!
//! Each kernel picks its traversal from where the differentiated axis sits in
//! row-major storage: the innermost axis is a contiguous line, the outer two
//! are strided blocks whose taps zip across whole blocks rather than cells.
//! A non-contiguous array falls back to three-index addressing, which computes
//! the same sums in the same order.
//!
//! The two operator families live one per leaf module — [`gradient`] sweeps
//! forward, [`divergence`] is its transpose — and share the tap window and
//! reflection helpers held here.

mod divergence;
mod gradient;

pub(super) use divergence::{divergence, divergence_row};
pub(super) use gradient::{gradient, gradient_row, plane_reads};

use super::StaggeredLeapfrog3D;
use eunomia::{FloatElement, NumericElement, RealField};

/// One window of `2·halo` source cells split at `halo` feeds one output cell of
/// either operator: `Σ_n c_n (hi[n−1] − lo[halo−n])`. The gradient's window for
/// face `i+½` starts at `i+1−halo`; the divergence's window for cell `j` starts
/// at `j−halo` — the transpose shifts the output by one cell and changes
/// nothing else. Taps accumulate in ascending `n`, the order the indexed
/// reference uses, so the two agree bit for bit.
pub(super) fn window_sum<T: RealField + FloatElement + Copy>(
    op: &StaggeredLeapfrog3D<T>,
    window: &[T],
) -> T {
    let taps = op.coefficients().taps();
    let (lo, hi) = window.split_at(taps.len());
    taps.iter()
        .zip(hi)
        .zip(lo.iter().rev())
        .fold(<T as NumericElement>::ZERO, |sum, ((&c, &hi), &lo)| {
            sum + c * (hi - lo)
        })
}

/// Mirror an index about the nearest wall until it lands inside `0..extent`.
///
/// Cell centres sit at `(i+½)Δ`, so the walls fall *between* cells and the
/// mirror is `−1−m` at the low end and `2·extent−1−m` at the high end — no cell
/// is its own reflection. The loop repeats for stencils deeper than the grid,
/// which only arises for extents below the halo width; it terminates for any
/// `extent ≥ 1`.
pub(super) fn reflect(mut m: isize, extent: isize) -> usize {
    debug_assert!(extent >= 1, "reflection needs a non-empty axis");
    loop {
        if m < 0 {
            m = -1 - m;
        } else if m >= extent {
            m = 2 * extent - 1 - m;
        } else {
            return usize::try_from(m).expect("invariant: m is non-negative and below extent");
        }
    }
}
