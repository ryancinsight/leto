//! The named-op contract and its shared entry points.

use super::super::traversal::{map_into_gated, mapv};
use crate::domain::RealScalar;
use leto::{Array, ArrayView, ArrayViewMut, Result, VecStorage};

/// Zero-sized (or value-carrying) named real unary operation contract.
///
/// Implementors route through the shared [`map_into`]/[`mapv`] traversal via
/// [`unary_map_into`]/[`unary_map`]; no implementor defines its own traversal.
pub trait UnaryOp<T: RealScalar>: Copy + Send + Sync + 'static {
    /// Apply the scalar operation.
    fn apply(&self, x: T) -> T;

    /// Whether this op is compute-bound — heavy enough per element that
    /// parallelism pays even while the data is cache-resident (transcendentals,
    /// `powf`). Bandwidth-bound ops (`neg`, `abs`: a trivial op over each element)
    /// override to `false` and gate parallelism on working-set-vs-LLC, like
    /// binary elementwise ops, so they are not parallelized into a slowdown.
    const COMPUTE_BOUND: bool = true;
}
/// Apply a named unary operation into caller-owned output through the shared
/// traversal kernel.
#[inline]
pub fn unary_map_into<T, Op, const N: usize>(
    op: Op,
    input: &ArrayView<'_, T, N>,
    output: &mut ArrayViewMut<'_, T, N>,
) -> Result<()>
where
    T: RealScalar,
    Op: UnaryOp<T>,
{
    map_into_gated(input, output, move |x| op.apply(x), Op::COMPUTE_BOUND)
}
/// Apply a named unary operation, allocating a C-contiguous output, through the
/// shared traversal kernel.
#[inline]
pub fn unary_map<T, Op, const N: usize>(
    op: Op,
    input: &ArrayView<'_, T, N>,
) -> Result<Array<T, VecStorage<T>, N>>
where
    T: RealScalar,
    Op: UnaryOp<T>,
{
    mapv(input, move |x| op.apply(x))
}
