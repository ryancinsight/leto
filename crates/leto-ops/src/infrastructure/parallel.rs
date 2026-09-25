#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

#[cfg(feature = "parallel")]
use leto::{TaskPartitionMut, TaskPartitionsMut};

/// Bytes a pass moves before it spreads over the runtime's workers.
///
/// Apollo's 3-D pass probe (`dimension_3d::pass_attribution`, 2026-09-09)
/// measured the crossover on this runtime: a 32³ volume — 32 matrices of
/// 16 KiB, 512 KiB in all — ran *slower* spread over tasks (28.5 µs against
/// 21.9 serial), because at eleven microseconds of work the runtime's
/// spawn-and-join is the larger part, while the 64³ batch, 4 MiB, won by 2.4x
/// to 3.9x. One mebibyte sits between the two. The batched complex transpose
/// and the leapfrog sweeps share it; the probe and `benches/leapfrog.rs` are
/// the instruments that move it.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_BYTES: usize = 1024 * 1024;

/// Element count past which a compute-bound elementwise op parallelizes.
///
/// An element count is the correct unit here precisely because the op is
/// compute-bound: its cost scales with the number of elements, not with the
/// working-set bytes, so the gate must not scale by scalar width. Measured on a
/// 36 MiB-L3 AVX2 host, in-place `sin` over `f64` at exactly this length runs
/// 0.186 ns/elem parallel against 1.210 ns/elem sequential — parallelism already
/// pays 6.5x at the gate, so it opens no later than it should.
///
/// Formerly the private `PARALLEL_THRESHOLD` of `application::unary`.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_ELEMENTS: usize = 65536;

/// Element count past which an axis reduction over `out_size` outputs — each
/// scanning `axis_len` inputs — parallelizes.
///
/// Reductions benefit from parallelism at lower element counts than elementwise
/// maps because each output element requires a full axis scan (O(N/out_size)
/// reads). Formerly the private `PARALLEL_THRESHOLD` of `application::reduction`.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_REDUCTION_OUTPUTS: usize = 32768;

/// Multiply-accumulate count past which a dense matmul parallelizes.
/// Formerly the literal in `application::matrix`.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_MATMUL_MACS: usize = 262_144;

/// Minimum row count a dense matmul must reach before it parallelizes.
/// Formerly the literal in `application::matrix`.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_MATMUL_ROWS: usize = 64;

/// Shared dense-slice task dispatch for unit-task wrappers.
#[cfg(feature = "parallel")]
#[inline]
fn dispatch_unit_task_mut_with<P, T, F>(
    output: &mut [T],
    unit_len: usize,
    element_bytes: usize,
    run: F,
) where
    P: moirai::ExecutionPolicy,
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    moirai::for_each_unit_task_mut_with::<P, _, _, _, _>(
        output,
        unit_len,
        unit_len.saturating_mul(element_bytes),
        || (),
        |(), first_unit, values| run(first_unit, values),
    );
}

/// Whether a bandwidth-bound elementwise op over `len` elements of `T` with
/// `operands` operands (reads plus writes) should run in parallel.
///
/// A binary map reads two operands and writes one, so its working set is
/// `operands · len · size_of::<T>()`. The op is memory-bandwidth-bound, and one
/// core nearly saturates that bandwidth while the data is resident in the shared
/// last-level cache; parallelism pays only once the working set spills past the
/// LLC, where additional cores add DRAM bandwidth. Below that, thread-dispatch
/// overhead is pure loss.
///
/// Replaces a fixed 65536-element gate (`256 KB` L2 / `4 B` f32) that ignored
/// both element size and arithmetic intensity, parallelizing bandwidth-bound
/// ops far too eagerly — a 64k `f64` `add` (1.5 MB working set) ran ~3x slower
/// parallel than serial. See gap_audit `2026-07-19 Parallel Threshold`.
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn parallelize_bandwidth_bound<T>(len: usize, operands: usize) -> bool {
    let working_set = len
        .saturating_mul(operands)
        .saturating_mul(core::mem::size_of::<T>());
    working_set > crate::infrastructure::cache::cached_cache_geometry().l3_bytes()
}

/// Whether a compute-bound elementwise op over `len` elements should run in
/// parallel (see [`PARALLEL_MIN_ELEMENTS`]).
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn parallelize_compute_bound(len: usize) -> bool {
    len >= PARALLEL_MIN_ELEMENTS
}

/// Runs `run(first_index, values)` over consecutive runs of a dense output,
/// each about one moirai unit task of work at `element_bytes` per element.
///
/// The caller decides whether to parallelize at all: an elementwise op gates on
/// its own arithmetic and working set. This decides only the task width, from
/// the bytes a unit moves, as moirai ADR 0059 requires, and hands each task a
/// real slice rather than an index range to rebuild.
#[cfg(feature = "parallel")]
pub(crate) fn for_each_unit_run_mut<T, F>(output: &mut [T], element_bytes: usize, run: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    schedule_unit_runs_mut(output, element_bytes, &run);
}

/// The scheduling half of [`for_each_unit_run_mut`], generic over the element
/// type only. Every kernel passes its own closure type; erasing it here
/// instantiates moirai's task split, job storage, and panic capture once per
/// element type instead of once per kernel, scalar, and rank. The cost is one
/// indirect call per unit task; the run body keeps its inlined element loop.
// dyn exception: type erasure at unit-task granularity bounds the scheduler's
// instantiation count; the per-element loop stays monomorphized inside `run`.
#[cfg(feature = "parallel")]
fn schedule_unit_runs_mut<T: Send>(
    output: &mut [T],
    element_bytes: usize,
    run: &(dyn Fn(usize, &mut [T]) + Sync),
) {
    moirai::for_each_unit_task_mut_with::<moirai::Parallel, _, _, _, _>(
        output,
        1,
        element_bytes,
        || (),
        |(), first_index, values| run(first_index, values),
    );
}

/// Runs `run(first_unit, count)` over consecutive runs of `units` units the
/// closure addresses itself, each run about one moirai unit task of work at
/// `unit_bytes` per unit.
///
/// The dense counterpart is [`for_each_unit_run_mut`]. This one serves a pass
/// whose units are rows, tiles or output indices rather than a slice the
/// runtime can split, so the caller keeps the disjointness proof for what its
/// indices address and the decision whether to parallelize at all; moirai
/// decides the width from the bytes a unit moves (moirai ADR 0059).
#[cfg(feature = "parallel")]
pub(crate) fn for_each_unit_range<F>(units: usize, unit_bytes: usize, run: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    schedule_unit_ranges(units, unit_bytes, &run);
}

/// The scheduling half of [`for_each_unit_range`], compiled once. Erasing the
/// caller's closure keeps moirai's scheduler out of every kernel's
/// instantiation; each unit task pays one indirect call into a body that keeps
/// its inlined row or tile loop.
// dyn exception: type erasure at unit-task granularity bounds the scheduler's
// instantiation count; the per-element loop stays monomorphized inside `run`.
#[cfg(feature = "parallel")]
fn schedule_unit_ranges(units: usize, unit_bytes: usize, run: &(dyn Fn(usize, usize) + Sync)) {
    moirai::for_each_unit_task_range_with::<moirai::Parallel, _, _, _>(
        units,
        unit_bytes,
        || (),
        |(), first_unit, count| run(first_unit, count),
    );
}

/// Runs `plane(&mut state, index, planes)` over the x-planes of `K` dense
/// outputs of one length in lockstep: `planes[d]` is plane `index` of output
/// `d`, and `state()` runs once per unit task, so every plane a task owns
/// sees the same state.
///
/// Any `K` is one parallel region. A kernel whose outputs share one
/// derivative pass writes all of them in that pass rather than repeating it
/// per output -- and rather than paying a region per output, which on a
/// fused elastic step measured slower than the unfused passes it replaced.
///
/// A kernel that needs row scratch builds it here rather than per plane, so
/// the allocation count follows the task count instead of the grid. Without
/// the `parallel` feature one state serves the whole output.
pub(crate) fn for_each_plane_mut_many_with<T, S, I, F, const K: usize>(
    outputs: [&mut [T]; K],
    plane_len: usize,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(
            unused_variables,
            reason = "only the parallel arm sizes tasks by bytes"
        )
    )]
    element_bytes: usize,
    state: I,
    plane: F,
) where
    T: Send,
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, [&mut [T]; K]) + Send + Sync,
{
    if plane_len == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    moirai::for_each_unit_task_many_mut_with::<moirai::WorkBytes<PARALLEL_MIN_BYTES>, _, _, _, _, K>(
        outputs,
        plane_len,
        plane_len.saturating_mul(element_bytes),
        state,
        |task_state, first_plane, runs| {
            let planes = runs.first().map_or(0, |run| run.len() / plane_len);
            let mut cursors = runs.map(|run| run.chunks_exact_mut(plane_len));
            for offset in 0..planes {
                let lockstep = core::array::from_fn(|d| {
                    cursors[d]
                        .next()
                        .expect("invariant: every run holds the same plane count")
                });
                plane(task_state, first_plane + offset, lockstep);
            }
        },
    );
    #[cfg(not(feature = "parallel"))]
    {
        let mut task_state = state();
        let planes = outputs.first().map_or(0, |output| output.len() / plane_len);
        let mut cursors = outputs.map(|output| output.chunks_exact_mut(plane_len));
        for index in 0..planes {
            let lockstep = core::array::from_fn(|d| {
                cursors[d]
                    .next()
                    .expect("invariant: every output holds the same plane count")
            });
            plane(&mut task_state, index, lockstep);
        }
    }
}

/// Runs `plane(index, values)` over the whole x-planes of a C-order 3-D output
/// whose planes hold `plane_len` elements: `index` is the plane's x index and
/// `values` its elements in storage order.
///
/// With the `parallel` feature, consecutive planes spread over moirai unit
/// tasks (moirai ADR 0059) once the sweep moves [`PARALLEL_MIN_BYTES`];
/// `element_bytes` counts one output element and every input element `plane`
/// reads beside it. Without the feature the planes run in order on the calling
/// thread. Either way each plane is written by exactly one call, so a `plane`
/// body that reads only shared inputs computes the same values in both.
pub(crate) fn for_each_plane_mut<T, F>(
    output: &mut [T],
    plane_len: usize,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(
            unused_variables,
            reason = "only the parallel arm sizes tasks by bytes"
        )
    )]
    element_bytes: usize,
    plane: F,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if plane_len == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    dispatch_unit_task_mut_with::<moirai::WorkBytes<PARALLEL_MIN_BYTES>, _, _>(
        output,
        plane_len,
        element_bytes,
        |first_plane, planes| {
            for (offset, values) in planes.chunks_exact_mut(plane_len).enumerate() {
                plane(first_plane + offset, values);
            }
        },
    );
    #[cfg(not(feature = "parallel"))]
    for (index, values) in output.chunks_exact_mut(plane_len).enumerate() {
        plane(index, values);
    }
}

/// Consume disjoint Leto task partitions through a caller-owned Moirai runtime.
///
/// Leto owns layout validation and physical-aliasing proof; this adapter owns
/// admission and completion. Each partition is moved into at most one scoped
/// task, and the scope waits for every admitted callback before returning.
/// Sequential policy selection invokes callbacks on the caller without
/// scheduler admission.
///
/// # Errors
/// Returns the Moirai executor error for shutdown, admission, or a panicking
/// scoped task. A resource-exhausted admission is handled by Moirai's caller
/// lane and is not returned as an error.
#[cfg(feature = "parallel")]
pub fn for_each_task_partition_mut_with<'scope, P, T, F, const N: usize>(
    runtime: &'scope moirai::Moirai,
    partitions: TaskPartitionsMut<'scope, T, N>,
    f: F,
) -> moirai::ExecutorResult<()>
where
    P: moirai::ExecutionPolicy,
    T: Send + 'scope,
    F: Fn(TaskPartitionMut<'scope, T, N>) + Send + Sync + 'scope,
{
    if !P::parallelize(partitions.len()) {
        for partition in partitions {
            f(partition);
        }
        return Ok(());
    }

    runtime.scope(|scope| {
        let f = &f;
        for partition in partitions {
            scope.spawn(move |_| f(partition))?;
        }
        Ok(())
    })
}

/// Consume Leto task partitions through Moirai's adaptive global runtime.
///
/// Use [`for_each_task_partition_mut_with`] when the caller owns a runtime or
/// must choose a compile-time execution policy explicitly.
///
/// # Errors
/// Returns the Moirai executor error for shutdown, admission, or a panicking
/// scoped task.
#[cfg(feature = "parallel")]
pub fn for_each_task_partition_mut<'scope, T, F, const N: usize>(
    partitions: TaskPartitionsMut<'scope, T, N>,
    f: F,
) -> moirai::ExecutorResult<()>
where
    T: Send + 'scope,
    F: Fn(TaskPartitionMut<'scope, T, N>) + Send + Sync + 'scope,
{
    for_each_task_partition_mut_with::<moirai::Adaptive, _, _, _>(moirai::global(), partitions, f)
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use leto::{Array, ArrayViewMut, Layout, VecStorage};
    use moirai::{Moirai, Parallel, Sequential};

    fn values<const N: usize>(array: &Array<i32, VecStorage<i32>, N>) -> Vec<i32> {
        array.iter().copied().collect()
    }

    #[test]
    fn sequential_policy_consumes_partitions_in_logical_order() {
        let runtime = Moirai::builder().worker_threads(1).build().unwrap();
        let mut array = Array::from_shape_vec([2, 3], vec![0; 6]).unwrap();
        let partitions = array.task_partitions_mut(2).unwrap();

        for_each_task_partition_mut_with::<Sequential, _, _, _>(
            &runtime,
            partitions,
            |partition| {
                let start = partition.logical_range().start as i32;
                for (offset, value) in partition.into_iter().enumerate() {
                    *value = start + offset as i32;
                }
            },
        )
        .unwrap();

        assert_eq!(values(&array), vec![0, 1, 2, 3, 4, 5]);
        runtime.shutdown();
    }

    #[test]
    fn parallel_policy_updates_strided_negative_layout() {
        let runtime = Moirai::builder().worker_threads(2).build().unwrap();
        let mut storage = vec![-1; 6];
        let layout =
            Layout::try_new([2, 3], [3, -1], 2).expect("invariant: fixture layout is valid");
        let view = ArrayViewMut::try_new(layout, &mut storage).unwrap();
        let partitions = view.task_partitions_mut(2).unwrap();

        for_each_task_partition_mut_with::<Parallel, _, _, _>(&runtime, partitions, |partition| {
            let start = partition.logical_range().start as i32;
            for (offset, value) in partition.into_iter().enumerate() {
                *value = start + offset as i32;
            }
        })
        .unwrap();

        assert_eq!(storage, vec![2, 1, 0, 5, 4, 3]);
        runtime.shutdown();
    }

    #[test]
    fn shutdown_is_reported_before_partition_callback_runs() {
        let runtime = Moirai::builder().worker_threads(1).build().unwrap();
        let mut array = Array::from_shape_vec([4], vec![0; 4]).unwrap();
        let partitions = array.task_partitions_mut(1).unwrap();
        runtime.shutdown();

        let result =
            for_each_task_partition_mut_with::<Parallel, _, _, _>(&runtime, partitions, |_| {
                panic!("shutdown must reject before callback")
            });

        assert_eq!(result, Err(moirai::ExecutorError::ShuttingDown));
    }
}
