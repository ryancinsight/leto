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

/// Runs `run(first_index, values)` over consecutive runs of a dense output,
/// each about one moirai unit task of work at `element_bytes` per element.
///
/// The caller decides whether to parallelize at all: an elementwise op gates on
/// its own arithmetic and working set. This decides only the task width, from
/// the bytes a unit moves, as moirai ADR 0059 requires, and hands each task a
/// real slice rather than an index range to rebuild. Without the `parallel`
/// feature the whole output runs on the calling thread.
pub(crate) fn for_each_unit_run_mut<T, F>(
    output: &mut [T],
    #[cfg_attr(
        not(feature = "parallel"),
        expect(
            unused_variables,
            reason = "only the parallel arm sizes tasks by bytes"
        )
    )]
    element_bytes: usize,
    run: F,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    #[cfg(feature = "parallel")]
    moirai::for_each_unit_task_mut_with::<moirai::Parallel, _, _, _, _>(
        output,
        1,
        element_bytes,
        || (),
        |(), first_index, values| run(first_index, values),
    );
    #[cfg(not(feature = "parallel"))]
    run(0, output);
}

/// Runs `run(first_unit, count)` over consecutive runs of `units` units the
/// closure addresses itself, each run about one moirai unit task of work at
/// `unit_bytes` per unit.
///
/// The dense counterpart is [`for_each_unit_run_mut`]. This one serves a pass
/// whose units are rows, tiles or output indices rather than a slice the
/// runtime can split, so the caller keeps the disjointness proof for what its
/// indices address and the decision whether to parallelize at all; moirai
/// decides the width from the bytes a unit moves (moirai ADR 0059). Without
/// the `parallel` feature every unit runs on the calling thread.
pub(crate) fn for_each_unit_range<F>(
    units: usize,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(
            unused_variables,
            reason = "only the parallel arm sizes tasks by bytes"
        )
    )]
    unit_bytes: usize,
    run: F,
) where
    F: Fn(usize, usize) + Send + Sync,
{
    #[cfg(feature = "parallel")]
    moirai::for_each_unit_task_range_with::<moirai::Parallel, _, _, _>(
        units,
        unit_bytes,
        || (),
        |(), first_unit, count| run(first_unit, count),
    );
    #[cfg(not(feature = "parallel"))]
    if units > 0 {
        run(0, units);
    }
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
    moirai::for_each_unit_task_mut_with::<moirai::WorkBytes<PARALLEL_MIN_BYTES>, _, _, _, _>(
        output,
        plane_len,
        plane_len.saturating_mul(element_bytes),
        || (),
        |(), first_plane, planes| {
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
