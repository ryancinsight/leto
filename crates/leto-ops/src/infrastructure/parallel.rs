use leto::{TaskPartitionMut, TaskPartitionsMut};

/// Partition and run a 1D loop in parallel using Moirai's work-stealing runtime.
///
/// # Safety
/// The caller must ensure that parallel execution does not violate aliasing invariants.
#[cfg(feature = "parallel")]
pub fn parallel_for<F>(start: usize, end: usize, f: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    let len = end.saturating_sub(start);
    if len == 0 {
        return;
    }
    moirai::for_each_index_with::<moirai::Adaptive, _>(len, move |i| {
        f(start + i);
    });
}

/// Run a loop in parallel chunks using Moirai's work-stealing runtime.
///
/// # Safety
/// The caller must ensure that parallel execution does not violate aliasing invariants.
#[cfg(feature = "parallel")]
pub fn parallel_for_chunks<F>(len: usize, chunk_size: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + 'static,
{
    if len == 0 {
        return;
    }
    if len >= 16384 {
        let num_chunks = len.div_ceil(chunk_size);
        moirai::for_each_index_with::<moirai::Parallel, _>(num_chunks, move |chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(len);
            f(start, end);
        });
    } else {
        f(0, len);
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

#[cfg(test)]
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
        let layout = Layout::new([2, 3], [3, -1], 2);
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

        assert!(result.is_err());
    }
}
