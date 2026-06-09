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
    let num_chunks = len.div_ceil(chunk_size);
    moirai::for_each_index_with::<moirai::Adaptive, _>(num_chunks, move |chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(len);
        f(start, end);
    });
}
