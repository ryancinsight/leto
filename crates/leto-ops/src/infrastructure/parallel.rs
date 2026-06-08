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
