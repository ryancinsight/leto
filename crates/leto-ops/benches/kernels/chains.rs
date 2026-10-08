//! Operator chain benchmark.

use super::*;

/// Operator-chain instrument for the core elementwise operator tier (ADR 0004).
///
/// Each pair is the same expression in the two receiver forms, timed in one run
/// so the comparison is immune to cross-run drift on a hybrid P/E-core host:
/// `borrowed` re-borrows every intermediate and allocates one array per binary
/// operator (n-1 for n terms); `owned` lets the middle temporary be consumed and
/// reused, allocating once regardless of term count. Operand construction is
/// outside the timed region; the closure returns its result so the chain cannot
/// be elided.
///
/// Two sizes because the forms differ in allocation, and allocation cost is
/// size-regime-dependent: 64x64 (32 KiB/array) keeps every intermediate in the
/// allocator's cache and cache-resident, so the timing reflects the traversal
/// itself; 256x256 (512 KiB/array) is the regime where each freed intermediate
/// returns to the OS and the borrowed form additionally pays first-touch page
/// faults. The 64x64 pair is the conservative comparison; 256x256 shows the
/// larger but noisier end (read its CI, not just its median).
fn bench_operator_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("operator_chain");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(1));

    for &n in &[64usize, 256] {
        let a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
        let b = Array::from_shape_vec([n, n], pinned_values(n * n, 0.5)).unwrap();
        let d = Array::from_shape_vec([n, n], pinned_values(n * n, 0.25)).unwrap();

        group.bench_function(format!("borrowed_3term_{n}x{n}"), |bencher| {
            bencher.iter(|| &(black_box(&a) + black_box(&b)) + black_box(&d));
        });
        group.bench_function(format!("owned_3term_{n}x{n}"), |bencher| {
            bencher.iter(|| black_box(&a) + black_box(&b) + black_box(&d));
        });
    }

    // Term-count scaling at the cache-resident size: the borrowed form's
    // allocation count grows with n, the owned form's does not.
    let n = 64usize;
    let a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
    let b = Array::from_shape_vec([n, n], pinned_values(n * n, 0.5)).unwrap();
    let d = Array::from_shape_vec([n, n], pinned_values(n * n, 0.25)).unwrap();
    let e = Array::from_shape_vec([n, n], pinned_values(n * n, 0.125)).unwrap();
    group.bench_function("borrowed_5term_64x64", |bencher| {
        bencher.iter(|| {
            &(&(&(black_box(&a) + black_box(&b)) + black_box(&d)) + black_box(&e)) + black_box(&a)
        });
    });
    group.bench_function("owned_5term_64x64", |bencher| {
        bencher
            .iter(|| black_box(&a) + black_box(&b) + black_box(&d) + black_box(&e) + black_box(&a));
    });
    group.finish();
}
