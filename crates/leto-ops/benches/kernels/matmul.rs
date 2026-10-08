//! Dense matmul kernel throughput.

use super::*;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[64usize, 256] {
        let a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        let b = Array::from_shape_vec([n, n], pinned_values(n * n, 2.0e-3)).unwrap();
        group.bench_function(format!("dense_{n}x{n}"), |bencher| {
            bencher.iter_batched(
                || Array::zeros([n, n]),
                |mut out| {
                    matmul(
                        black_box(&a.view()),
                        black_box(&b.view()),
                        &mut out.view_mut(),
                    )
                    .unwrap();
                    out
                },
                BatchSize::LargeInput,
            );
        });
    }

    // Wide strided coverage exercises the row-block policy through the
    // generic layout route. The identical prepared views are used for the
    // automatic policy and the fixed 32-row control below.
    let rows = 64usize;
    let shared = 64usize;
    let cols = 4_096usize;
    let wide_lhs_storage = Array::from_shape_vec(
        [rows * 2, shared * 2],
        pinned_values(rows * shared * 4, 1.0e-3),
    )
    .unwrap();
    let wide_lhs = wide_lhs_storage
        .view()
        .slice_with::<2>(&[
            SliceArg::range(None, None, 2),
            SliceArg::range(None, None, 2),
        ])
        .unwrap();
    let wide_rhs =
        Array::from_shape_vec([shared, cols], pinned_values(shared * cols, 2.0e-3)).unwrap();
    let geometry = cached_cache_geometry();
    let auto_policy = MatmulTilePolicy::for_geometry(geometry, core::mem::size_of::<f64>(), cols);
    let fixed_policy = MatmulTilePolicy::fixed(32).expect("32-row control is supported");

    // Validate value preservation before timing. The policy changes only row
    // partitioning, so both routes must produce identical output for the same
    // prepared strided inputs.
    let mut auto_output = Array::zeros([rows, cols]);
    let mut fixed_output = Array::zeros([rows, cols]);
    matmul_with_tile_policy(
        &wide_lhs,
        &wide_rhs.view(),
        &mut auto_output.view_mut(),
        auto_policy,
    )
    .unwrap();
    matmul_with_tile_policy(
        &wide_lhs,
        &wide_rhs.view(),
        &mut fixed_output.view_mut(),
        fixed_policy,
    )
    .unwrap();
    assert_eq!(auto_output.view().data(), fixed_output.view().data());

    eprintln!(
        "matmul wide policy: l2={} auto_row_block={} fixed_row_block={}",
        geometry.l2_bytes(),
        auto_policy.row_block(),
        fixed_policy.row_block()
    );
    // Counterbalance the policy comparison at the iteration level. The target
    // order alternates auto→fixed and fixed→auto, while only the target route
    // is timed. Full-output checksums consume both results outside the timed
    // interval, preventing dead-code elimination without charging validation to
    // either policy.
    group.bench_function("wide_policy_auto_64x64x4096", |bencher| {
        bencher.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            let mut control = Array::zeros([rows, cols]);
            let mut target = Array::zeros([rows, cols]);
            for iteration in 0..iterations {
                if iteration % 2 == 0 {
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut control.view_mut(),
                        fixed_policy,
                    )
                    .unwrap();
                    black_box(control.view().data().iter().copied().sum::<f64>());
                    let start = Instant::now();
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut target.view_mut(),
                        auto_policy,
                    )
                    .unwrap();
                    let duration = start.elapsed();
                    black_box(target.view().data().iter().copied().sum::<f64>());
                    elapsed += duration;
                } else {
                    let start = Instant::now();
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut target.view_mut(),
                        auto_policy,
                    )
                    .unwrap();
                    let duration = start.elapsed();
                    black_box(target.view().data().iter().copied().sum::<f64>());
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut control.view_mut(),
                        fixed_policy,
                    )
                    .unwrap();
                    black_box(control.view().data().iter().copied().sum::<f64>());
                    elapsed += duration;
                }
            }
            elapsed
        });
    });
    group.bench_function("wide_policy_fixed32_64x64x4096", |bencher| {
        bencher.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            let mut control = Array::zeros([rows, cols]);
            let mut target = Array::zeros([rows, cols]);
            for iteration in 0..iterations {
                if iteration % 2 == 0 {
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut control.view_mut(),
                        auto_policy,
                    )
                    .unwrap();
                    black_box(control.view().data().iter().copied().sum::<f64>());
                    let start = Instant::now();
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut target.view_mut(),
                        fixed_policy,
                    )
                    .unwrap();
                    let duration = start.elapsed();
                    black_box(target.view().data().iter().copied().sum::<f64>());
                    elapsed += duration;
                } else {
                    let start = Instant::now();
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut target.view_mut(),
                        fixed_policy,
                    )
                    .unwrap();
                    let duration = start.elapsed();
                    black_box(target.view().data().iter().copied().sum::<f64>());
                    matmul_with_tile_policy(
                        black_box(&wide_lhs),
                        black_box(&wide_rhs.view()),
                        &mut control.view_mut(),
                        auto_policy,
                    )
                    .unwrap();
                    black_box(control.view().data().iter().copied().sum::<f64>());
                    elapsed += duration;
                }
            }
            elapsed
        });
    });

    let n = 256usize;
    let strided_lhs_storage =
        Array::from_shape_vec([n * 2, n * 2], pinned_values(n * n * 4, 1.0e-3)).unwrap();
    let strided_lhs = strided_lhs_storage
        .view()
        .slice_with::<2>(&[
            SliceArg::range(None, None, 2),
            SliceArg::range(None, None, 2),
        ])
        .unwrap();
    let rhs = Array::from_shape_vec([n, n], pinned_values(n * n, 2.0e-3)).unwrap();
    group.bench_function("strided_step2_lhs_256x256", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, n]),
            |mut out| {
                matmul(
                    black_box(&strided_lhs),
                    black_box(&rhs.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}
