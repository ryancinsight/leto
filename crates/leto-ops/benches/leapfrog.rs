//! Criterion baselines for the staggered leapfrog stencil sweeps.
//!
//! A kwavers FDTD staggered step runs six of these sweeps (three gradients and
//! three divergences) beside a handful of pointwise updates. Each sweep is
//! timed per axis and order at 64³, the grid of kwavers' `fdtd_step_64_cubed`
//! instrument, with one pointwise velocity update of the same volume as the
//! control, so a sweep's share of a step reads directly. Inputs are pinned and
//! destinations preallocated, so neither construction nor allocation enters a
//! timed closure; each closure returns a computed cell so the sweep cannot be
//! folded away.
//!
//! Check the host is quiet first (see `kernels.rs`): a concurrent build
//! elsewhere in the stack widens these intervals past the effects measured.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use criterion::{criterion_group, criterion_main, Criterion};
use leto::Array;
use leto_ops::{zip_mut_with, Axis, StaggeredLeapfrog3D};
use std::hint::black_box;

/// Cells per axis of kwavers' FDTD step instrument.
const N: usize = 64;
/// Grid spacing of that instrument, in metres.
const SPACING: f64 = 1.0e-4;
/// Time step of the pointwise control, in seconds (the instrument's Courant step).
const DT: f64 = 3.1e-8;

/// Deterministic, non-trivial values: no RNG, so inputs reproduce.
fn pinned_values(seed: f64) -> Vec<f64> {
    (0..N * N * N)
        .map(|i| (i as f64).mul_add(0.731, seed).sin())
        .collect()
}

fn bench_leapfrog_sweeps(c: &mut Criterion) {
    let field = Array::from_shape_vec([N, N, N], pinned_values(1.0)).unwrap();
    let mut dst = Array::from_shape_vec([N, N, N], vec![0.0_f64; N * N * N]).unwrap();
    let centre = [N / 2, N / 2, N / 2];
    let mut group = c.benchmark_group("leapfrog_64_cubed");

    for order in [2_usize, 4] {
        let op = StaggeredLeapfrog3D::<f64>::new(order, SPACING, SPACING, SPACING).unwrap();
        for (label, axis) in [("x", Axis::X), ("y", Axis::Y), ("z", Axis::Z)] {
            group.bench_function(format!("gradient_order{order}_{label}"), |bencher| {
                bencher.iter(|| {
                    op.gradient_into(axis, black_box(field.view()), &mut dst.view_mut())
                        .unwrap();
                    black_box(dst[centre])
                });
            });
            group.bench_function(format!("divergence_order{order}_{label}"), |bencher| {
                bencher.iter(|| {
                    op.divergence_into(axis, black_box(field.view()), &mut dst.view_mut())
                        .unwrap();
                    black_box(dst[centre])
                });
            });
        }
    }

    // The control: one pointwise velocity update over the same volume, the
    // kind of pass a kwavers step runs beside the sweeps.
    let gradient = Array::from_shape_vec([N, N, N], pinned_values(2.0)).unwrap();
    let density = Array::from_shape_vec(
        [N, N, N],
        pinned_values(3.0)
            .into_iter()
            .map(|v| 1_000.0 + v)
            .collect(),
    )
    .unwrap();
    let mut velocity = Array::from_shape_vec([N, N, N], pinned_values(4.0)).unwrap();
    group.bench_function("pointwise_velocity_update", |bencher| {
        bencher.iter(|| {
            zip_mut_with(
                velocity.view_mut(),
                (&black_box(&gradient).view(), &density.view()),
                |v, (&g, &rho)| *v -= DT / rho * g,
            )
            .unwrap();
            black_box(velocity[centre])
        });
    });
    group.finish();
}

criterion_group! {
    name = leapfrog;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
        .without_plots();
    targets = bench_leapfrog_sweeps
}
criterion_main!(leapfrog);
