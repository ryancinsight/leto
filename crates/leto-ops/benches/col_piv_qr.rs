#![expect(clippy::unwrap_used, reason = "fixed benchmark fixture shapes")]

use criterion::{criterion_group, criterion_main, Criterion};
use leto::{Array, Array2};
use leto_ops::col_piv_qr;
use std::hint::black_box;
use std::time::Duration;

fn matrix(order: usize) -> Array2<f64> {
    let values = (0..order)
        .flat_map(|row| {
            (0..order).map(move |column| {
                let residue = u32::try_from((row * 131 + column * 197 + row * column * 17) % 1_021)
                    .expect("invariant: residue is below 1,021");
                f64::from(residue) / 512.0 - 1.0 + if row == column { 4.0 } else { 0.0 }
            })
        })
        .collect();
    Array::from_shape_vec([order, order], values)
        .expect("invariant: fixture count matches its square shape")
}

fn bench_col_piv_qr_scaling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("col_piv_qr_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for order in [64usize, 128, 256] {
        let input = matrix(order);
        group.bench_function(format!("factor_{order}x{order}"), |bencher| {
            bencher.iter(|| col_piv_qr(black_box(&input.view())).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_col_piv_qr_scaling);
criterion_main!(benches);
