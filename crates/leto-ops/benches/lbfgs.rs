//! Criterion baseline for the L-BFGS two-loop direction under the
//! flat-ring storage (ATLAS-ARCH-008 sub-slice).
//!
//! The conversion replaced `Vec<Vec<f64>>` history plus `Vec::remove(0)`
//! eviction with two CSR-shaped flat ring buffers (`s_buf`, `y_buf` of
//! capacity `memory * n`) plus a scalar `rho_buf` ring, addressed by a
//! single `head` index mod `memory`. The acceptance oracle for the slice
//! is a baseline comparison: flat-ring `LbfgsMemory::direction` against
//! an inline jagged `Vec<Vec<f64>>` reference whose two-loop recursion is
//! semantically identical but uses the pre-conversion storage shape, over
//! the four configurations `{8, 32} memory × {100, 1000} dim`. The
//! production `LbfgsMemory` is what downstream callers (kwavers FWI
//! inverse, etc.) exercise; the inline reference is a direct jagged
//! baseline — no production caller reaches it.
//!
//! Per the stack's performance-engineering rule, an optimization-labeled
//! change requires a stored criterion baseline comparison. Report median
//! and confidence interval from Criterion's standard output; read the CI
//! (a busy host widens the intervals past any realistic kernel step).
//!
//! Run as `cargo bench -p leto-ops --bench lbfgs`. Benchmark time budget:
//! full Criterion sampling exceeds the run-output budget, so smoke runs
//! use `cargo bench -p leto-ops --bench lbfgs -- --test` (single-sample
//! quick-pass); full timing runs use the committed bounded runner
//! (≤300s/binary wall-clock per `engineering_gates`).

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use leto_ops::LbfgsMemory;
use std::hint::black_box;

/// Inner dot product `a·b` for `f64` slices; matches the kernel under test.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Deterministic, non-trivial correction pair for slot `i` at `dim` n.
/// Values are small positives with `sᵀy > 1e-12` so the curvature gate
/// always accepts the pair and the ring fills.
fn pair_at(i: usize, dim: usize) -> (Vec<f64>, Vec<f64>) {
    let s: Vec<f64> = (0..dim)
        .map(|j| ((i * 7 + j) as f64).sin().abs() * 1e-3 + 1e-6)
        .collect();
    let y: Vec<f64> = (0..dim)
        .map(|j| ((i * 11 + j) as f64).cos().abs() * 1e-3 + 1e-6)
        .collect();
    (s, y)
}

/// A pinned gradient of length `dim`, also deterministic and nonzero.
fn g_at(dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|j| ((j as f64) * 0.123).sin() * 1e-2)
        .collect()
}

/// Inline pre-conversion storage shape: trapezoidal `Vec<Vec<f64>>` for `s`
/// and `y`, plus a flat `Vec<f64>` for `rho`, evicted via `Vec::remove(0)`.
/// The two-loop recursion is mirrored 1:1 from the original implementation
/// so the bench isolates the storage-layout delta, not algorithmic drift.
struct JaggedLbfgs {
    memory: usize,
    s_hist: Vec<Vec<f64>>,
    y_hist: Vec<Vec<f64>>,
    rho_hist: Vec<f64>,
}

impl JaggedLbfgs {
    fn new(memory: usize) -> Self {
        Self {
            memory: memory.max(1),
            s_hist: Vec::with_capacity(memory.max(1)),
            y_hist: Vec::with_capacity(memory.max(1)),
            rho_hist: Vec::with_capacity(memory.max(1)),
        }
    }

    fn push(&mut self, s: Vec<f64>, y: Vec<f64>) -> bool {
        let sy = dot(&s, &y);
        if sy <= 1e-12 {
            return false;
        }
        if self.s_hist.len() == self.memory {
            self.s_hist.remove(0);
            self.y_hist.remove(0);
            self.rho_hist.remove(0);
        }
        self.rho_hist.push(1.0 / sy);
        self.s_hist.push(s);
        self.y_hist.push(y);
        true
    }

    /// Two-loop recursion, mirroring the pre-conversion production `direction`.
    #[allow(clippy::unwrap_used)]
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let k = self.s_hist.len();
        if k == 0 {
            return g.iter().map(|&gi| -gi).collect();
        }
        let mut q = g.to_vec();
        let mut alpha = vec![0.0_f64; k];
        for i in (0..k).rev() {
            let a = self.rho_hist[i] * dot(&self.s_hist[i], &q);
            alpha[i] = a;
            q.iter_mut()
                .zip(&self.y_hist[i])
                .for_each(|(qj, &yj)| *qj -= a * yj);
        }
        let last = k - 1;
        let sy = dot(&self.s_hist[last], &self.y_hist[last]);
        let yy = dot(&self.y_hist[last], &self.y_hist[last]);
        let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();
        for (((s_i, y_i), &rho_i), &alpha_i) in self
            .s_hist
            .iter()
            .zip(&self.y_hist)
            .zip(&self.rho_hist)
            .zip(&alpha)
        {
            let beta = rho_i * dot(y_i, &r);
            let coef = alpha_i - beta;
            r.iter_mut().zip(s_i).for_each(|(rj, &sj)| *rj += coef * sj);
        }
        r.iter().map(|&ri| -ri).collect()
    }
}

fn bench_lbfgs_direction(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgs_direction");
    // Acceptance oracle: {8, 32} memory × {100, 1000} dim.
    for &(memory, dim) in &[(8usize, 100usize), (32, 100), (8, 1000), (32, 1000)] {
        // Pre-build the deterministic pair sequence and gradient outside the
        // timed closure so per-iteration setup stays out of the measurement.
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = (0..(memory * 4)).map(|i| pair_at(i, dim)).collect();
        let g = g_at(dim);
        let label = format!("m{memory}_n{dim}");
        // Flat-ring production path.
        group.bench_function(format!("ring/{label}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut mem = LbfgsMemory::new(memory);
                    for (s, y) in &pairs {
                        let _ = mem.push(s.clone(), y.clone());
                    }
                    mem
                },
                |mem| {
                    let d = mem.direction(black_box(&g));
                    black_box(d);
                },
                BatchSize::SmallInput,
            );
        });
        // Jagged pre-conversion baseline.
        group.bench_function(format!("jagged/{label}"), |bencher| {
            bencher.iter_batched(
                || {
                    let mut mem = JaggedLbfgs::new(memory);
                    for (s, y) in &pairs {
                        let _ = mem.push(s.clone(), y.clone());
                    }
                    mem
                },
                |mem| {
                    let d = mem.direction(black_box(&g));
                    black_box(d);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lbfgs_direction);
criterion_main!(benches);
