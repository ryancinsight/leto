//! Statistical reductions over Leto arrays.
//!
//! [`sum_all`], [`mean_all`], [`pearson_correlation`] are computed over
//! borrowd array views, leaving ownership with the caller.

use leto::{mean_all, pearson_correlation, sum_all, Array1, Array2};

fn main() {
    // ── sum_all ──
    let a: Array1<f64> = Array1::from_vec(5, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("shape matches");
    let s = sum_all(&a).expect("sum");
    println!("sum([1..5]) = {s}");
    assert!((s - 15.0).abs() < 1e-10);

    // ── mean_all ──
    let m = mean_all(&a).expect("mean");
    println!("mean([1..5]) = {m}");
    assert!((m - 3.0).abs() < 1e-10);

    // ── pearson_correlation on a 2-column matrix ──
    // Build a 5×2 matrix: first column is x, second is 2x+1.
    let mut rows = Vec::with_capacity(10);
    for i in 0..5 {
        rows.push(i as f64);
        rows.push(2.0 * i as f64 + 1.0);
    }
    let mat: Array2<f64> = Array2::from_vec((5, 2), rows).expect("5×2");
    let corr = pearson_correlation(&mat).expect("correlation matrix");
    // Perfect linear correlation → all off-diagonal entries = 1.0.
    let r01 = corr[[0, 1]];
    println!("pearson(x, 2x+1) = {r01:.6}");
    assert!(
        (r01 - 1.0).abs() < 1e-6,
        "perfectly correlated columns should have r = 1.0, got {r01}"
    );

    // ── Uncorrelated: constant column has zero variance ──
    // (pearson_correlation returns the full N×N matrix; diagonal = 1.0)
    let diag = corr[[0, 0]];
    println!("pearson diagonal = {diag:.6}");
    assert!((diag - 1.0).abs() < 1e-6, "diagonal of correlation matrix must be 1.0");

    println!("all statistics assertions passed");
}
