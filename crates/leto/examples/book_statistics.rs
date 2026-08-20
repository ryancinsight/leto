//! Statistical reductions over Leto arrays.
//!
//! [`sum_all`], [`mean_all`], and [`pearson_correlation`] are computed over
//! borrowed array views, leaving ownership with the caller.

use leto::{Array1, Array2, mean_all, pearson_correlation, sum_all};

fn main() {
    let a: Array1<f64> = Array1::from_vec(5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("shape matches");
    let sum = sum_all(&a).expect("sum");
    println!("sum([1..5]) = {sum}");
    assert!((sum - 15.0).abs() < 1e-10);

    let mean = mean_all(&a).expect("mean");
    println!("mean([1..5]) = {mean}");
    assert!((mean - 3.0).abs() < 1e-10);

    let mut rows = Vec::with_capacity(10);
    for i in 0..5 {
        let x = f64::from(i);
        rows.extend([x, 2.0 * x + 1.0]);
    }
    let matrix: Array2<f64> = Array2::from_vec((5, 2), rows).expect("shape matches");
    let correlation = pearson_correlation(&matrix).expect("correlation matrix");
    let off_diagonal = correlation[[0, 1]];
    println!("pearson(x, 2x+1) = {off_diagonal:.6}");
    assert!((off_diagonal - 1.0).abs() < 1e-6);

    let diagonal = correlation[[0, 0]];
    println!("pearson diagonal = {diagonal:.6}");
    assert!((diagonal - 1.0).abs() < 1e-6);

    println!("all statistics assertions passed");
}
