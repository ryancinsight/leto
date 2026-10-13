mod asymmetry;
mod jacobi;
mod matrix;
mod robustness;

fn column_norm(values: &[f64], n: usize, col: usize) -> f64 {
    (0..n)
        .map(|row| values[row * n + col] * values[row * n + col])
        .sum::<f64>()
        .sqrt()
}

fn column_dot(values: &[f64], n: usize, lhs: usize, rhs: usize) -> f64 {
    (0..n)
        .map(|row| values[row * n + lhs] * values[row * n + rhs])
        .sum()
}

/// Seeded `n × n` symmetric matrix with entries uniform in `[-1, 1)`, plus the
/// same shape rank-deficient: `XᵀX` of a `rank × n` seeded `X`.
fn seeded_pair(n: usize, rank: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = leto_ops::Xorshift64::new(seed);
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = 2.0 * rng.next_unit_f64() - 1.0;
            dense[i * n + j] = value;
            dense[j * n + i] = value;
        }
    }
    let x: Vec<f64> = (0..rank * n).map(|_| rng.next_unit_f64() - 0.5).collect();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = (0..rank).map(|r| x[r * n + i] * x[r * n + j]).sum();
        }
    }
    (dense, gram)
}
