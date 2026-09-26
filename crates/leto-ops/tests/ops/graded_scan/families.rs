//! The six scanned matrix families and their placement at a binary exponent.

use super::super::format::{scale, Format};
use leto::Array2;
use leto_ops::Xorshift64;

pub(super) fn unit(rng: &mut Xorshift64) -> f64 {
    2.0 * rng.next_unit_f64() - 1.0
}

/// A random orthogonal `n × n` (Gram–Schmidt, twice, on uniform columns).
pub(super) fn orthogonal(n: usize, rng: &mut Xorshift64) -> Vec<f64> {
    let mut q = vec![0.0; n * n];
    for j in 0..n {
        let mut v: Vec<f64> = (0..n).map(|_| unit(rng)).collect();
        for _ in 0..2 {
            for k in 0..j {
                let d: f64 = (0..n).map(|i| v[i] * q[i * n + k]).sum();
                for i in 0..n {
                    v[i] -= d * q[i * n + k];
                }
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..n {
            q[i * n + j] = v[i] / norm;
        }
    }
    q
}

/// `Q·diag(d)·Qᵀ`, exactly symmetric.
pub(super) fn symmetric_from(q: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v: f64 = (0..n).map(|k| q[i * n + k] * d[k] * q[j * n + k]).sum();
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }
    s
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Family {
    Graded,
    Clustered,
    RankDeficient,
    SymmetricRankDeficient,
    SkewTridiagonal,
    SkewDense,
}

pub(super) const FAMILIES: [Family; 6] = [
    Family::Graded,
    Family::Clustered,
    Family::RankDeficient,
    Family::SymmetricRankDeficient,
    Family::SkewTridiagonal,
    Family::SkewDense,
];

impl Family {
    /// Normal (`AAᵀ = AᵀA`): an orthogonal eigenbasis, `κ = 1`.
    pub(super) fn normal(self) -> bool {
        !matches!(self, Self::Graded | Self::RankDeficient)
    }

    /// A seeded matrix of order `n` at unit scale.
    pub(super) fn matrix(self, n: usize, rng: &mut Xorshift64) -> Vec<f64> {
        match self {
            Self::Graded => (0..n * n)
                .map(|k| unit(rng) * scale(1.0, -2 * i32::try_from(k / n).unwrap()))
                .collect(),
            Self::Clustered => {
                let q = orthogonal(n, rng);
                let d: Vec<f64> = (0..n)
                    .map(|i| {
                        if i + 1 == n {
                            0.5
                        } else {
                            1.0 + (i / 2) as f64 * 1e-3
                        }
                    })
                    .collect();
                symmetric_from(&q, &d, n)
            }
            Self::RankDeficient => {
                let r = (n / 2).max(1);
                let x: Vec<f64> = (0..n * r).map(|_| unit(rng)).collect();
                let y: Vec<f64> = (0..r * n).map(|_| unit(rng)).collect();
                (0..n * n)
                    .map(|k| (0..r).map(|t| x[(k / n) * r + t] * y[t * n + k % n]).sum())
                    .collect()
            }
            Self::SymmetricRankDeficient => {
                let q = orthogonal(n, rng);
                let r = (n / 2).max(1);
                let d: Vec<f64> = (0..n)
                    .map(|i| if i < r { 0.5 + unit(rng).abs() } else { 0.0 })
                    .collect();
                symmetric_from(&q, &d, n)
            }
            Self::SkewTridiagonal => {
                let mut m = vec![0.0; n * n];
                for i in 0..n - 1 {
                    let sign = if unit(rng) < 0.0 { -1.0 } else { 1.0 };
                    let magnitude = 10.0_f64.powf(-4.0 * rng.next_unit_f64());
                    m[i * n + i + 1] = sign * magnitude;
                    m[(i + 1) * n + i] = -sign * magnitude;
                }
                m
            }
            Self::SkewDense => {
                let mut m = vec![0.0; n * n];
                for i in 0..n {
                    for j in i + 1..n {
                        let v = unit(rng);
                        m[i * n + j] = v;
                        m[j * n + i] = -v;
                    }
                }
                m
            }
        }
    }
}

/// The `T` matrix `2^e·unit` and the exact image of what `T` holds, in units
/// of `2^e`.
pub(super) fn place<T: Format>(
    unit_values: &[f64],
    n: usize,
    exponent: i32,
) -> (Array2<T>, Vec<f64>) {
    let s = T::ONE.scale_binary(exponent);
    let values: Vec<T> = unit_values.iter().map(|&v| T::from_f64(v).mul(s)).collect();
    let image = values
        .iter()
        .map(|v| scale(v.to_f64(), -exponent))
        .collect();
    (Array2::from_shape_vec([n, n], values).unwrap(), image)
}
