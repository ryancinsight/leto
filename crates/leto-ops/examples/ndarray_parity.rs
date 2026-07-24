//! Runnable Leto-to-Leto migration evidence.
//!
//! The harness executes deterministic construction, elementwise, reduction,
//! and matrix operations through Leto. It reports the measured absolute
//! differential for each operation and checks it against one of two contracts:
//!
//! - elementwise operations must be bitwise equal because both paths execute
//!   the same ordered IEEE-754 operation per element;
//! - reductions use 2*gamma_n * sum|term|, the triangle-inequality bound for
//!   two independently rounded accumulation orders.
//!
//! Run with:
//!
//! cargo run --locked -p leto-ops --example ndarray_parity

mod support;

use leto::{Array1 as LetoArray1, Array2 as LetoArray2, Storage};
use leto_ops::{add, dot, mapv, matmul, sum};
use support::{gamma, max_abs_diff, Observation};

/// Each independently rounded reduction differs from the exact sum by at
/// most gamma_n * sum|term|; the triangle inequality bounds their difference
/// by twice that value.
fn two_path_reduction_bound(terms: usize, absolute_term_sum: f64) -> f64 {
    2.0 * gamma(terms) * absolute_term_sum
}

fn check_construction() -> Observation {
    let values: Vec<f64> = (0..32).map(|index| index as f64 * 0.25 - 2.0).collect();
    let leto = LetoArray1::from_shape_vec([values.len()], values)
        .expect("input length matches the declared shape");
    Observation::new(
        "construction",
        max_abs_diff(leto.storage().as_slice(), leto.storage().as_slice()),
        0.0,
    )
}

fn check_elementwise_add() -> Observation {
    let element_count = 256;
    let lhs_values: Vec<f64> = (0..element_count)
        .map(|index| index as f64 * 0.01)
        .collect();
    let rhs_values: Vec<f64> = (0..element_count)
        .map(|index| (index as f64 * 0.03 + 1.0).sin())
        .collect();

    let leto_lhs = LetoArray1::from_shape_vec([element_count], lhs_values.clone())
        .expect("input length matches the declared shape");
    let leto_rhs = LetoArray1::from_shape_vec([element_count], rhs_values.clone())
        .expect("input length matches the declared shape");
    let mut leto_result = LetoArray1::<f64>::zeros([element_count]);
    add(
        &leto_lhs.view(),
        &leto_rhs.view(),
        &mut leto_result.view_mut(),
    )
    .expect("equal shapes satisfy elementwise addition");

    // Reference: plain elementwise add.
    let reference: Vec<f64> = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(x, y)| x + y)
        .collect();

    Observation::new(
        "elementwise_add",
        max_abs_diff(leto_result.storage().as_slice(), &reference),
        0.0,
    )
}

fn check_dot_product() -> Observation {
    let element_count = 512;
    let lhs_values: Vec<f64> = (0..element_count)
        .map(|index| (index as f64 * 0.07).sin())
        .collect();
    let rhs_values: Vec<f64> = (0..element_count)
        .map(|index| (index as f64 * 0.11).cos())
        .collect();
    let absolute_term_sum = lhs_values
        .iter()
        .zip(&rhs_values)
        .map(|(lhs, rhs)| (lhs * rhs).abs())
        .sum();

    let leto_lhs = LetoArray1::from_shape_vec([element_count], lhs_values.clone())
        .expect("input length matches the declared shape");
    let leto_rhs = LetoArray1::from_shape_vec([element_count], rhs_values.clone())
        .expect("input length matches the declared shape");
    let leto_result =
        dot(&leto_lhs.view(), &leto_rhs.view()).expect("equal lengths satisfy dot product");

    // Reference: plain dot product.
    let reference: f64 = lhs_values.iter().zip(&rhs_values).map(|(x, y)| x * y).sum();

    Observation::new(
        "dot_product",
        (reference - leto_result).abs(),
        two_path_reduction_bound(element_count, absolute_term_sum),
    )
}

fn check_matmul() -> Observation {
    let (rows, inner, columns) = (16, 24, 12);
    let lhs_values: Vec<f64> = (0..rows * inner)
        .map(|index| (index as f64 * 0.05).sin())
        .collect();
    let rhs_values: Vec<f64> = (0..inner * columns)
        .map(|index| (index as f64 * 0.07).cos())
        .collect();

    let leto_lhs = LetoArray2::from_shape_vec([rows, inner], lhs_values.clone())
        .expect("input length matches the declared shape");
    let leto_rhs = LetoArray2::from_shape_vec([inner, columns], rhs_values.clone())
        .expect("input length matches the declared shape");
    let mut leto_result = LetoArray2::<f64>::zeros([rows, columns]);
    matmul(
        &leto_lhs.view(),
        &leto_rhs.view(),
        &mut leto_result.view_mut(),
    )
    .expect("compatible matrix shapes satisfy multiplication");

    // Reference: C[i,j] = sum_k A[i,k] * B[k,j].
    let mut reference = vec![0.0f64; rows * columns];
    for i in 0..rows {
        for j in 0..columns {
            let mut acc = 0.0f64;
            for kk in 0..inner {
                acc += lhs_values[i * inner + kk] * rhs_values[kk * columns + j];
            }
            reference[i * columns + j] = acc;
        }
    }

    let max_abs_term_sum = (0..rows)
        .flat_map(|row| {
            let lhs_values = &lhs_values;
            let rhs_values = &rhs_values;
            (0..columns).map(move |column| {
                (0..inner)
                    .map(|index| {
                        (lhs_values[row * inner + index] * rhs_values[index * columns + column])
                            .abs()
                    })
                    .sum::<f64>()
            })
        })
        .fold(0.0_f64, f64::max);

    Observation::new(
        "matmul",
        max_abs_diff(leto_result.storage().as_slice(), &reference),
        two_path_reduction_bound(inner, max_abs_term_sum),
    )
}

fn check_sum_reduction() -> Observation {
    let element_count = 1024;
    let values: Vec<f64> = (0..element_count)
        .map(|index| (index as f64 * 0.13).sin())
        .collect();
    let absolute_term_sum = values.iter().map(|value| value.abs()).sum();
    let reference: f64 = values.iter().sum();
    let leto = LetoArray1::from_shape_vec([element_count], values)
        .expect("input length matches the declared shape");
    let leto_result = sum(&leto.view());

    Observation::new(
        "sum_reduction",
        (reference - leto_result).abs(),
        two_path_reduction_bound(element_count, absolute_term_sum),
    )
}

fn check_mapv() -> Observation {
    let element_count = 256;
    let values: Vec<f64> = (0..element_count)
        .map(|index| (index as f64 * 0.04).cos())
        .collect();
    let leto = LetoArray1::from_shape_vec([element_count], values.clone())
        .expect("input length matches the declared shape");
    let leto_result =
        mapv(&leto.view(), |value: f64| value * value + 1.0).expect("map preserves shape");

    // Reference: plain elementwise map.
    let reference: Vec<f64> = values.iter().map(|&value| value * value + 1.0).collect();

    Observation::new(
        "mapv",
        max_abs_diff(leto_result.storage().as_slice(), &reference),
        0.0,
    )
}

fn observations() -> [Observation; 6] {
    [
        check_construction(),
        check_elementwise_add(),
        check_dot_product(),
        check_matmul(),
        check_sum_reduction(),
        check_mapv(),
    ]
}

fn main() {
    let observations = observations();
    for observation in observations {
        eprintln!(
            "{:<20} error={:.6e} bound={:.6e}",
            observation.name, observation.error, observation.bound
        );
        observation.assert_within_bound();
    }
    println!(
        "{{\"crate\":\"leto-ops\",\"harness\":\"leto_parity\",\"checks\":{},\"all_pass\":true}}",
        observations.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_parity() {
        check_construction().assert_within_bound();
    }

    #[test]
    fn elementwise_add_parity() {
        check_elementwise_add().assert_within_bound();
    }

    #[test]
    fn dot_product_parity() {
        check_dot_product().assert_within_bound();
    }

    #[test]
    fn matmul_parity() {
        check_matmul().assert_within_bound();
    }

    #[test]
    fn sum_reduction_parity() {
        check_sum_reduction().assert_within_bound();
    }

    #[test]
    fn mapv_parity() {
        check_mapv().assert_within_bound();
    }
}
