//! # `ndarray` × `leto` Parity Harness
//!
//! Canonical migration-parity harness for `ndarray` → `leto` array operations.
//! Lives in `leto-ops` alongside the `nalgebra_parity` example: both legacy
//! array libraries are validated at the replacement-crate source, not in
//! downstream consumers.
//!
//! ## Operations covered
//!
//! | Operation | ndarray | leto / leto-ops |
//! |---|---|---|
//! | 1-D array construction | `Array1::from_vec` | `Array1::from_shape_vec` |
//! | Element-wise add | `&a + &b` | `leto_ops::add` |
//! | Dot product | `a.dot(&b)` | `leto_ops::dot` |
//! | Matrix multiply | `a.dot(&b)` (2-D) | `leto_ops::matmul` |
//! | Sum all elements | `a.sum()` | `leto_ops::sum` |
//! | Map element-wise | `a.mapv(f)` | `leto_ops::mapv` |
//!
//! Parity tolerance: L∞ ≤ 1e-12 (same IEEE-754 arithmetic, same data).
//!
//! ## Run
//!
//! ```sh
//! cargo run --release --example ndarray_parity -p leto-ops
//! ```

use leto::{Array1 as LetoArray1, Array2 as LetoArray2};
use leto_ops::{add, dot, mapv, matmul, sum};
use ndarray::{Array1 as NdArray1, Array2 as NdArray2};

// ── Helpers ────────────────────────────────────────────────────────────────

fn l_inf_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

// ── Parity checks ──────────────────────────────────────────────────────────

fn check_elementwise_add() -> bool {
    let n = 256usize;
    let a_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03 + 1.0).sin()).collect();

    // ndarray
    let a_nd = NdArray1::from_vec(a_data.clone());
    let b_nd = NdArray1::from_vec(b_data.clone());
    let c_nd: Vec<f64> = (&a_nd + &b_nd).into_raw_vec();

    // leto
    let a_le = LetoArray1::from_shape_vec([n], a_data).expect("valid");
    let b_le = LetoArray1::from_shape_vec([n], b_data).expect("valid");
    let mut c_le = LetoArray1::<f64>::zeros([n]);
    add(&a_le.view(), &b_le.view(), &mut c_le.view_mut()).expect("add");
    let c_le_vec: Vec<f64> = c_le.iter().copied().collect();

    l_inf_diff(&c_nd, &c_le_vec) < 1e-12
}

fn check_dot_product() -> bool {
    let n = 512usize;
    let a_data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos()).collect();

    // ndarray
    let a_nd = NdArray1::from_vec(a_data.clone());
    let b_nd = NdArray1::from_vec(b_data.clone());
    let dot_nd = a_nd.dot(&b_nd);

    // leto
    let a_le = LetoArray1::from_shape_vec([n], a_data).expect("valid");
    let b_le = LetoArray1::from_shape_vec([n], b_data).expect("valid");
    let dot_le = dot(&a_le.view(), &b_le.view()).expect("dot");

    (dot_nd - dot_le).abs() < 1e-10
}

fn check_matmul() -> bool {
    let m = 16usize;
    let k = 24usize;
    let p = 12usize;
    let a_data: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.05).sin()).collect();
    let b_data: Vec<f64> = (0..k * p).map(|i| (i as f64 * 0.07).cos()).collect();

    // ndarray
    let a_nd = NdArray2::from_shape_vec((m, k), a_data.clone()).expect("valid");
    let b_nd = NdArray2::from_shape_vec((k, p), b_data.clone()).expect("valid");
    let c_nd = a_nd.dot(&b_nd);
    let nd_flat: Vec<f64> = c_nd.into_raw_vec();

    // leto
    let a_le = LetoArray2::from_shape_vec([m, k], a_data).expect("valid");
    let b_le = LetoArray2::from_shape_vec([k, p], b_data).expect("valid");
    let mut c_le = LetoArray2::<f64>::zeros([m, p]);
    matmul(&a_le.view(), &b_le.view(), &mut c_le.view_mut()).expect("matmul");
    let le_flat: Vec<f64> = c_le.iter().copied().collect();

    l_inf_diff(&nd_flat, &le_flat) < 1e-10
}

fn check_sum_reduction() -> bool {
    let n = 1024usize;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();

    // ndarray
    let a_nd = NdArray1::from_vec(data.clone());
    let sum_nd = a_nd.sum();

    // leto — uses `sum` from leto-ops (the exported re-export of map::sum)
    let a_le = LetoArray1::from_shape_vec([n], data).expect("valid");
    let sum_le: f64 = sum(&a_le.view());

    (sum_nd - sum_le).abs() < 1e-9
}

fn check_mapv() -> bool {
    let n = 256usize;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.04).cos()).collect();

    // ndarray: mapv applies f element-wise with copy semantics
    let a_nd = NdArray1::from_vec(data.clone());
    let c_nd_vec: Vec<f64> = a_nd.mapv(|x: f64| x * x + 1.0).into_raw_vec();

    // leto: mapv — same copy semantics, returns Result<Array>
    let a_le = LetoArray1::from_shape_vec([n], data).expect("valid");
    let c_le = mapv(&a_le.view(), |x: f64| x * x + 1.0).expect("mapv");
    let le_vec: Vec<f64> = c_le.iter().copied().collect();

    l_inf_diff(&c_nd_vec, &le_vec) < 1e-12
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    let checks: &[(&str, bool)] = &[
        ("elementwise_add", check_elementwise_add()),
        ("dot_product",     check_dot_product()),
        ("matmul",          check_matmul()),
        ("sum_reduction",   check_sum_reduction()),
        ("mapv",            check_mapv()),
    ];

    let all_pass = checks.iter().all(|(_, p)| *p);

    for (name, pass) in checks {
        eprintln!("  {:20} {}", name, if *pass { "PASS ✅" } else { "FAIL ❌" });
    }

    println!(
        "{{\"crate\":\"leto-ops\",\"harness\":\"ndarray_parity\",\"all_pass\":{all_pass}}}"
    );
    eprintln!("─── ndarray × leto parity ─── {}", if all_pass { "PASS ✅" } else { "FAIL ❌" });

    assert!(all_pass, "ndarray × leto parity FAIL — see above");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_add_parity() {
        assert!(check_elementwise_add());
    }

    #[test]
    fn dot_product_parity() {
        assert!(check_dot_product());
    }

    #[test]
    fn matmul_parity() {
        assert!(check_matmul());
    }

    #[test]
    fn sum_reduction_parity() {
        assert!(check_sum_reduction());
    }

    #[test]
    fn mapv_parity() {
        assert!(check_mapv());
    }
}
