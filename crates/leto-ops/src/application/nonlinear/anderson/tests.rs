#![allow(clippy::unwrap_used, reason = "test scope")]

use super::{AndersonAccelerator, AndersonConfig, AndersonMethod};
use crate::application::nonlinear::linalg::{add_scaled, vector_from_vec, vector_len};
use eunomia::assert_relative_eq;
use leto::{Array1, Array2};

fn vec(values: Vec<f64>) -> Array1<f64> {
    vector_from_vec(values)
}

fn mat_vec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    let [rows, cols] = matrix.shape();
    assert_eq!(cols, vector_len(vector));
    vector_from_vec(
        (0..rows)
            .map(|row| (0..cols).fold(0.0, |acc, col| acc + matrix[[row, col]] * vector[[col]]))
            .collect(),
    )
}

fn make_linear_system() -> (Array2<f64>, Array1<f64>) {
    // G(x) = A x + b, fixed point at x* = [1.0, 2.0]
    let a = Array2::from_shape_vec([2, 2], vec![0.5, 0.1, 0.0, 0.5]).unwrap();
    let b = vec(vec![1.0 - (0.5 * 1.0 + 0.1 * 2.0), 2.0 - 0.5 * 2.0]);
    (a, b)
}

fn run_fixed_point(config: AndersonConfig<f64>, iters: usize) -> Array1<f64> {
    let (a, b) = make_linear_system();
    let mut accelerator = AndersonAccelerator::new(config);
    let mut x = vec(vec![0.0, 0.0]);
    for _ in 0..iters {
        let g_x = add_scaled(&mat_vec(&a, &x), &b, 1.0);
        x = accelerator.compute_next(&x, &g_x);
    }
    x
}

/// Verify QR variant converges to x* = [1.0, 2.0] within 1e-8.
#[test]
fn test_anderson_qr_convergence() {
    let config = AndersonConfig::<f64> {
        history_depth: 3,
        relaxation: 1.0,
        drop_tolerance: 1e-12,
        method: AndersonMethod::QR,
    };
    let x = run_fixed_point(config, 30);
    assert_relative_eq!(x[[0]], 1.0, epsilon = 1e-8);
    assert_relative_eq!(x[[1]], 2.0, epsilon = 1e-8);
}

/// QR and NormalEquations variants must converge to the same fixed point.
#[test]
fn test_anderson_qr_equivalence_to_normal_equations() {
    let config_qr = AndersonConfig::<f64> {
        history_depth: 3,
        relaxation: 1.0,
        drop_tolerance: 1e-12,
        method: AndersonMethod::QR,
    };
    let config_ne = AndersonConfig::<f64> {
        method: AndersonMethod::NormalEquations,
        ..config_qr.clone()
    };
    let x_qr = run_fixed_point(config_qr, 30);
    let x_ne = run_fixed_point(config_ne, 30);
    assert_relative_eq!(x_qr[[0]], x_ne[[0]], epsilon = 1e-7);
    assert_relative_eq!(x_qr[[1]], x_ne[[1]], epsilon = 1e-7);
}

/// VecDeque history must not exceed m entries after m+5 steps (GAP-PERF-004).
#[test]
fn test_vecdeque_history_bounded() {
    let m = 3usize;
    let config = AndersonConfig::<f64> {
        history_depth: m,
        relaxation: 1.0,
        drop_tolerance: 1e-12,
        method: AndersonMethod::QR,
    };
    let (a, b) = make_linear_system();
    let mut acc = AndersonAccelerator::new(config);
    let mut x = vec(vec![0.0, 0.0]);
    for _ in 0..(m + 5) {
        let g_x = add_scaled(&mat_vec(&a, &x), &b, 1.0);
        x = acc.compute_next(&x, &g_x);
        assert!(
            acc.delta_x.len() <= m,
            "delta_x history len {} exceeds m={}",
            acc.delta_x.len(),
            m
        );
        assert!(
            acc.delta_f.len() <= m,
            "delta_f history len {} exceeds m={}",
            acc.delta_f.len(),
            m
        );
    }
}

/// Original NormalEquations convergence test preserved (regression guard).
#[test]
fn test_anderson_acceleration_linear_convergence() {
    let config = AndersonConfig::<f64> {
        history_depth: 3,
        relaxation: 1.0,
        drop_tolerance: 1e-12,
        method: AndersonMethod::NormalEquations,
    };
    let x = run_fixed_point(config, 20);
    assert!((x[[0]] - 1.0).abs() < 1e-6);
    assert!((x[[1]] - 2.0).abs() < 1e-6);
}

/// QR should handle near-collinear history without divergence (ill-conditioning test).
/// When ΔF columns are nearly identical, the normal equations approach is prone to
/// numerical rank deficiency — QR should silently discard bad columns.
#[test]
fn test_anderson_qr_ill_conditioned_history_graceful() {
    // Very slowly-converging fixed point: contraction ratio 0.99 → lots of similar steps
    let a = Array2::from_shape_vec([2, 2], vec![0.99, 0.0, 0.0, 0.99]).unwrap();
    let b = vec(vec![0.01, 0.02]); // x* = [1.0, 2.0]

    let config = AndersonConfig::<f64> {
        history_depth: 5,
        relaxation: 1.0,
        drop_tolerance: 1e-12,
        method: AndersonMethod::QR,
    };
    let mut acc = AndersonAccelerator::new(config);
    let mut x = vec(vec![0.0, 0.0]);

    // Run until convergence or 200 iterations — must not panic or diverge
    for _ in 0..200 {
        let g_x = add_scaled(&mat_vec(&a, &x), &b, 1.0);
        x = acc.compute_next(&x, &g_x);
        if (x[[0]] - 1.0).abs() < 1e-6 && (x[[1]] - 2.0).abs() < 1e-6 {
            break;
        }
    }
    assert!(
        (x[[0]] - 1.0).abs() < 1e-4,
        "QR Anderson must converge on ill-conditioned problem: x[0]={:.6}",
        x[[0]]
    );
    assert!(
        (x[[1]] - 2.0).abs() < 1e-4,
        "QR Anderson must converge on ill-conditioned problem: x[1]={:.6}",
        x[[1]]
    );
}

/// Lockstep invariant (OPEN-033 mitigation): when the QR path silently
/// rejects a near-collinear column, the parallel `delta_x`/`delta_f`
/// deques must stay aligned with `qr.q_cols` so subsequent
/// `compute_next` calls index `(γᵢ, Δxᵢ, Δfᵢ)` triples that refer to
/// the **same** history entry. A regression here would re-introduce the
/// desync that the gap_audit flagged.
#[test]
fn test_anderson_qr_lockstep_invariant_under_rejection() {
    // Fixed point map that produces ΔF vectors that are highly
    // collinear after iter 0, forcing QR to reject most columns.
    let a = Array2::from_shape_vec([2, 2], vec![0.99, 0.0, 0.0, 0.99]).unwrap();
    // Tiny RHS so the iterates shrink slowly and ΔF stays near-linear.
    let b = vec(vec![1e-12, 2e-12]);

    let config = AndersonConfig::<f64> {
        history_depth: 5,
        relaxation: 1.0,
        // A loose drop tolerance forces *every* new column to be
        // rejected after the first, so we exercise the eviction path
        // many times in a row.
        drop_tolerance: 1e-2,
        method: AndersonMethod::QR,
    };
    let mut acc = AndersonAccelerator::new(config);
    let mut x = vec(vec![1.0, 2.0]);

    for iter in 0..200 {
        let g_x = add_scaled(&mat_vec(&a, &x), &b, 1.0);
        x = acc.compute_next(&x, &g_x);

        // Invariant: every step, the parallel deque lengths must match
        // the QR column count exactly — or be the empty initial state
        // (before the first iteration has produced (Δx, Δf)).
        let qr_cols = acc.qr_state.as_ref().map_or(0, |qr| qr.q_cols.len());
        assert_eq!(
            acc.delta_x.len(),
            qr_cols,
            "iter {}: delta_x.len()={} should equal qr.q_cols.len()={}",
            iter,
            acc.delta_x.len(),
            qr_cols,
        );
        assert_eq!(
            acc.delta_f.len(),
            qr_cols,
            "iter {}: delta_f.len()={} should equal qr.q_cols.len()={}",
            iter,
            acc.delta_f.len(),
            qr_cols,
        );
    }
}
