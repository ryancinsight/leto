//! LU unit tests: blocked-vs-unblocked agreement and the spill gate.

use super::decompose::lu_uses_blocked_path;
use super::eliminate::{lu_eliminate_blocked, lu_eliminate_step};
use crate::infrastructure::cache::CacheGeometry;

fn diagonally_dominant(n: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = ((i * 7 + j * 13) % 17) as f64 * 0.01 - 0.08;
        }
        data[i * n + i] += n as f64;
    }
    data
}

fn eliminate_unblocked(a: &mut [f64], n: usize, pivots: &mut [usize], sign: &mut i8) {
    for k in 0..n {
        lu_eliminate_step(a, n, k, n, pivots, sign).expect("fixture is nonsingular");
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn blocked_path_matches_unblocked_within_rounding() {
    // n = 100: a full 64-wide panel, a partial panel, and two GEMM
    // trailing updates. Same pivot sequence, factors within ~n·ε.
    let n = 100usize;
    let data = diagonally_dominant(n);
    let mut blocked = data.clone();
    let mut blocked_pivots: Vec<usize> = (0..n).collect();
    let mut blocked_sign = 1i8;
    lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
        .expect("fixture is nonsingular");
    let mut plain = data.clone();
    let mut plain_pivots: Vec<usize> = (0..n).collect();
    let mut plain_sign = 1i8;
    eliminate_unblocked(&mut plain, n, &mut plain_pivots, &mut plain_sign);
    assert_eq!(blocked_pivots, plain_pivots);
    assert_eq!(blocked_sign, plain_sign);
    let err = max_abs_diff(&blocked, &plain);
    assert!(err < 1e-9, "blocked-vs-unblocked max factor diff {err}");
}

#[test]
fn blocked_path_matches_unblocked_for_reduced_width_scalar() {
    // Same n = 100 shape in f32: ~n·ε₃₂ rounding over O(n) entries.
    let n = 100usize;
    let data: Vec<f32> = diagonally_dominant(n).iter().map(|&v| v as f32).collect();
    let mut blocked = data.clone();
    let mut blocked_pivots: Vec<usize> = (0..n).collect();
    let mut blocked_sign = 1i8;
    lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
        .expect("fixture is nonsingular");
    let mut plain = data.clone();
    let mut plain_pivots: Vec<usize> = (0..n).collect();
    let mut plain_sign = 1i8;
    for k in 0..n {
        lu_eliminate_step(&mut plain, n, k, n, &mut plain_pivots, &mut plain_sign)
            .expect("fixture is nonsingular");
    }
    assert_eq!(blocked_pivots, plain_pivots);
    assert_eq!(blocked_sign, plain_sign);
    let err: f32 = blocked
        .iter()
        .zip(&plain)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(err < 1e-2, "blocked-vs-unblocked max factor diff {err}");
}

#[test]
fn sub_panel_orders_are_bitwise_identical_to_unblocked() {
    // At most one panel and no trailing update: the blocked kernel runs
    // the same step sequence as unblocked elimination, so the factors,
    // pivots, and sign agree bit for bit.
    for n in [1usize, 2, 63, 64] {
        let data = diagonally_dominant(n);
        let mut blocked = data.clone();
        let mut blocked_pivots: Vec<usize> = (0..n).collect();
        let mut blocked_sign = 1i8;
        lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
            .expect("fixture is nonsingular");
        let mut plain = data.clone();
        let mut plain_pivots: Vec<usize> = (0..n).collect();
        let mut plain_sign = 1i8;
        eliminate_unblocked(&mut plain, n, &mut plain_pivots, &mut plain_sign);
        assert_eq!(blocked, plain, "factors differ at n = {n}");
        assert_eq!(blocked_pivots, plain_pivots, "pivots differ at n = {n}");
        assert_eq!(blocked_sign, plain_sign, "sign differs at n = {n}");
    }
}

#[test]
fn partial_second_panel_matches_unblocked() {
    // n = 65: full panel, a 1×1 GEMM update, then a 1-wide panel.
    let n = 65usize;
    let data = diagonally_dominant(n);
    let mut blocked = data.clone();
    let mut blocked_pivots: Vec<usize> = (0..n).collect();
    let mut blocked_sign = 1i8;
    lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
        .expect("fixture is nonsingular");
    let mut plain = data.clone();
    let mut plain_pivots: Vec<usize> = (0..n).collect();
    let mut plain_sign = 1i8;
    eliminate_unblocked(&mut plain, n, &mut plain_pivots, &mut plain_sign);
    assert_eq!(blocked_pivots, plain_pivots);
    assert_eq!(blocked_sign, plain_sign);
    let err = max_abs_diff(&blocked, &plain);
    assert!(err < 1e-9, "blocked-vs-unblocked max factor diff {err}");
}

#[test]
fn blocked_path_rejects_singular_like_unblocked() {
    // Row 2 = 2 × row 1: rank deficient; both paths report the zero
    // pivot column with the same reason.
    let mut blocked = vec![1.0f64, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
    let mut pivots: Vec<usize> = (0..3).collect();
    let mut sign = 1i8;
    let blocked_err = lu_eliminate_blocked(&mut blocked, 3, &mut pivots, &mut sign)
        .expect_err("singular matrix must fail");
    assert!(
        format!("{blocked_err:?}").contains("singular"),
        "unexpected error: {blocked_err:?}"
    );
    let mut plain = vec![1.0f64, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
    let mut pivots: Vec<usize> = (0..3).collect();
    let mut sign = 1i8;
    let plain_err = (0..3)
        .try_for_each(|k| lu_eliminate_step(&mut plain, 3, k, 3, &mut pivots, &mut sign))
        .expect_err("singular matrix must fail");
    assert!(
        format!("{plain_err:?}").contains("singular"),
        "unexpected error: {plain_err:?}"
    );
}

#[test]
fn gate_pins_the_spill_boundary() {
    // Against the 8 MiB fallback LLC: exactly 8 MiB stays unblocked
    // (the gate is `>`, not `>=`), one element over routes blocked.
    let fallback = CacheGeometry::fallback();
    assert!(!lu_uses_blocked_path(fallback, 1024, 8));
    assert!(lu_uses_blocked_path(fallback, 1025, 8));
    assert!(!lu_uses_blocked_path(fallback, 1448, 4));
    assert!(lu_uses_blocked_path(fallback, 1449, 4));
    assert!(!lu_uses_blocked_path(fallback, 0, 8));
    // Saturating arithmetic: absurd sizes route blocked, never wrap.
    assert!(lu_uses_blocked_path(fallback, usize::MAX, 8));
}
