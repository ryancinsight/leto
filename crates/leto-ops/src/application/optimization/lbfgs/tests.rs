use super::solver::dot;
use super::{minimize, LbfgsConfig, LbfgsMemory};

/// f(x) = ½ xᵀA x − bᵀx with SPD A → minimiser x* = A⁻¹b.
#[test]
fn lbfgs_minimises_spd_quadratic() {
    // A = [[3,1],[1,2]], b = [1,1]; x* = [0.2, 0.4]
    let a = [[3.0, 1.0], [1.0, 2.0]];
    let b = [1.0, 1.0];
    let f = |x: &[f64]| {
        let ax0 = a[0][0] * x[0] + a[0][1] * x[1];
        let ax1 = a[1][0] * x[0] + a[1][1] * x[1];
        0.5 * (x[0] * ax0 + x[1] * ax1) - (b[0] * x[0] + b[1] * x[1])
    };
    let grad = |x: &[f64]| {
        vec![
            a[0][0] * x[0] + a[0][1] * x[1] - b[0],
            a[1][0] * x[0] + a[1][1] * x[1] - b[1],
        ]
    };
    let res = minimize(vec![0.0, 0.0], f, grad, LbfgsConfig::default());
    assert!(res.converged, "L-BFGS should converge on a quadratic");
    assert!((res.x[0] - 0.2).abs() < 1e-6, "x0 = {}", res.x[0]);
    assert!((res.x[1] - 0.4).abs() < 1e-6, "x1 = {}", res.x[1]);
    // a quadratic is solved in very few quasi-Newton steps
    assert!(res.iterations <= 15, "took {} iters", res.iterations);
}

/// Separable convex objective Σ (xᵢ − tᵢ)⁴ → minimiser is t.
#[test]
fn lbfgs_minimises_quartic_well() {
    let t = [1.5, -2.0, 0.7, 3.1];
    let f = |x: &[f64]| {
        x.iter()
            .zip(t)
            .map(|(xi, ti)| (xi - ti).powi(4))
            .sum::<f64>()
    };
    let grad = |x: &[f64]| {
        x.iter()
            .zip(t)
            .map(|(xi, ti)| 4.0 * (xi - ti).powi(3))
            .collect::<Vec<_>>()
    };
    let cfg = LbfgsConfig {
        gtol: 1e-10,
        max_iters: 500,
        ..Default::default()
    };
    let res = minimize(vec![0.0; 4], f, grad, cfg);
    for (xi, ti) in res.x.iter().zip(t) {
        assert!((xi - ti).abs() < 1e-2, "got {xi}, want {ti}");
    }
}

#[test]
fn lbfgs_returns_immediately_at_optimum() {
    // start at the minimiser of ½‖x‖² (gradient zero)
    let f = |x: &[f64]| 0.5 * x.iter().map(|v| v * v).sum::<f64>();
    let grad = |x: &[f64]| x.to_vec();
    let res = minimize(vec![0.0, 0.0, 0.0], f, grad, LbfgsConfig::default());
    assert!(res.converged);
    assert_eq!(res.iterations, 0);
}

/// The ring evicts in strict insertion order: a `push` past `memory` is
/// the only case where the conversion changed externally observable
/// behavior (the old `Vec::remove(0)` shifted everything, the new ring
/// overwrites the slot).
///
/// Sanity floor: after deliberately filling the ring past capacity, the
/// stored history length saturates at `memory` (LinkedIn-style eviction).
#[test]
fn ring_evicts_oldest_at_capacity() {
    let mut mem = LbfgsMemory::new(3);
    for k in 1..=7u32 {
        let s = vec![k as f64, 0.0];
        // any `y` with sᵀy > 1e-12; here `y = s` for simplicity.
        let y = s.clone();
        assert!(mem.push(s, y), "pair {k} should be accepted (curvature ok)");
    }
    assert_eq!(mem.len(), 3, "ring should saturate at capacity 3");
}

/// Reproducible regression test for the two-loop recursion over a wrapped
/// ring: rotate the correction pairs, then verify the `direction` agrees
/// with an independent textbook implementation that indexes history by
/// logical insertion order (newest == index 0). This is the
/// reduction-order-sensitive oracle the migration has to preserve (the
/// diff against the pre-conversion history-order reference).
#[test]
fn direction_preserves_two_loop_after_wrap() {
    // Reference (jagged) implementation, written independently of the
    // internal storage so its correctness stands on its own.
    fn reference_inverse_dot(
        s_hist: &[Vec<f64>],
        y_hist: &[Vec<f64>],
        rho_hist: &[f64],
        g: &[f64],
    ) -> Vec<f64> {
        let k = s_hist.len();
        if k == 0 {
            return g.iter().map(|&gi| -gi).collect();
        }
        let mut q = g.to_vec();
        let mut alpha = vec![0.0_f64; k];
        // Two-loop recursion: walk newest → oldest.
        for i in (0..k).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            q.iter_mut()
                .zip(&y_hist[i])
                .for_each(|(qj, &yj)| *qj -= a * yj);
        }
        let s_newest = &s_hist[k - 1];
        let y_newest = &y_hist[k - 1];
        let sy = dot(s_newest, y_newest);
        let yy = dot(y_newest, y_newest);
        let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();
        for i in 0..k {
            let s_i = &s_hist[i];
            let y_i = &y_hist[i];
            let beta = rho_hist[i] * dot(y_i, &r);
            let coef = alpha[i] - beta;
            r.iter_mut()
                .zip(s_i.iter())
                .for_each(|(rj, &sj)| *rj += coef * sj);
        }
        r.iter().map(|&ri| -ri).collect()
    }

    // Build a small ring big enough to wrap, then push more pairs than
    // capacity to cross a couple of boundaries.
    let mut mem = LbfgsMemory::new(2);
    let pushed: Vec<(Vec<f64>, Vec<f64>)> = (1u32..=4)
        .map(|k| {
            let s = vec![k as f64, 2.0 * k as f64];
            let y = vec![1.0 + k as f64 / 4.0, 0.5];
            (s, y)
        })
        .collect();
    // Build the canonical jagged history the reference expects *after*
    // FIFO eviction: index 0 == oldest surviving pair, index k-1 == newest,
    // exactly the convention `reference_inverse_dot` (textbook Nocedal &
    // Wright Alg. 7.5) indexes against. The ring evicted pairs 1 and 2; the
    // survivors are 3 (oldest) and 4 (newest), kept in age order.
    let mut s_ref: Vec<Vec<f64>> = Vec::new();
    let mut y_ref: Vec<Vec<f64>> = Vec::new();
    let mut rho_ref: Vec<f64> = Vec::new();
    for (s, y) in pushed.iter().skip(pushed.len() - 2) {
        let sy = dot(s, y);
        s_ref.push(s.clone());
        y_ref.push(y.clone());
        rho_ref.push(1.0 / sy);
    }
    // sanity floor for the reference construction.
    assert_eq!(s_ref.len(), 2);
    // Push pairs into the ring; let it wrap.
    for (s, y) in &pushed {
        assert!(mem.push(s.clone(), y.clone()));
    }
    assert_eq!(mem.len(), 2, "ring should saturate at capacity 2");
    let g = vec![0.3, -0.7];
    let got = mem.direction(&g);
    let want = reference_inverse_dot(&s_ref, &y_ref, &rho_ref, &g);
    assert_eq!(got.len(), want.len());
    for (i, (g_, w_)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g_ - w_).abs() <= 1e-12,
            "direction[{i}] = {g_}, reference {w_}"
        );
    }
}
