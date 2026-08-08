//! Limited-memory BFGS (L-BFGS) quasi-Newton optimiser.
//!
//! L-BFGS approximates the inverse Hessian from the last `m` gradient/step pairs
//! `(s_k, y_k)` and computes a descent direction by the Nocedal two-loop
//! recursion, giving super-linear convergence without storing or inverting a
//! dense Hessian. It is the standard refinement step for full-waveform inversion
//! and PINN training (Inverse Problems §9.1).
//!
//! # References
//! - Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
//!   *Math. Comp.*, 35(151), 773–782.
//! - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Alg. 7.4–7.5.

/// L-BFGS configuration.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsConfig {
    /// Number of `(s, y)` correction pairs kept (`m`).
    pub memory: usize,
    /// Maximum outer iterations.
    pub max_iters: usize,
    /// Convergence tolerance on the gradient infinity-norm.
    pub gtol: f64,
    /// Armijo sufficient-decrease constant `c₁ ∈ (0, 1)`.
    pub c1: f64,
    /// Maximum backtracking line-search steps per iteration.
    pub max_line_search: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            memory: 8,
            max_iters: 200,
            gtol: 1e-8,
            c1: 1e-4,
            max_line_search: 30,
        }
    }
}

/// Result of an L-BFGS run.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Minimiser estimate.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fx: f64,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Whether the gradient tolerance was met.
    pub converged: bool,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Limited-memory inverse-Hessian state: the last `m` correction pairs
/// `(sₖ, yₖ)` with `sₖ = xₖ₊₁ − xₖ`, `yₖ = ∇f(xₖ₊₁) − ∇f(xₖ)`.
///
/// This is the canonical (SSOT) implementation of the Nocedal two-loop
/// recursion. Both the in-process [`minimize`] driver and externally-driven
/// optimisation loops (e.g. adjoint-state full-waveform inversion, where each
/// objective/gradient evaluation is an expensive PDE solve owned by the caller)
/// share it: the caller computes `(f, ∇f)`, asks for a search [`direction`],
/// runs its own line search, then records the resulting pair via [`push`].
///
/// # Storage layout
///
/// Correction pairs are kept in two flat ring buffers (`s_buf`, `y_buf`) of
/// capacity `memory * n` plus a parallel scalar ring `rho_buf` of capacity
/// `memory`, addressed by a single `head` index modulo `memory`. This is the
/// CSR-shaped form of the textbook sliding window: it removes the per-row
/// allocation of `Vec<Vec<f64>>` and the O(m) `Vec::remove(0)` eviction of the
/// naive FIFO, replacing both with a single in-place overwrite at the ring
/// head. Two-loop traversal reads the ring in reverse insertion order, which
/// is the order the recursion requires anyway.
///
/// [`direction`]: LbfgsMemory::direction
/// [`push`]: LbfgsMemory::push
#[derive(Debug, Clone)]
pub struct LbfgsMemory {
    /// Maximum number of correction pairs (`m`).
    memory: usize,
    /// Problem dimension `n` for stored pairs; recorded on the first [`push`]
    /// and enforced to match on subsequent pushes. `None` until the first pair.
    dim: Option<usize>,
    /// Flat ring buffer for `s` correction vectors, capacity `memory * n`.
    s_buf: Vec<f64>,
    /// Flat ring buffer for `y` correction vectors, capacity `memory * n`.
    y_buf: Vec<f64>,
    /// Parallel scalar ring buffer for the `rho = 1/(sᵀy)` history,
    /// capacity `memory`.
    rho_buf: Vec<f64>,
    /// Index of the next write slot in `[0, memory)`; oldest pair lives at
    /// `head` when the ring is full.
    head: usize,
    /// Number of populated pairs in `[0, memory]`; once it reaches `memory`,
    /// every [`push`] evicts the oldest by overwriting `head`.
    len: usize,
}

impl LbfgsMemory {
    /// Create an empty memory keeping at most `memory` correction pairs.
    #[must_use]
    pub fn new(memory: usize) -> Self {
        Self {
            memory: memory.max(1),
            dim: None,
            s_buf: Vec::new(),
            y_buf: Vec::new(),
            rho_buf: Vec::new(),
            head: 0,
            len: 0,
        }
    }

    /// Number of stored correction pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no correction pairs are stored yet (first iteration).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slice into the flat ring buffer for the `s` vector at logical
    /// insertion-pair index `i` (`0` == newest, `len - 1` == oldest).
    ///
    /// Insertion order walks the ring backward from `head`, so the vector that
    /// was pushed most recently lives one slot *behind* `head` (modulo
    /// `memory`), and the oldest surviving pair lives *at* `head` when the ring
    /// is full.
    #[inline]
    fn pair_slot(&self, i: usize) -> usize {
        // `i = 0` is newest: one step behind the write `head`.
        // `i = len - 1` is oldest: at `head` when the ring is full, or at the
        // first slot when the ring has not yet wrapped.
        debug_assert!(i < self.len, "pair index {i} out of range len={}", self.len);
        (self.head + self.memory - 1 - i) % self.memory
    }

    /// `n`-element `&[f64]` view of the `s` vector at logical insertion-pair
    /// index `i` (`0` == newest).
    #[inline]
    fn s_row(&self, i: usize) -> &[f64] {
        let n = self.dim.expect("direction() called before any push()");
        let slot = self.pair_slot(i);
        &self.s_buf[slot * n..(slot + 1) * n]
    }

    /// `n`-element `&[f64]` view of the `y` vector at logical insertion-pair
    /// index `i` (`0` == newest).
    #[inline]
    fn y_row(&self, i: usize) -> &[f64] {
        let n = self.dim.expect("direction() called before any push()");
        let slot = self.pair_slot(i);
        &self.y_buf[slot * n..(slot + 1) * n]
    }

    /// Descent direction `d = −H·g` from the two-loop recursion, where `H` is
    /// the implicit limited-memory inverse-Hessian approximation. With no stored
    /// pairs this reduces to steepest descent `d = −g`.
    ///
    /// The initial Hessian scaling `γ = (sₖᵀyₖ)/(yₖᵀyₖ)` uses the newest pair
    /// (Nocedal & Wright, Alg. 7.4).
    #[must_use]
    pub fn direction(&self, g: &[f64]) -> Vec<f64> {
        let k = self.len;
        if k == 0 {
            return g.iter().map(|&gi| -gi).collect();
        }
        let n = self
            .dim
            .expect("invariant: len > 0 implies dim was recorded on first push");
        // The caller may pass a gradient whose length differs from `n` after a
        // hot-restart against a re-dimensioned problem; that is a contract
        // violation (the recorded `s/y` history is meaningless for a different
        // `n`), so reject it with the same shape contract as a fresh memory.
        assert_eq! {
            g.len(),
            n,
            "LbfgsMemory::direction: gradient length {} != stored dim {n}",
            g.len()
        };
        let mut q = g.to_vec();
        let mut alpha = vec![0.0_f64; k];
        // Two-loop recursion (Nocedal & Wright, Alg. 7.5). The ring's logical
        // index `i` runs newest (`i = 0`, one step behind `head`) to oldest
        // (`i = k - 1`, at `head` when full). The first pass walks
        // newest→oldest computing α; the second pass walks oldest→newest
        // accumulating the search direction — mirror the textbook index order
        // by reversing the loop range, not the ring indexing.
        //
        // "i" indexes the logical ring position (`s_row`/`y_row`/`pair_slot` all
        // resolve the slot from it) and `alpha` in lockstep; enumerate over
        // `alpha` would lose the logical-index contract, so each loop is a
        // genuine range loop, not an `iter().enumerate()` candidate.
        #[expect(
            clippy::needless_range_loop,
            reason = "i is the logical ring index for s_row/y_row/pair_slot, not just alpha position"
        )]
        for i in 0..k {
            let s_i = self.s_row(i);
            let y_i = self.y_row(i);
            let rho_i = self.rho_buf[self.pair_slot(i)];
            let a = rho_i * dot(s_i, &q);
            alpha[i] = a;
            q.iter_mut()
                .zip(y_i.iter())
                .for_each(|(qj, &yj)| *qj -= a * yj);
        }
        // γ uses the newest pair, which is logical index 0 (one step behind head).
        let s_newest = self.s_row(0);
        let y_newest = self.y_row(0);
        let sy = dot(s_newest, y_newest);
        let yy = dot(y_newest, y_newest);
        let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();
        // Walk oldest (i = k-1) → newest (i = 0) by reversing the range.
        for i in (0..k).rev() {
            let s_i = self.s_row(i);
            let y_i = self.y_row(i);
            let rho_i = self.rho_buf[self.pair_slot(i)];
            let beta = rho_i * dot(y_i, &r);
            let coef = alpha[i] - beta;
            r.iter_mut()
                .zip(s_i.iter())
                .for_each(|(rj, &sj)| *rj += coef * sj);
        }
        r.iter().map(|&ri| -ri).collect()
    }

    /// Record a correction pair, evicting the oldest when full. The pair is
    /// stored only if the curvature condition `sᵀy > 1e-12` holds (skipping
    /// preserves positive-definiteness of the implicit inverse-Hessian); returns
    /// whether it was stored.
    ///
    /// The pair overwrites the slot at the current ring `head` in place — no
    /// `Vec::remove(0)` and no per-vector realloc — and then advances `head`
    /// modulo `memory`, so a full ring evicts by overwriting rather than by
    /// shifting. Both vectors must share the same length; the dimension is
    /// recorded on the first accepted push and enforced on every later one.
    pub fn push(&mut self, s: Vec<f64>, y: Vec<f64>) -> bool {
        let n = s.len();
        if n == 0 || n != y.len() {
            return false;
        }
        let sy = dot(&s, &y);
        if sy <= 1e-12 {
            return false;
        }
        match self.dim {
            None => {
                // First accepted pair: allocate the ring buffers at full capacity.
                self.dim = Some(n);
                self.s_buf = vec![0.0_f64; self.memory * n];
                self.y_buf = vec![0.0_f64; self.memory * n];
                self.rho_buf = vec![0.0_f64; self.memory];
            }
            Some(recorded) => {
                debug_assert_eq!(
                    recorded, n,
                    "LbfgsMemory::push: pair length {n} != stored dim {recorded}"
                );
            }
        }
        let slot = self.head;
        self.s_buf[slot * n..(slot + 1) * n].copy_from_slice(&s);
        self.y_buf[slot * n..(slot + 1) * n].copy_from_slice(&y);
        self.rho_buf[slot] = 1.0 / sy;
        self.head = (self.head + 1) % self.memory;
        if self.len < self.memory {
            self.len += 1;
        }
        true
    }
}

/// Minimise `f` with gradient `grad`, starting from `x0`, via L-BFGS.
///
/// `f: &[f64] -> f64` is the objective; `grad: &[f64] -> Vec<f64>` its gradient.
/// Returns the minimiser, the objective there, the iteration count, and whether
/// the gradient infinity-norm fell below `config.gtol`.
pub fn minimize<F, G>(x0: Vec<f64>, mut f: F, mut grad: G, config: LbfgsConfig) -> LbfgsResult
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0;
    let mut fx = f(&x);
    let mut g = grad(&x);

    let mut mem = LbfgsMemory::new(config.memory);

    if inf_norm(&g) < config.gtol {
        return LbfgsResult {
            x,
            fx,
            iterations: 0,
            converged: true,
        };
    }

    for it in 1..=config.max_iters {
        // ---- two-loop recursion: direction d = -H·g (shared SSOT) ----
        let dir = mem.direction(&g);

        // ---- Armijo backtracking line search ----
        let gd = dot(&g, &dir); // directional derivative (< 0 for a descent dir)
        let mut step = if mem.is_empty() {
            // first iteration: scale the steepest-descent step
            (1.0 / inf_norm(&g)).min(1.0)
        } else {
            1.0
        };
        let mut x_new = x.clone();
        let mut fx_new = fx;
        let mut accepted = false;
        for _ in 0..config.max_line_search {
            for j in 0..n {
                x_new[j] = x[j] + step * dir[j];
            }
            fx_new = f(&x_new);
            if fx_new <= fx + config.c1 * step * gd {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            // line search failed to make progress → stop
            return LbfgsResult {
                x,
                fx,
                iterations: it,
                converged: inf_norm(&g) < config.gtol,
            };
        }

        let g_new = grad(&x_new);

        // ---- store correction pair (curvature condition enforced inside) ----
        let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        mem.push(s, y);

        x = x_new;
        fx = fx_new;
        g = g_new;

        if inf_norm(&g) < config.gtol {
            return LbfgsResult {
                x,
                fx,
                iterations: it,
                converged: true,
            };
        }
    }

    LbfgsResult {
        x,
        fx,
        iterations: config.max_iters,
        converged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
