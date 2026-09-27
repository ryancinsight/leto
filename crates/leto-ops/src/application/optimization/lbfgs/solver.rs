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
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
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
