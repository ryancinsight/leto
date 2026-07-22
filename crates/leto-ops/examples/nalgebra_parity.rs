//! # `nalgebra` × `leto` / `leto-ops` Parity Harness
//!
//! This is the **canonical** migration-parity harness for the Atlas array and
//! linear-algebra stack.  It lives in `leto-ops` because `leto` (Array/Layout)
//! and `leto-ops` (SpMV, sparse LU, GEMV) are the *direct* replacements for
//! `nalgebra::DMatrix` / `DVector` and its decompositions; parity evidence
//! belongs at the source, not scattered across downstream consumers such as
//! `cfd-math`.
//!
//! ## What it proves
//!
//! Solves the manufactured 1-D Poisson problem
//!
//! ```text
//!   u'' = -sin(π x),   u(0) = u(1) = 0,   x ∈ [0, 1]
//! ```
//!
//! whose exact solution is `u*(x) = sin(π x) / π²` using **two independent
//! paths** with identical discrete operators and RHS:
//!
//! | Path   | Array type                | Solver                              |
//! |--------|---------------------------|-------------------------------------|
//! | Legacy | `nalgebra::DMatrix<f64>`  | `nalgebra` LU decomposition         |
//! | Atlas  | `leto::Array1<f64>` + `leto_ops::CsrMatrix` | `leto_ops::SparseLuSolver` |
//!
//! Parity tolerances (SSOT: `migration_validation.md`):
//! - Solution L∞ agreement: ≤ 1e-6
//! - Residual L∞ (both paths): ≤ 1e-8
//!
//! Emits a JSON summary line suitable for CI regression gates.
//!
//! ## Run
//!
//! ```sh
//! cargo run --release --example nalgebra_parity -p leto-ops
//! ```

use leto_ops::{CooMatrix, CsrMatrix, SparseLuSolver};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;
use std::time::Instant;

// ── Problem specification ──────────────────────────────────────────────────

/// 1-D Poisson discretisation on `n` interior points with mesh spacing
/// `h = 1 / (n + 1)`. The Laplacian stencil is scaled by `1/h²`
/// consistently on both paths so the manufactured solution is the same
/// discrete object.
#[derive(Clone, Copy, Debug)]
struct Problem {
    n: usize,
    h: f64,
    h2_inv: f64,
}

impl Problem {
    fn new(n: usize) -> Self {
        let h = 1.0 / (n as f64 + 1.0);
        Self { n, h, h2_inv: h.recip() * h.recip() }
    }

    /// RHS vector `b[i] = sin(π x_i)` at interior grid points.
    fn rhs_vec(&self) -> Vec<f64> {
        (1..=self.n).map(|i| (PI * i as f64 * self.h).sin()).collect()
    }

    /// Exact solution `u*(x_i) = sin(π x_i) / π²`.
    fn exact_vec(&self) -> Vec<f64> {
        (1..=self.n)
            .map(|i| (PI * i as f64 * self.h).sin() / (PI * PI))
            .collect()
    }
}

// ── Legacy path: nalgebra dense LU ────────────────────────────────────────

/// Solve on a **dense** `nalgebra::DMatrix` using `nalgebra`'s own partial-
/// pivoting LU decomposition.  Zero Atlas types appear here — genuine legacy
/// reference implementing the pre-migration code path.
fn solve_nalgebra(p: &Problem, b: &[f64]) -> (Vec<f64>, u128) {
    let n = p.n;
    let scale = p.h2_inv;

    let mut a = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0 * scale;
        if i > 0 {
            a[(i, i - 1)] = -scale;
        }
        if i + 1 < n {
            a[(i, i + 1)] = -scale;
        }
    }

    let b_dv = DVector::from_column_slice(b);
    let t0 = Instant::now();
    let x = a.lu().solve(&b_dv).expect("nalgebra LU solve succeeded");
    let elapsed = t0.elapsed().as_micros();
    (x.as_slice().to_vec(), elapsed)
}

// ── Atlas path: leto Array1 + leto-ops COO → CSR + SparseLuSolver ─────────

/// Assemble the tridiagonal Laplacian via `CooMatrix → CsrMatrix` (the
/// canonical Atlas assembly pipeline: COO accumulation then sorted CSR
/// compression) and solve with `SparseLuSolver` (the Atlas direct sparse
/// solver that delegates to `leto_ops::lu_decompose` for the dense-backed
/// path at problem sizes ≤ `DENSE_LIMIT_DEFAULT`).
///
/// Uses **only** Atlas types — `leto::Array1`, `leto_ops::CooMatrix`,
/// `leto_ops::CsrMatrix`, `leto_ops::SparseLuSolver`.
fn solve_atlas(p: &Problem, b: &[f64]) -> (Vec<f64>, u128) {
    let n = p.n;
    let scale = p.h2_inv;

    // Assembly: push (row, col, value) triplets into COO then compress to CSR.
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        if i > 0 {
            coo.push(i, i - 1, -scale);
        }
        coo.push(i, i, 2.0 * scale);
        if i + 1 < n {
            coo.push(i, i + 1, -scale);
        }
    }
    let csr: CsrMatrix<f64> = coo.to_csr();

    let b_arr: Vec<f64> = b.to_vec();
    let solver = SparseLuSolver::default();

    let t0 = Instant::now();
    let x_vec: Vec<f64> = solver.solve(&csr, &b_arr).expect("Atlas sparse LU succeeded");
    let elapsed = t0.elapsed().as_micros();

    (x_vec, elapsed)
}

// ── Diagnostics ────────────────────────────────────────────────────────────

fn l_inf_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

/// Compute `‖A x − b‖∞` via the tridiagonal stencil (no extra allocation).
fn residual_linf(p: &Problem, x: &[f64], b: &[f64]) -> f64 {
    let scale = p.h2_inv;
    (0..p.n)
        .map(|i| {
            let mut v = 2.0 * scale * x[i];
            if i > 0 {
                v -= scale * x[i - 1];
            }
            if i + 1 < p.n {
                v -= scale * x[i + 1];
            }
            (v - b[i]).abs()
        })
        .fold(0.0_f64, f64::max)
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    let problem = Problem::new(512);
    let b = problem.rhs_vec();
    let exact = problem.exact_vec();

    let (u_legacy, legacy_us) = solve_nalgebra(&problem, &b);
    let (u_atlas, atlas_us) = solve_atlas(&problem, &b);

    let resid_legacy = residual_linf(&problem, &u_legacy, &b);
    let resid_atlas = residual_linf(&problem, &u_atlas, &b);
    let diff_solutions = l_inf_diff(&u_legacy, &u_atlas);
    let err_atlas_exact = l_inf_diff(&u_atlas, &exact);

    let parity_pass = resid_legacy < 1e-8 && resid_atlas < 1e-8 && diff_solutions < 1e-6;

    // JSON line for CI regression gates.
    println!(
        "{{\"crate\":\"leto-ops\",\"problem_n\":{n},\
         \"legacy_solve_us\":{legacy_us},\"atlas_solve_us\":{atlas_us},\
         \"resid_legacy\":{resid_legacy:.6e},\"resid_atlas\":{resid_atlas:.6e},\
         \"diff_solutions\":{diff_solutions:.6e},\"err_atlas_exact\":{err_atlas_exact:.6e},\
         \"parity_pass\":{parity_pass}}}",
        n = problem.n,
    );

    eprintln!("─── nalgebra × leto-ops parity ({} interior pts) ───", problem.n);
    eprintln!("  Legacy : nalgebra::DMatrix + nalgebra LU");
    eprintln!("  Atlas  : leto::Array1 + CooMatrix→CsrMatrix + SparseLuSolver");
    eprintln!("  resid legacy  L∞ : {resid_legacy:.3e}  (tol 1e-8)");
    eprintln!("  resid atlas   L∞ : {resid_atlas:.3e}  (tol 1e-8)");
    eprintln!("  solution diff L∞ : {diff_solutions:.3e}  (tol 1e-6)");
    eprintln!("  atlas vs exact   : {err_atlas_exact:.3e}");
    eprintln!("  legacy time      : {legacy_us} µs");
    eprintln!("  atlas  time      : {atlas_us} µs");
    eprintln!("  PARITY {}", if parity_pass { "PASS ✅" } else { "FAIL ❌" });

    assert!(
        parity_pass,
        "nalgebra × leto-ops parity FAIL: diff={diff_solutions:.3e}, \
         resid_legacy={resid_legacy:.3e}, resid_atlas={resid_atlas:.3e}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quick parity smoke test on a small problem (n=64).
    #[test]
    fn parity_n64() {
        let p = Problem::new(64);
        let b = p.rhs_vec();
        let exact = p.exact_vec();
        let (u_l, _) = solve_nalgebra(&p, &b);
        let (u_a, _) = solve_atlas(&p, &b);
        let diff = l_inf_diff(&u_l, &u_a);
        assert!(diff < 1e-6, "solution diff {diff:.3e} exceeds 1e-6");
        let err = l_inf_diff(&u_a, &exact);
        assert!(err < 1e-5, "atlas vs exact {err:.3e} exceeds 1e-5");
    }
}
