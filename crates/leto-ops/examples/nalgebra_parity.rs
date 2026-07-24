//! Runnable Leto-to-Leto migration evidence.
//!
//! Both providers solve the manufactured one-dimensional Poisson system
//!  with homogeneous Dirichlet boundaries. The legacy path uses
//! a dense Leto matrix and LU; the Atlas path assembles Leto Ops COO/CSR,
//! carries its right-hand side in a Leto Array1, and uses SparseLuSolver.
//! The latter intentionally exercises the solver's documented dense-backed
//! boundary for systems within DENSE_LIMIT_DEFAULT.
//!
//! Verification is independent of provider agreement:
//!
//! - each normalized residual is bounded by gamma(3n), the backward-error bound
//!   for partial-pivoting LU on this diagonally dominant, growth-factor-one
//!   Poisson matrix;
//! - both solutions are compared with the exact discrete sine eigenmode;
//! - the forward bounds use the exact infinity-norm condition number
//!   kappa_inf(A) = 2 max_i i(n + 1 - i) and the standard perturbation inequality
//!   kappa * eta / (1 - kappa * eta).
//!
//! No timing is emitted: performance evidence belongs in a controlled
//! Criterion benchmark, not a one-shot example.
//!
//! Run with:
//!
//! cargo run --locked -p leto-ops --example nalgebra_parity

mod support;

use leto::{Array1, Array2, Storage};
use leto_ops::{CooMatrix, CsrMatrix, SparseLuSolver};
use std::f64::consts::PI;
use support::{gamma, max_abs_diff, Observation};

#[derive(Clone, Copy, Debug)]
struct Problem {
    order: usize,
    spacing: f64,
    inverse_spacing_squared: f64,
}

impl Problem {
    fn new(order: usize) -> Self {
        assert!(order >= 3, "the parity problem requires an interior row");
        let spacing = 1.0 / (order as f64 + 1.0);
        Self {
            order,
            spacing,
            inverse_spacing_squared: spacing.recip() * spacing.recip(),
        }
    }

    fn rhs(&self) -> Vec<f64> {
        (1..=self.order)
            .map(|index| (PI * index as f64 * self.spacing).sin())
            .collect()
    }

    /// Exact solution of the discrete operator, using its first sine
    /// eigenvalue 4 h^-2 sin^2(pi*h/2) rather than the continuum eigenvalue pi^2.
    fn discrete_solution(&self) -> Vec<f64> {
        let half_angle = 0.5 * PI * self.spacing;
        let eigenvalue = 4.0 * self.inverse_spacing_squared * half_angle.sin() * half_angle.sin();
        (1..=self.order)
            .map(|index| (PI * index as f64 * self.spacing).sin() / eigenvalue)
            .collect()
    }

    fn continuum_solution(&self) -> Vec<f64> {
        (1..=self.order)
            .map(|index| (PI * index as f64 * self.spacing).sin() / (PI * PI))
            .collect()
    }

    /// Exact infinity-norm condition number for the scaled Dirichlet
    /// tridiagonal matrix. Scaling by h^-2 cancels between A and A^-1.
    fn condition_number_infinity(&self) -> f64 {
        let left = self.order.div_ceil(2);
        let right = self.order + 1 - left;
        2.0 * (left * right) as f64
    }

    fn matrix_norm_infinity(&self) -> f64 {
        4.0 * self.inverse_spacing_squared
    }
}

fn solve_dense(problem: &Problem, rhs: &[f64]) -> Vec<f64> {
    // Dense path: assemble a full Leto Array2, convert to CSR via from_dense,
    // and solve with SparseLuSolver (which dispatches to the dense-backed
    // boundary for systems within DENSE_LIMIT_DEFAULT).
    let mut matrix = Array2::<f64>::zeros([problem.order, problem.order]);
    {
        let storage = matrix.as_slice_mut().expect("Array2 storage is contiguous");
        for row in 0..problem.order {
            let diag = 2.0 * problem.inverse_spacing_squared;
            let off = -problem.inverse_spacing_squared;
            storage[row * problem.order + row] = diag;
            if row > 0 {
                storage[row * problem.order + (row - 1)] = off;
            }
            if row + 1 < problem.order {
                storage[row * problem.order + (row + 1)] = off;
            }
        }
    }
    let csr = CsrMatrix::from_dense(&matrix.view());
    SparseLuSolver::default()
        .solve(&csr, rhs)
        .expect("the system order and positive-definite matrix satisfy the solver contract")
}

fn solve_sparse(problem: &Problem, rhs: &[f64]) -> Vec<f64> {
    let mut matrix = CooMatrix::<f64>::new(problem.order, problem.order);
    for row in 0..problem.order {
        if row > 0 {
            matrix.push(row, row - 1, -problem.inverse_spacing_squared);
        }
        matrix.push(row, row, 2.0 * problem.inverse_spacing_squared);
        if row + 1 < problem.order {
            matrix.push(row, row + 1, -problem.inverse_spacing_squared);
        }
    }
    let matrix: CsrMatrix<f64> = matrix.to_csr();
    let rhs = Array1::from_shape_vec([problem.order], rhs.to_vec())
        .expect("right-hand-side length matches the system order");
    SparseLuSolver::default()
        .solve(&matrix, rhs.storage().as_slice())
        .expect("the system order and positive-definite matrix satisfy the solver contract")
}

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn residual_infinity(problem: &Problem, solution: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(solution.len(), problem.order);
    assert_eq!(rhs.len(), problem.order);
    (0..problem.order)
        .map(|row| {
            let mut value = 2.0 * problem.inverse_spacing_squared * solution[row];
            if row > 0 {
                value -= problem.inverse_spacing_squared * solution[row - 1];
            }
            if row + 1 < problem.order {
                value -= problem.inverse_spacing_squared * solution[row + 1];
            }
            (value - rhs[row]).abs()
        })
        .fold(0.0_f64, f64::max)
}

fn normalized_backward_error(problem: &Problem, solution: &[f64], rhs: &[f64]) -> f64 {
    let denominator =
        problem.matrix_norm_infinity() * maximum_absolute(solution) + maximum_absolute(rhs);
    residual_infinity(problem, solution, rhs) / denominator
}

fn forward_error_bound(problem: &Problem, solution_scale: f64, backward_bound: f64) -> f64 {
    let conditioned_error = problem.condition_number_infinity() * backward_bound;
    assert!(
        conditioned_error < 1.0,
        "forward-error inequality requires condition * backward error < 1"
    );
    solution_scale * conditioned_error / (1.0 - conditioned_error)
}

fn observations(problem: &Problem) -> ([Observation; 5], f64) {
    let rhs = problem.rhs();
    let discrete_solution = problem.discrete_solution();
    let continuum_solution = problem.continuum_solution();
    let dense_solution = solve_dense(problem, &rhs);
    let sparse_solution = solve_sparse(problem, &rhs);

    let backward_bound = gamma(3 * problem.order);
    let solution_scale = maximum_absolute(&discrete_solution);
    let provider_forward_bound = forward_error_bound(problem, solution_scale, backward_bound);
    let observations = [
        Observation::new(
            "dense_backward",
            normalized_backward_error(problem, &dense_solution, &rhs),
            backward_bound,
        ),
        Observation::new(
            "sparse_backward",
            normalized_backward_error(problem, &sparse_solution, &rhs),
            backward_bound,
        ),
        Observation::new(
            "dense_discrete",
            max_abs_diff(&dense_solution, &discrete_solution),
            provider_forward_bound,
        ),
        Observation::new(
            "sparse_discrete",
            max_abs_diff(&sparse_solution, &discrete_solution),
            provider_forward_bound,
        ),
        Observation::new(
            "provider_agreement",
            max_abs_diff(&dense_solution, &sparse_solution),
            2.0 * provider_forward_bound,
        ),
    ];
    let discretization_error = max_abs_diff(&discrete_solution, &continuum_solution);
    (observations, discretization_error)
}

fn run(order: usize) {
    let problem = Problem::new(order);
    let (observations, discretization_error) = observations(&problem);
    for observation in observations {
        eprintln!(
            "{:<20} error={:.6e} bound={:.6e}",
            observation.name, observation.error, observation.bound
        );
        observation.assert_within_bound();
    }
    eprintln!("continuum discretization error={discretization_error:.6e}");
    println!(
        "{{\"crate\":\"leto-ops\",\"harness\":\"dense_sparse_parity\",\"problem_n\":{order},\"checks\":{},\"all_pass\":true}}",
        observations.len()
    );
}

fn main() {
    run(512);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_solve_parity() {
        let problem = Problem::new(64);
        let (observations, _) = observations(&problem);
        for observation in observations {
            observation.assert_within_bound();
        }
    }
}
