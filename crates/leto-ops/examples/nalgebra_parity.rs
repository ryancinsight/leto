//! Runnable `nalgebra` to Leto migration evidence.
//!
//! Both providers solve the manufactured one-dimensional Poisson system
//! `-u'' = sin(πx)` with homogeneous Dirichlet boundaries. The legacy path uses
//! a dense `nalgebra` matrix and LU; the Atlas path assembles Leto Ops COO/CSR,
//! carries its right-hand side in a Leto `Array1`, and uses `SparseLuSolver`.
//! The latter intentionally exercises the solver's documented dense-backed
//! boundary for systems within `DENSE_LIMIT_DEFAULT`.
//!
//! Verification is independent of provider agreement:
//!
//! - each normalized residual is bounded by `γ₍₃ₙ₎`, the backward-error bound
//!   for partial-pivoting LU on this diagonally dominant, growth-factor-one
//!   Poisson matrix;
//! - both solutions are compared with the exact discrete sine eigenmode;
//! - the forward bounds use the exact infinity-norm condition number
//!   `κ∞(A) = 2 maxᵢ i(n + 1 - i)` and the standard perturbation inequality
//!   `κ η / (1 - κ η)`.
//!
//! No timing is emitted: performance evidence belongs in a controlled
//! Criterion benchmark, not a one-shot example.
//!
//! Run with:
//!
//! ```sh
//! cargo run --locked -p leto-ops --example nalgebra_parity
//! ```

mod support;

use leto::{Array1, Storage};
use leto_ops::{CooMatrix, CsrMatrix, SparseLuSolver};
use nalgebra::{DMatrix, DVector};
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
    /// eigenvalue `4 h⁻² sin²(πh/2)` rather than the continuum eigenvalue `π²`.
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
    /// tridiagonal matrix. Scaling by `h⁻²` cancels between `A` and `A⁻¹`.
    fn condition_number_infinity(&self) -> f64 {
        let left = self.order.div_ceil(2);
        let right = self.order + 1 - left;
        2.0 * (left * right) as f64
    }

    fn matrix_norm_infinity(&self) -> f64 {
        4.0 * self.inverse_spacing_squared
    }
}

fn solve_nalgebra(problem: &Problem, rhs: &[f64]) -> Vec<f64> {
    let mut matrix = DMatrix::<f64>::zeros(problem.order, problem.order);
    for row in 0..problem.order {
        matrix[(row, row)] = 2.0 * problem.inverse_spacing_squared;
        if row > 0 {
            matrix[(row, row - 1)] = -problem.inverse_spacing_squared;
        }
        if row + 1 < problem.order {
            matrix[(row, row + 1)] = -problem.inverse_spacing_squared;
        }
    }
    let rhs = DVector::from_column_slice(rhs);
    matrix
        .lu()
        .solve(&rhs)
        .expect("the Dirichlet Poisson matrix is positive definite")
        .as_slice()
        .to_vec()
}

fn solve_leto(problem: &Problem, rhs: &[f64]) -> Vec<f64> {
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
    let nalgebra_solution = solve_nalgebra(problem, &rhs);
    let leto_solution = solve_leto(problem, &rhs);

    let backward_bound = gamma(3 * problem.order);
    let solution_scale = maximum_absolute(&discrete_solution);
    let provider_forward_bound = forward_error_bound(problem, solution_scale, backward_bound);
    let observations = [
        Observation::new(
            "nalgebra_backward",
            normalized_backward_error(problem, &nalgebra_solution, &rhs),
            backward_bound,
        ),
        Observation::new(
            "leto_backward",
            normalized_backward_error(problem, &leto_solution, &rhs),
            backward_bound,
        ),
        Observation::new(
            "nalgebra_discrete",
            max_abs_diff(&nalgebra_solution, &discrete_solution),
            provider_forward_bound,
        ),
        Observation::new(
            "leto_discrete",
            max_abs_diff(&leto_solution, &discrete_solution),
            provider_forward_bound,
        ),
        Observation::new(
            "provider_agreement",
            max_abs_diff(&nalgebra_solution, &leto_solution),
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
        "{{\"crate\":\"leto-ops\",\"harness\":\"nalgebra_parity\",\"problem_n\":{order},\"checks\":{},\"all_pass\":true}}",
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
