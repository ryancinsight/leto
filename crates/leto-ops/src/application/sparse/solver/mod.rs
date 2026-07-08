//! Iterative linear solvers for sparse matrices: CG (symmetric positive-definite)
//! and GMRES (general non-symmetric). Each solver works over [`CsrMatrix<T>`]
//! and returns a [`SolverResult`] with the solution, iteration count, and final
//! residual. Both compute in-native-precision without allocation in the inner
//! loop beyond the pre-allocated workspace vectors.

pub(crate) mod cg;
pub(crate) mod gmres;

pub use cg::{cg, CgResult};
pub use gmres::{gmres, GmresResult};
