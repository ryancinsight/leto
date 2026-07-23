//! Identity preconditioner — no preconditioning (z ← r).

use super::super::traits::Preconditioner;
use eunomia::RealField;
use leto::{Array1, LetoError, Result};

/// Identity preconditioner.
///
/// Applies no transformation: `z ← r`.  Used as a no-op baseline so all
/// iterative solver paths share a uniform preconditioner interface.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityPreconditioner;

impl<T: RealField + Copy> Preconditioner<T> for IdentityPreconditioner {
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<()> {
        let n = r.shape()[0];
        if z.shape()[0] != n {
            return Err(LetoError::InvalidInput(format!(
                "identity preconditioner output length {}, expected {n}",
                z.shape()[0]
            )));
        }
        for i in 0..n {
            z[i] = r[i];
        }
        Ok(())
    }
}
