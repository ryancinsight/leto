use super::{decompose, solve};
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, Result};

/// Unpivoted symmetric `U D Uᵀ` decomposition.
#[derive(Debug, Clone)]
pub struct UduDecomposition<T> {
    u: Vec<T>,
    d: Vec<T>,
    n: usize,
}

impl<T: RealScalar> UduDecomposition<T> {
    /// Unit upper-triangular factor `U` (`n × n`).
    #[must_use]
    pub fn u(&self) -> Array2<T> {
        Array2::from_shape_vec([self.n, self.n], self.u.clone()).expect("U shape matches storage")
    }

    /// Diagonal entries of `D`.
    #[must_use]
    pub fn diagonal(&self) -> &[T] {
        &self.d
    }

    /// Determinant `∏ᵢ Dᵢᵢ`.
    #[must_use]
    pub fn det(&self) -> T {
        self.d
            .iter()
            .copied()
            .fold(T::ONE, |acc, value| acc.mul(value))
    }

    /// Solve `A x = rhs` using the factored `A = U D Uᵀ`.
    ///
    /// # Errors
    /// [`leto::LetoError`] on a right-hand-side shape mismatch.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        solve::solve(&self.as_factored(), rhs)
    }

    /// Inverse `A⁻¹`.
    ///
    /// # Errors
    /// Propagates allocation or internal shape errors from the output array
    /// constructors.
    pub fn inv(&self) -> Result<Array2<T>> {
        solve::inverse(&self.as_factored())
    }

    fn as_factored(&self) -> decompose::Factored<T> {
        decompose::Factored {
            u: self.u.clone(),
            d: self.d.clone(),
            n: self.n,
        }
    }
}

/// Factor a symmetric matrix as `A = U D Uᵀ` without pivoting.
///
/// # Errors
/// `LetoError::ShapeMismatch` for non-square input.
/// `LetoError::StorageError` for non-finite entries, nonsymmetric input, or
/// a zero pivot requiring symmetric pivoting.
pub fn udu_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<UduDecomposition<T>> {
    let f = decompose::factor(matrix)?;
    Ok(UduDecomposition {
        u: f.u,
        d: f.d,
        n: f.n,
    })
}
