//! Derived first-derivative stencil coefficients to arbitrary even order.
//!
//! # The stencils
//!
//! Both families write the derivative as a sum over `N = half_order` tap
//! pairs, and differ only in where the taps sit.
//!
//! Staggered (Yee) — the derivative is evaluated halfway between samples, so
//! the taps sit at the half-points `a_n = n − ½`:
//!
//! ```text
//!   ∂f/∂x |_{i+1/2} ≈ (1/Δx) Σ_{n=1..N} c_n · ( f_{i+n} − f_{i−n+1} )
//! ```
//!
//! Collocated — the taps sit at whole points `a_n = n`:
//!
//! ```text
//!   ∂f/∂x |_i ≈ (1/Δx) Σ_{n=1..N} c_n · ( f_{i+n} − f_{i−n} )
//! ```
//!
//! Both are accurate to order `2N`. `N = 1` gives the familiar
//! `(f_{i+1} − f_i)/Δx` and `(f_{i+1} − f_{i−1})/2Δx` respectively.
//!
//! # Why derived rather than tabulated
//!
//! Expanding about the evaluation point, `f(+a) − f(−a)` is odd, so only odd
//! derivatives survive:
//!
//! ```text
//!   (1/Δx) Σ_n c_n [f(+a_n Δx) − f(−a_n Δx)]
//!     = 2 Σ_n c_n [ a_n f′ + a_n³ Δx² f‴/6 + a_n⁵ Δx⁴ f⁽⁵⁾/120 + … ]
//! ```
//!
//! Matching `f′` and annihilating the next `N−1` odd derivatives gives the
//! square linear system
//!
//! ```text
//!   Σ_n c_n a_n^{2m+1} = ½·δ_{m,0},     m = 0 … N−1
//! ```
//!
//! solved here through the crate's own LU. A new order is then a parameter
//! rather than a hand-entered constant table to get wrong, and the derivation
//! is checked against the published values for orders 2 to 8 plus a measured
//! order-of-accuracy test.
//!
//! # Precision
//!
//! The linear system is solved in `f64` and its solution converted to `T`.
//! The coefficients are analytical constants of the stencil, not field
//! arithmetic: solving an ill-conditioned Vandermonde-like system in a reduced
//! precision would return noise for a quantity that does not depend on the
//! field's precision at all. Every subsequent operation on field data executes
//! in `T`.
//!
//! # References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on
//!   arbitrarily spaced grids." *Mathematics of Computation* 51(184), 699-706.
//!   DOI: 10.1090/S0025-5718-1988-0935077-0
//! - Levander, A. R. (1988). "Fourth-order finite-difference P-SV seismograms."
//!   *Geophysics* 53(11), 1425-1436. (The staggered fourth-order 9/8, −1/24.)

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, Array2, LetoError, Result};

use crate::application::linalg::lu_decompose;

/// Largest supported half-order, so the largest supported accuracy order is
/// `2 · MAX_HALF_ORDER`.
///
/// The Vandermonde-like system is increasingly ill-conditioned in `N`. At
/// `N = 4` (eighth order — the highest a wave solver here profitably uses) the
/// coefficients match the published rationals to `1e-13` relative. By `N = 8`
/// the high Taylor moments cancel terms of order `10^12` against each other and
/// the residual is only about `1e-11` of the summed magnitude — still far
/// tighter than any discretization error, but no longer exact. The cap keeps
/// the derivation inside its verified range rather than silently returning
/// noise.
pub const MAX_HALF_ORDER: usize = 8;

/// Tap coefficients `c_1 … c_N` of a first-derivative stencil, in units of
/// `1/Δx`.
///
/// Inline storage keeps the owning operator `Copy` and its kernels
/// allocation-free: the derivation allocates once at construction, the sweep
/// never does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TapCoefficients<T> {
    taps: [T; MAX_HALF_ORDER],
    len: usize,
}

impl<T: Copy + NumericElement> TapCoefficients<T> {
    /// The `N` derived taps, `c_1` first.
    #[must_use]
    #[inline]
    pub fn taps(&self) -> &[T] {
        &self.taps[..self.len]
    }

    /// Half-order `N`: the number of tap pairs, and the stencil's reach in
    /// cells on either side of the evaluation point.
    #[must_use]
    #[inline]
    pub fn half_order(&self) -> usize {
        self.len
    }

    /// Accuracy order `2N`.
    #[must_use]
    #[inline]
    pub fn order(&self) -> usize {
        2 * self.len
    }

    /// `Σ_n |c_n|`, the bound on the stencil's Fourier symbol that sets the
    /// scheme's Courant limit.
    #[must_use]
    pub fn absolute_sum(&self) -> T {
        self.taps()
            .iter()
            .fold(<T as NumericElement>::ZERO, |sum, &c| sum + c.abs())
    }
}

/// Derive the staggered first-derivative taps for accuracy order
/// `2 · half_order`.
///
/// # Errors
///
/// Returns [`LetoError::InvalidInput`] when `half_order` is zero or exceeds
/// [`MAX_HALF_ORDER`], or when the derived system is singular (which cannot
/// occur for the well-posed offsets used here, but is surfaced rather than
/// silently producing garbage).
///
/// # Examples
///
/// ```
/// use leto_ops::staggered_first_derivative_coefficients;
///
/// // Second order: the plain half-grid difference.
/// let c = staggered_first_derivative_coefficients::<f64>(1).unwrap();
/// assert!((c.taps()[0] - 1.0).abs() < 1e-14);
///
/// // Fourth order: Levander's 9/8 and -1/24.
/// let c = staggered_first_derivative_coefficients::<f64>(2).unwrap();
/// assert!((c.taps()[0] - 9.0 / 8.0).abs() < 1e-14);
/// assert!((c.taps()[1] + 1.0 / 24.0).abs() < 1e-14);
/// ```
pub fn staggered_first_derivative_coefficients<T>(half_order: usize) -> Result<TapCoefficients<T>>
where
    T: RealField + FloatElement + Copy,
{
    coefficients_for_offsets(half_order, |n| n as f64 + 0.5, "staggered")
}

/// Derive the collocated central first-derivative taps for accuracy order
/// `2 · half_order`.
///
/// The taps are antisymmetric by construction, which is what makes the
/// operator skew-symmetric — and therefore energy-conserving in a leapfrog —
/// once out-of-range taps are treated as zero rather than replaced by a
/// one-sided formula.
///
/// # Errors
///
/// See [`staggered_first_derivative_coefficients`].
///
/// # Examples
///
/// ```
/// use leto_ops::central_first_derivative_coefficients;
///
/// let c = central_first_derivative_coefficients::<f64>(1).unwrap();
/// assert!((c.taps()[0] - 0.5).abs() < 1e-14);
///
/// let c = central_first_derivative_coefficients::<f64>(2).unwrap();
/// assert!((c.taps()[0] - 2.0 / 3.0).abs() < 1e-14);
/// assert!((c.taps()[1] + 1.0 / 12.0).abs() < 1e-14);
/// ```
pub fn central_first_derivative_coefficients<T>(half_order: usize) -> Result<TapCoefficients<T>>
where
    T: RealField + FloatElement + Copy,
{
    coefficients_for_offsets(half_order, |n| n as f64 + 1.0, "central")
}

/// Shared derivation: solve `Σ_n c_n a_n^{2m+1} = ½·δ_{m,0}` for the given tap
/// offsets. Staggered and collocated stencils differ only in where the taps
/// sit, so the linear system and its solve are the same.
fn coefficients_for_offsets<T>(
    half_order: usize,
    offset: impl Fn(usize) -> f64,
    kind: &str,
) -> Result<TapCoefficients<T>>
where
    T: RealField + FloatElement + Copy,
{
    if half_order == 0 || half_order > MAX_HALF_ORDER {
        return Err(LetoError::InvalidInput(format!(
            "{kind} half-order must be 1..={MAX_HALF_ORDER}, got {half_order}"
        )));
    }
    let n = half_order;

    let offsets: Vec<f64> = (0..n).map(&offset).collect();
    let mut matrix = vec![0.0_f64; n * n];
    for m in 0..n {
        let power =
            i32::try_from(2 * m + 1).expect("invariant: 2m+1 fits i32 for m <= MAX_HALF_ORDER");
        for (j, &a) in offsets.iter().enumerate() {
            matrix[m * n + j] = a.powi(power);
        }
    }
    // Only the f′ condition carries a right-hand side.
    let mut rhs = vec![0.0_f64; n];
    rhs[0] = 0.5;

    let matrix = Array2::from_shape_vec([n, n], matrix)
        .map_err(|error| LetoError::InvalidInput(format!("{kind} coefficient matrix: {error}")))?;
    let rhs = Array1::from_shape_vec([n], rhs)
        .map_err(|error| LetoError::InvalidInput(format!("{kind} coefficient rhs: {error}")))?;
    let solution = lu_decompose(&matrix.view())
        .and_then(|lu| lu.solve(&rhs.view()))
        .map_err(|error| {
            LetoError::InvalidInput(format!(
                "{kind} coefficient system of half-order {half_order} is singular: {error}"
            ))
        })?;

    let mut taps = [<T as NumericElement>::ZERO; MAX_HALF_ORDER];
    for (tap, &value) in taps.iter_mut().zip(
        solution
            .as_slice()
            .expect("invariant: a freshly built Array1 is contiguous"),
    ) {
        *tap = T::from_f64(value);
    }
    Ok(TapCoefficients { taps, len: n })
}
