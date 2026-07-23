//! CPU finite-difference stencil implementations.

use crate::RealScalar;
use eunomia::NumericElement;
use leto::{ArrayView1, ArrayViewMut1, BoundaryCondition, Laplacian2D, LaplacianError};

/// Apply a validated two-dimensional Laplacian into caller-owned storage.
///
/// The row-major flattening is `index = y * nx + x`. Interior points use the
/// second-order five-point stencil. Boundary formulas are selected by the
/// provider-owned [`BoundaryCondition`] in `stencil`.
///
/// # Errors
///
/// Returns [`LaplacianError`] when either array length differs from the
/// validated grid length.
pub fn laplacian_2d_into<T: RealScalar>(
    stencil: &Laplacian2D<T>,
    input: &ArrayView1<'_, T>,
    output: &mut ArrayViewMut1<'_, T>,
) -> Result<(), LaplacianError> {
    let expected = stencil.len();
    if input.size() != expected {
        return Err(LaplacianError::InputLength {
            expected,
            actual: input.size(),
        });
    }
    if output.size() != expected {
        return Err(LaplacianError::OutputLength {
            expected,
            actual: output.size(),
        });
    }

    let nx = stencil.nx();
    let ny = stencil.ny();
    let [dx_inv2, dy_inv2] = stencil.signed_inverse_spacing_squared();
    let coefficients = BoundaryCoefficients {
        two: T::from_f32(2.0),
        four: T::from_f32(4.0),
        five: T::from_f32(5.0),
    };

    for y in 0..ny {
        for x in 0..nx {
            let index = y * nx + x;
            let center = input[index];
            let mut laplacian = <T as NumericElement>::ZERO;

            if x > 0 && x < nx - 1 {
                laplacian +=
                    (input[index - 1] - coefficients.two * center + input[index + 1]) * dx_inv2;
            } else {
                laplacian += boundary_second_derivative(
                    stencil.boundary(),
                    center,
                    BoundaryNeighbors {
                        inward: if x == 0 {
                            input[index + 1]
                        } else {
                            input[index - 1]
                        },
                        periodic: if x == 0 {
                            [input[y * nx + nx - 2], input[index + 1]]
                        } else {
                            [input[index - 1], input[y * nx + 1]]
                        },
                        neumann: if nx >= 4 {
                            Some(if x == 0 {
                                [input[index + 1], input[index + 2], input[index + 3]]
                            } else {
                                [input[index - 1], input[index - 2], input[index - 3]]
                            })
                        } else {
                            None
                        },
                    },
                    dx_inv2,
                    coefficients,
                );
            }

            if y > 0 && y < ny - 1 {
                laplacian +=
                    (input[index - nx] - coefficients.two * center + input[index + nx]) * dy_inv2;
            } else {
                laplacian += boundary_second_derivative(
                    stencil.boundary(),
                    center,
                    BoundaryNeighbors {
                        inward: if y == 0 {
                            input[index + nx]
                        } else {
                            input[index - nx]
                        },
                        periodic: if y == 0 {
                            [input[(ny - 2) * nx + x], input[index + nx]]
                        } else {
                            [input[index - nx], input[nx + x]]
                        },
                        neumann: if ny >= 4 {
                            Some(if y == 0 {
                                [
                                    input[index + nx],
                                    input[index + 2 * nx],
                                    input[index + 3 * nx],
                                ]
                            } else {
                                [
                                    input[index - nx],
                                    input[index - 2 * nx],
                                    input[index - 3 * nx],
                                ]
                            })
                        } else {
                            None
                        },
                    },
                    dy_inv2,
                    coefficients,
                );
            }

            output[index] = laplacian;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct BoundaryNeighbors<T> {
    inward: T,
    periodic: [T; 2],
    neumann: Option<[T; 3]>,
}

#[derive(Clone, Copy)]
struct BoundaryCoefficients<T> {
    two: T,
    four: T,
    five: T,
}

#[inline]
fn boundary_second_derivative<T: RealScalar>(
    boundary: BoundaryCondition,
    center: T,
    neighbors: BoundaryNeighbors<T>,
    inverse_spacing_squared: T,
    coefficients: BoundaryCoefficients<T>,
) -> T {
    match boundary {
        BoundaryCondition::Dirichlet => {
            (<T as NumericElement>::ZERO - coefficients.two * center) * inverse_spacing_squared
        }
        BoundaryCondition::Neumann => {
            if let Some([u1, u2, u3]) = neighbors.neumann {
                (coefficients.two * center - coefficients.five * u1 + coefficients.four * u2 - u3)
                    * inverse_spacing_squared
            } else {
                (neighbors.inward - coefficients.two * center + neighbors.inward)
                    * inverse_spacing_squared
            }
        }
        BoundaryCondition::Periodic => {
            (neighbors.periodic[0] - coefficients.two * center + neighbors.periodic[1])
                * inverse_spacing_squared
        }
    }
}
