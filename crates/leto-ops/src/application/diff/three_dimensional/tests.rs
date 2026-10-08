#![expect(
    clippy::unwrap_used,
    reason = "test code: unwraps on fixed fixtures are the floor's test exemption"
)]

use super::*;
use leto::Array3;

// ── X-axis parity (central 2nd/4th/6th are exact for linear/quadratic/quartic) ──

#[test]
fn central2_x_of_linear_function_is_exact() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_second_order(dx, dx, dx).unwrap();
    let mut field = Array3::zeros([20, 5, 5]);
    for i in 0..20 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = 3.7 * (i as f64) * dx + 0.2 * (j as f64) * dx;
            }
        }
    }
    let mut g = Array3::zeros([20, 5, 5]);
    op.apply_x_into(field.view(), &mut g.view_mut()).unwrap();
    for i in 0..20 {
        assert!((g[[i, 2, 2]] - 3.7).abs() < 1e-12, "i={i}");
    }
}

#[test]
fn central4_x_of_quadratic_is_exact() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
    let mut field = Array3::zeros([20, 5, 5]);
    for i in 0..20 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x * x;
            }
        }
    }
    let mut g = Array3::zeros([20, 5, 5]);
    op.apply_x_into(field.view(), &mut g.view_mut()).unwrap();
    for i in 2..18 {
        let x = (i as f64) * dx;
        assert!((g[[i, 2, 2]] - 2.0 * x).abs() < 1e-12, "i={i}");
    }
}

#[test]
fn central6_x_of_quartic_polynomial_is_exact() {
    // 6th-order central is exact for polynomials up to degree 7. Quartic
    // gives ample interior accuracy to floating-point precision (error
    // ~1e-15). Cubic is also exact for 6th-order; we keep quartic here
    // for forward symmetry with central4_x_of_quadratic.
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
    let mut field = Array3::zeros([20, 5, 5]);
    for i in 0..20 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x.powi(4);
            }
        }
    }
    let mut g = Array3::zeros([20, 5, 5]);
    op.apply_x_into(field.view(), &mut g.view_mut()).unwrap();
    for i in 3..17 {
        let x = (i as f64) * dx;
        let expected = 4.0 * x.powi(3);
        assert!((g[[i, 2, 2]] - expected).abs() < 1e-10, "i={i}");
    }
}

// ── Y-axis parity ──────────────────────────────────────────────────────────

#[test]
fn central2_y_of_linear_function_is_exact() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_second_order(dx, dx, dx).unwrap();
    let mut field = Array3::zeros([5, 20, 5]);
    for i in 0..5 {
        for j in 0..20 {
            for k in 0..5 {
                field[[i, j, k]] = 2.5 * (j as f64) * dx + 0.7 * (i as f64) * dx;
            }
        }
    }
    let mut g = Array3::zeros([5, 20, 5]);
    op.apply_y_into(field.view(), &mut g.view_mut()).unwrap();
    for j in 0..20 {
        assert!((g[[2, j, 2]] - 2.5).abs() < 1e-12, "j={j}");
    }
}

// ── Z-axis parity ──────────────────────────────────────────────────────────

#[test]
fn central4_z_of_quadratic_is_exact() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
    let mut field = Array3::zeros([5, 5, 20]);
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..20 {
                let z = (k as f64) * dx;
                field[[i, j, k]] = z * z;
            }
        }
    }
    let mut g = Array3::zeros([5, 5, 20]);
    op.apply_z_into(field.view(), &mut g.view_mut()).unwrap();
    for k in 2..18 {
        let z = (k as f64) * dx;
        assert!((g[[2, 2, k]] - 2.0 * z).abs() < 1e-12, "k={k}");
    }
}

#[test]
fn central6_z_linear_function_is_exact() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
    let mut field = Array3::zeros([5, 5, 20]);
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..20 {
                let z = (k as f64) * dx;
                field[[i, j, k]] = 4.0 * z;
            }
        }
    }
    let mut g = Array3::zeros([5, 5, 20]);
    op.apply_z_into(field.view(), &mut g.view_mut()).unwrap();
    for k in 0..20 {
        assert!((g[[2, 2, k]] - 4.0).abs() < 1e-12, "k={k}");
    }
}

// ── Boundary fall-back parity (central_6 multi-order) ─────────────────────
//
// We use a quintic `u = x⁵` field to make the fall-back error visible.
// - 4th-order central stencil applied to x⁵ introduces O(h⁴) error from
//   the f^{(5)} = 120 term (error coefficient 4·h⁴ ≈ 4·1e-4 = 4e-4).
// - 2nd-order stencil applied to x⁵ has O(h²) error (much larger).
// - 1st-order one-sided at the boundary has O(h) error (largest).
//
// The test asserts the 4th-order fall-back at i=2 lands in the 4·h⁴ ≈ 4e-4
// window, distinct from exact and from the O(h²) baseline.

#[test]
fn central6_x_boundary_fall_back_orders() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
    let nx = 12_usize;
    let mut field = Array3::zeros([nx, 5, 5]);
    for i in 0..nx {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x.powi(5);
            }
        }
    }
    let mut g = Array3::zeros([nx, 5, 5]);
    op.apply_x_into(field.view(), &mut g.view_mut()).unwrap();

    // Interior cell i=6 — 6th-order central, exact for quintic (degree ≤ 7).
    let x_interior = 6.0 * dx;
    let exact_interior: f64 = 5.0 * x_interior.powi(4);
    assert!(
        (g[[interior_i(6), 2, 2]] - exact_interior).abs() < 1e-10,
        "interior mismatch"
    );

    // Near-boundary i=2 — 4th-order fall-back. O(h⁴) error ≈ 4·h⁴ = 4e-4.
    let x_near = 2.0 * dx;
    let exact_near: f64 = 5.0 * x_near.powi(4);
    let err_near = (g[[2, 2, 2]] - exact_near).abs();
    let err_coefficient = err_near / dx.powi(4);
    assert!(
        err_coefficient > 1.0 && err_coefficient < 10.0,
        "4th-order fall-back O(h⁴) error coefficient out of expected band: {err_coefficient}"
    );

    // Near-boundary i=1 — 2nd-order fall-back. O(h²) error ≈ C·h².
    let x_nb1 = 1.0 * dx;
    let exact_nb1: f64 = 5.0 * x_nb1.powi(4);
    let err_nb1 = (g[[1, 2, 2]] - exact_nb1).abs();
    let err_coefficient_2nd = err_nb1 / dx.powi(2);
    assert!(
        err_coefficient_2nd > 0.1,
        "2nd-order fall-back O(h²) error coefficient too small: {err_coefficient_2nd}"
    );

    // Boundary i=0 — 1st-order one-sided forward. For u = x⁵, returns
    // `(f(x+h) - f(x)) / h` at the origin: `(dx⁵ − 0) / dx = dx⁴ = 1e-4`.
    let computed_boundary = (field[[1, 2, 2]] - field[[0, 2, 2]]) / dx;
    assert!(
        (g[[0, 2, 2]] - computed_boundary).abs() < 1e-12,
        "1st-order forward fall-back must return (f[1]-f[0])/dx at i=0"
    );
    // Exact derivative at the origin is 0; the 1st-order value is 1e-4.
    assert!(
        (g[[0, 2, 2]] - 1e-4).abs() < 1e-12,
        "quintic 1st-order @ origin must equal dx⁴ = 1e-4"
    );
}

fn interior_i(_idx: usize) -> usize {
    6
}

// ── Staggered schemes (X + Y + Z shapes) ───────────────────────────────────

#[test]
fn staggered_forward_x_face_centered() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::staggered_forward(dx, dx, dx).unwrap();
    let field = Array3::from_elem((10, 5, 5), 0.0_f64);
    let mut g = Array3::zeros([9, 5, 5]);
    op.apply_x_into(field.view(), &mut g.view_mut()).unwrap();
    for v in g.iter() {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn staggered_backward_x_zero_field() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::staggered_backward(dx, dx, dx).unwrap();
    let field = Array3::from_elem((10, 5, 5), 0.0_f64);
    let mut g = Array3::zeros([10, 5, 5]);
    op.apply_x_into(field.view(), &mut g.view_mut()).unwrap();
    for v in g.iter() {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn staggered_backward_z_mixed_dst_shape() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::staggered_backward(dx, dx, dx).unwrap();
    let field = Array3::from_elem((5, 5, 10), 0.0_f64);
    let mut g = Array3::zeros([5, 5, 10]);
    op.apply_z_into(field.view(), &mut g.view_mut()).unwrap();
    assert_eq!(g.shape(), [5, 5, 10]);
    for v in g.iter() {
        assert_eq!(*v, 0.0);
    }
    // Linear field along z: dst at i=0 forward, interior backward.
    let mut field = Array3::from_elem((5, 5, 10), 0.0_f64);
    for k in 0..10 {
        for i in 0..5 {
            for j in 0..5 {
                field[[i, j, k]] = (k as f64) * dx;
            }
        }
    }
    let mut g = Array3::zeros([5, 5, 10]);
    op.apply_z_into(field.view(), &mut g.view_mut()).unwrap();
    assert!((g[[2, 2, 0]] - 1.0).abs() < 1e-12); // forward fall-back
    for k in 1..10 {
        assert!((g[[2, 2, k]] - 1.0).abs() < 1e-12, "k={k}");
    }
}

// ── dst shape contract (release-mode safety) ──────────────────────────────

#[test]
fn staggered_forward_rejects_dst_shape_mismatch() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::staggered_forward(dx, dx, dx).unwrap();
    let field = Array3::from_elem((10, 5, 5), 0.0_f64);
    let mut g = Array3::zeros([10, 5, 5]); // wrong: should be [9, 5, 5]
    assert!(op.apply_x_into(field.view(), &mut g.view_mut()).is_err());
}

#[test]
fn central4_rejects_dst_shape_mismatch() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
    let field = Array3::from_elem((10, 5, 5), 0.0_f64);
    let mut g = Array3::zeros([9, 5, 5]); // wrong: should be [10, 5, 5]
    assert!(op.apply_x_into(field.view(), &mut g.view_mut()).is_err());
}

// ── Dispersion ordering & spacing rejection (regression suite) ────────────

#[test]
fn dispersion_ordering_central_2_4_6() {
    let dx = 0.1;
    let lambda = 2.0;
    let k = 2.0 * std::f64::consts::PI / lambda;
    let n = 80_usize;
    let mut field = Array3::zeros([n, 5, 5]);
    for i in 0..n {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for kk in 0..5 {
                field[[i, j, kk]] = (k * x).sin();
            }
        }
    }
    let op2 = FiniteDifference3D::<f64>::central_second_order(dx, dx, dx).unwrap();
    let op4 = FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx).unwrap();
    let op6 = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();

    let mut g2 = Array3::zeros([n, 5, 5]);
    let mut g4 = Array3::zeros([n, 5, 5]);
    let mut g6 = Array3::zeros([n, 5, 5]);
    op2.apply_x_into(field.view(), &mut g2.view_mut()).unwrap();
    op4.apply_x_into(field.view(), &mut g4.view_mut()).unwrap();
    op6.apply_x_into(field.view(), &mut g6.view_mut()).unwrap();

    let mut err2 = 0.0;
    let mut err4 = 0.0;
    let mut err6 = 0.0;
    let mut count = 0_usize;
    for i in 10..(n - 10) {
        let x = (i as f64) * dx;
        let exact = k * (k * x).cos();
        err2 += (g2[[i, 2, 2]] - exact).abs();
        err4 += (g4[[i, 2, 2]] - exact).abs();
        err6 += (g6[[i, 2, 2]] - exact).abs();
        count += 1;
    }
    let count = count as f64;
    err2 /= count;
    err4 /= count;
    err6 /= count;
    assert!(
        err4 < err2,
        "4th-order should be more accurate than 2nd-order: err4={err4}, err2={err2}"
    );
    assert!(
        err6 < err4,
        "6th-order should be more accurate than 4th-order: err6={err6}, err4={err4}"
    );
}

#[test]
fn rejects_non_positive_spacing() {
    assert!(FiniteDifference3D::<f64>::central_second_order(0.0, 0.1, 0.1).is_err());
    assert!(FiniteDifference3D::<f64>::central_fourth_order(0.1, -0.1, 0.1).is_err());
    assert!(FiniteDifference3D::<f64>::central_sixth_order(0.1, 0.1, 0.0).is_err());
    assert!(FiniteDifference3D::<f64>::staggered_forward(0.0, 0.1, 0.1).is_err());
    assert!(FiniteDifference3D::<f64>::staggered_backward(0.1, 0.1, 0.0).is_err());
}

#[test]
fn rejects_too_few_points() {
    let dx = 0.1;
    let op = FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx).unwrap();
    let small = Array3::zeros([6, 10, 10]);
    let mut g = Array3::zeros([6, 10, 10]);
    assert!(op.apply_x_into(small.view(), &mut g.view_mut()).is_err());
}

#[test]
fn stencil_width_matches_scheme() {
    let dx = 0.1;
    assert_eq!(
        FiniteDifference3D::<f64>::central_second_order(dx, dx, dx)
            .unwrap()
            .stencil_width(),
        3
    );
    assert_eq!(
        FiniteDifference3D::<f64>::central_fourth_order(dx, dx, dx)
            .unwrap()
            .stencil_width(),
        5
    );
    assert_eq!(
        FiniteDifference3D::<f64>::central_sixth_order(dx, dx, dx)
            .unwrap()
            .stencil_width(),
        7
    );
    assert_eq!(
        FiniteDifference3D::<f64>::staggered_forward(dx, dx, dx)
            .unwrap()
            .stencil_width(),
        2
    );
    assert_eq!(
        FiniteDifference3D::<f64>::staggered_backward(dx, dx, dx)
            .unwrap()
            .stencil_width(),
        2
    );
}

/// The destination is a mutable view, not a Leto-owned array, so a caller
/// whose storage comes from somewhere else — a device buffer's host-addressable
/// slice, an arena, an FFI allocation — writes into it directly.
///
/// This is the contract the Coeus backend seam needs: its CPU kernels receive
/// `&mut [T]` out of a `DeviceBuffer`, and an owned-array parameter would force
/// an allocation and a copy per sweep.
#[test]
fn writes_through_a_view_over_a_foreign_slice() {
    use leto::{ArrayViewMut3, Layout};

    let shape = [6usize, 5, 7];
    let count = shape[0] * shape[1] * shape[2];
    let mut field = Array3::<f64>::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                field[[i, j, k]] = (i * 100 + j * 10 + k) as f64 * 0.25;
            }
        }
    }
    let op = FiniteDifference3D::central_fourth_order(0.5, 0.5, 0.5).unwrap();

    // The owned path, for reference.
    let mut owned = Array3::<f64>::zeros(shape);
    op.apply_y_into(field.view(), &mut owned.view_mut())
        .unwrap();

    // The same operator writing into a plain slice this crate does not own.
    let mut foreign = vec![f64::NAN; count];
    let strides = [(shape[1] * shape[2]) as isize, shape[2] as isize, 1_isize];
    let layout = Layout::<3>::try_new(shape, strides, 0).unwrap();
    let mut view = ArrayViewMut3::try_new(layout, foreign.as_mut_slice()).unwrap();
    op.apply_y_into(field.view(), &mut view).unwrap();

    // Bitwise: the same kernel ran over the same values in the same order.
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let index = (i * shape[1] + j) * shape[2] + k;
                assert_eq!(foreign[index], owned[[i, j, k]], "({i}, {j}, {k})");
            }
        }
    }
}

// ── Adjoint identity: ⟨Af, u⟩ = ⟨f, Aᵀu⟩ ────────────────────────────────────

const ADJOINT_SCHEMES: [FiniteDifference3DScheme; 5] = [
    FiniteDifference3DScheme::CentralSecondOrder,
    FiniteDifference3DScheme::CentralFourthOrder,
    FiniteDifference3DScheme::CentralSixthOrder,
    FiniteDifference3DScheme::StaggeredForward,
    FiniteDifference3DScheme::StaggeredBackward,
];

const ADJOINT_AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// Deterministic non-separable lane values: no symmetry for a sign error to
/// hide behind.
fn lane_value(index: usize, seed: f64) -> f64 {
    (index as f64 * 0.37 + seed).sin() * 1.7 + (index as f64 * 0.11 + seed * 0.3).cos() * 0.9 - 0.4
}

fn adjoint_identity_case(scheme: FiniteDifference3DScheme, axis: Axis, extent: usize) {
    let h = 0.5_f64;
    let op = FiniteDifference3D::<f64>::new(scheme, h, h, h).unwrap();
    let mut shape = [3_usize, 2, 4];
    shape[axis.index()] = extent;
    let mut out_shape = shape;
    if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
        out_shape[axis.index()] -= 1;
    }
    let mut f = Array3::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let c = [i, j, k][axis.index()];
                f[[i, j, k]] = lane_value(c + 7 * (i + j + k), 0.23);
            }
        }
    }
    let mut u = Array3::zeros(out_shape);
    for i in 0..out_shape[0] {
        for j in 0..out_shape[1] {
            for k in 0..out_shape[2] {
                let c = [i, j, k][axis.index()];
                u[[i, j, k]] = lane_value(c + 3 * (i + 2 * j + 5 * k), 1.71);
            }
        }
    }
    let mut af = Array3::zeros(out_shape);
    let mut atu = Array3::zeros(shape);
    match axis {
        Axis::X => {
            op.apply_x_into(f.view(), &mut af.view_mut()).unwrap();
            op.adjoint_x_into(u.view(), &mut atu.view_mut()).unwrap();
        }
        Axis::Y => {
            op.apply_y_into(f.view(), &mut af.view_mut()).unwrap();
            op.adjoint_y_into(u.view(), &mut atu.view_mut()).unwrap();
        }
        Axis::Z => {
            op.apply_z_into(f.view(), &mut af.view_mut()).unwrap();
            op.adjoint_z_into(u.view(), &mut atu.view_mut()).unwrap();
        }
    }
    let mut left = 0.0_f64;
    for i in 0..out_shape[0] {
        for j in 0..out_shape[1] {
            for k in 0..out_shape[2] {
                left += af[[i, j, k]] * u[[i, j, k]];
            }
        }
    }
    let mut right = 0.0_f64;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                right += f[[i, j, k]] * atu[[i, j, k]];
            }
        }
    }
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= 1e-9 * scale,
        "{scheme:?} on {axis:?} over {shape:?}: <Af,u>={left:e} vs <f,A^Tu>={right:e}"
    );
}

#[test]
fn adjoint_satisfies_the_transpose_identity_on_every_scheme_axis_and_extent() {
    for scheme in ADJOINT_SCHEMES {
        let minimum = match scheme {
            FiniteDifference3DScheme::CentralSecondOrder => 3,
            FiniteDifference3DScheme::CentralFourthOrder => 1,
            FiniteDifference3DScheme::CentralSixthOrder => 7,
            FiniteDifference3DScheme::StaggeredForward
            | FiniteDifference3DScheme::StaggeredBackward => 2,
        };
        for axis in ADJOINT_AXES {
            // The minimum, where every fall-back branch is live, plus a run
            // of wider axes covering each position class of the adjoint.
            for extent in minimum..minimum + 6 {
                adjoint_identity_case(scheme, axis, extent);
            }
        }
    }
}

/// The identity is necessary and sufficient for random vectors, but a direct
/// dense-transpose cross-check on small grids pins the derivation lane by
/// lane: build `A` column by column from basis sweeps, then compare `Aᵀu`.
#[test]
fn adjoint_matches_the_dense_transpose_on_small_grids() {
    for scheme in ADJOINT_SCHEMES {
        let minimum = match scheme {
            FiniteDifference3DScheme::CentralSecondOrder => 3,
            FiniteDifference3DScheme::CentralSixthOrder => 7,
            FiniteDifference3DScheme::CentralFourthOrder
            | FiniteDifference3DScheme::StaggeredForward
            | FiniteDifference3DScheme::StaggeredBackward => 2,
        };
        for extent in [minimum, minimum + 1] {
            let h = 0.5_f64;
            let op = FiniteDifference3D::<f64>::new(scheme, h, h, h).unwrap();
            let m = if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
                extent - 1
            } else {
                extent
            };
            // Columns of A: sweep each basis vector along a one-line field.
            let mut a = vec![vec![0.0_f64; extent]; m];
            for column in 0..extent {
                let mut f = Array3::zeros([extent, 1, 1]);
                f[[column, 0, 0]] = 1.0;
                let mut col = Array3::zeros([m, 1, 1]);
                op.apply_x_into(f.view(), &mut col.view_mut()).unwrap();
                for row in 0..m {
                    a[row][column] = col[[row, 0, 0]];
                }
            }
            let mut u = Array3::zeros([m, 1, 1]);
            for row in 0..m {
                u[[row, 0, 0]] = lane_value(row, 2.97);
            }
            let mut atu = Array3::zeros([extent, 1, 1]);
            op.adjoint_x_into(u.view(), &mut atu.view_mut()).unwrap();
            for j in 0..extent {
                let expected: f64 = (0..m).map(|i| a[i][j] * u[[i, 0, 0]]).sum();
                assert!(
                    (atu[[j, 0, 0]] - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "{scheme:?} over {extent}: lane {j}: {} vs dense {}",
                    atu[[j, 0, 0]],
                    expected
                );
            }
        }
    }
}

#[test]
fn adjoint_rejects_mismatched_and_thin_grids() {
    let op = FiniteDifference3D::<f64>::central_sixth_order(0.5, 0.5, 0.5).unwrap();
    // Six points cannot carry a sixth-order sweep, forward or adjoint.
    let thin = Array3::zeros([6, 2, 2]);
    let mut grad = Array3::zeros([6, 2, 2]);
    assert!(op
        .adjoint_x_into(thin.view(), &mut grad.view_mut())
        .is_err());
    // The upstream must have the forward sweep's output shape.
    let full = Array3::zeros([8, 2, 2]);
    let mut grad = Array3::zeros([8, 2, 2]);
    assert!(op.adjoint_x_into(full.view(), &mut grad.view_mut()).is_ok());
    let short = Array3::zeros([7, 2, 2]);
    assert!(op
        .adjoint_x_into(short.view(), &mut grad.view_mut())
        .is_err());

    // A forward sweep shrinks the grid, so its adjoint fans back out: the
    // upstream is one lane short, the gradient is full.
    let fwd = FiniteDifference3D::<f64>::staggered_forward(0.5, 0.5, 0.5).unwrap();
    let upstream = Array3::zeros([7, 2, 2]);
    let mut grad = Array3::zeros([8, 2, 2]);
    assert!(fwd
        .adjoint_x_into(upstream.view(), &mut grad.view_mut())
        .is_ok());
    let full_upstream = Array3::zeros([8, 2, 2]);
    assert!(fwd
        .adjoint_x_into(full_upstream.view(), &mut grad.view_mut())
        .is_err());

    // A singleton fourth-order axis is flat in both directions.
    let fourth = FiniteDifference3D::<f64>::central_fourth_order(0.5, 0.5, 0.5).unwrap();
    let upstream = Array3::zeros([1, 2, 2]);
    let mut grad = Array3::zeros([1, 2, 2]);
    fourth
        .adjoint_x_into(upstream.view(), &mut grad.view_mut())
        .unwrap();
    assert_eq!(grad[[0, 0, 0]], 0.0);
    assert_eq!(grad[[0, 1, 1]], 0.0);
}
