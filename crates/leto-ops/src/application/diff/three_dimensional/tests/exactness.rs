#![expect(
    clippy::unwrap_used,
    reason = "test code: unwraps on fixed fixtures are the floor's test exemption"
)]

use super::super::*;
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
