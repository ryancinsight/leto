//! Operator construction and second-order reduction checks.

use super::*;

// ── Construction ─────────────────────────────────────────────────────────────

#[test]
fn rejects_invalid_orders_and_spacings() {
    assert!(StaggeredLeapfrog3D::<f64>::new(0, 1.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(3, 1.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2 * MAX_HALF_ORDER + 2, 1.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 0.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 1.0, -1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 1.0, 1.0, 0.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, f64::NAN, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 1.0, f64::INFINITY, 1.0).is_err());
    let op = StaggeredLeapfrog3D::<f64>::new(8, 1e-3, 2e-3, 3e-3).unwrap();
    assert_eq!(op.order(), 8);
    assert_eq!(op.halo_width(), 4);
    assert_eq!(op.spacing(), (1e-3, 2e-3, 3e-3));
}

/// Rebuilding from the operator's own taps and reciprocal spacings sweeps
/// bitwise identically, at a spacing (0.3) whose reciprocal is inexact.
#[test]
fn from_parts_reproduces_the_derived_operator_bitwise() {
    let op = StaggeredLeapfrog3D::<f32>::new(6, 0.3, 0.7, 1.1).unwrap();
    let (dx, dy, dz) = op.spacing();
    let rebuilt = StaggeredLeapfrog3D::from_parts(
        TapCoefficients::from_taps(op.coefficients().taps()).unwrap(),
        [dx.recip(), dy.recip(), dz.recip()],
    )
    .unwrap();
    let shape = [7, 6, 8];
    let field = seeded(shape, 0.37).mapv(|value| value as f32);
    for axis in AXES {
        let mut expected = Array3::<f32>::zeros(shape);
        let mut got = Array3::<f32>::zeros(shape);
        op.gradient_into(axis, field.view(), &mut expected.view_mut())
            .unwrap();
        rebuilt
            .gradient_into(axis, field.view(), &mut got.view_mut())
            .unwrap();
        assert_eq!(got, expected, "gradient along {axis:?}");
        op.divergence_into(axis, field.view(), &mut expected.view_mut())
            .unwrap();
        rebuilt
            .divergence_into(axis, field.view(), &mut got.view_mut())
            .unwrap();
        assert_eq!(got, expected, "divergence along {axis:?}");
    }
}

#[test]
fn from_parts_and_from_taps_reject_invalid_input() {
    let taps = staggered_first_derivative_coefficients::<f64>(2).unwrap();
    assert!(StaggeredLeapfrog3D::from_parts(taps, [1.0, 0.0, 1.0]).is_err());
    assert!(StaggeredLeapfrog3D::from_parts(taps, [1.0, 1.0, f64::NAN]).is_err());
    assert!(TapCoefficients::<f64>::from_taps(&[]).is_err());
    assert!(TapCoefficients::<f64>::from_taps(&[1.0; MAX_HALF_ORDER + 1]).is_err());
    assert!(TapCoefficients::from_taps(&[1.0, f64::INFINITY]).is_err());
    let adopted = TapCoefficients::from_taps(&[1.125, -1.0 / 24.0]).unwrap();
    assert_eq!(adopted.taps(), &[1.125, -1.0 / 24.0]);
    assert_eq!(adopted.order(), 4);
}

#[test]
fn a_shape_mismatch_is_reported_not_asserted() {
    let op = StaggeredLeapfrog3D::<f64>::new(2, 1.0, 1.0, 1.0).unwrap();
    let field = Array3::<f64>::zeros([4, 4, 4]);
    let mut dst = Array3::<f64>::zeros([4, 4, 3]);
    assert!(op
        .gradient_into(Axis::X, field.view(), &mut dst.view_mut())
        .is_err());
    assert!(op
        .divergence_into(Axis::X, field.view(), &mut dst.view_mut())
        .is_err());
}

// ── Value semantics ──────────────────────────────────────────────────────────

#[test]
fn second_order_reduces_to_the_plain_half_grid_difference() {
    let shape = [5, 4, 6];
    let field = seeded(shape, 0.3);
    let op = StaggeredLeapfrog3D::<f64>::new(2, 1.0, 1.0, 1.0).unwrap();
    let mut dst = Array3::zeros(shape);
    op.gradient_into(Axis::Z, field.view(), &mut dst.view_mut())
        .unwrap();

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            // Interior faces are the plain forward difference.
            for k in 0..shape[2] - 1 {
                let expected = field[[i, j, k + 1]] - field[[i, j, k]];
                assert_eq!(dst[[i, j, k]], expected);
            }
            // The far face is the reflected wall: the tap mirrors onto itself.
            assert_eq!(dst[[i, j, shape[2] - 1]], 0.0);
        }
    }
}

#[test]
fn second_order_interior_agrees_with_the_fixed_staggered_forward_kernel() {
    // `FiniteDifference3D::staggered_forward` writes one cell fewer on the
    // differentiated axis and imposes no wall closure; the leapfrog pair is
    // grid-shaped and reflects. They must agree wherever both are defined.
    let shape = [6, 5, 7];
    let field = seeded(shape, 1.1);
    let dx = 2.5e-4;
    let leapfrog = StaggeredLeapfrog3D::<f64>::new(2, dx, dx, dx).unwrap();
    let fixed =
        FiniteDifference3D::<f64>::new(FiniteDifference3DScheme::StaggeredForward, dx, dx, dx)
            .unwrap();

    let mut from_leapfrog = Array3::zeros(shape);
    leapfrog
        .gradient_into(Axis::X, field.view(), &mut from_leapfrog.view_mut())
        .unwrap();
    let mut from_fixed = Array3::zeros([shape[0] - 1, shape[1], shape[2]]);
    fixed
        .apply_x_into(field.view(), &mut from_fixed.view_mut())
        .unwrap();

    for i in 0..shape[0] - 1 {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                assert_eq!(
                    from_leapfrog[[i, j, k]],
                    from_fixed[[i, j, k]],
                    "face ({i}, {j}, {k})"
                );
            }
        }
    }
}
