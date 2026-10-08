#![expect(
    clippy::unwrap_used,
    reason = "test code: unwraps on fixed fixtures are the floor's test exemption"
)]

use super::super::*;
use leto::Array3;

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
