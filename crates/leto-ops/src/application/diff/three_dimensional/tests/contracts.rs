#![expect(
    clippy::unwrap_used,
    reason = "test code: unwraps on fixed fixtures are the floor's test exemption"
)]

use super::super::*;
use leto::Array3;

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
