#![expect(
    clippy::unwrap_used,
    reason = "test code: unwraps on fixed fixtures are the floor's test exemption"
)]

use super::super::*;
use leto::Array3;

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
