#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, LetoError, SliceArg, Storage, VecStorage};
use leto_ops::{ray_line_integrals, ray_line_integrals_into};

// 9x5x5 field, spacing 2: an 18x10x10 world-unit volume at the origin,
// exactly the conformance geometry.
const ORIGIN: [f32; 3] = [0.0, 0.0, 0.0];
const SPACING: [f32; 3] = [2.0, 2.0, 2.0];

// Absolute bound from the deepest march in these clauses (same derivation as
// the device contract): at most 16 / 0.125 = 128 terms of magnitude ~0.5, so
// naive f32 summation error stays well under 1e-4. The miss oracle is exact.
const SUM_BOUND: f32 = 1e-4;

fn build_field(f: impl Fn(u32, u32, u32) -> f32) -> Array<f32, VecStorage<f32>, 3> {
    let mut host = Vec::new();
    for ix in 0..9 {
        for iy in 0..5 {
            for iz in 0..5 {
                host.push(f(ix, iy, iz));
            }
        }
    }
    Array::from_shape_vec([9, 5, 5], host).unwrap()
}

fn hit_and_miss_rays() -> Array<f32, VecStorage<f32>, 2> {
    Array::from_shape_vec(
        [2, 6],
        vec![
            -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, // hit
            -10.0, 100.0, 4.0, 1.0, 0.0, 0.0, // miss
        ],
    )
    .unwrap()
}

/// Uniform field: a +x ray through the middle has chord 16, so the integral
/// is 0.25 * 16 = 4; a ray far outside in y never samples the volume and
/// stays exactly zero.
#[test]
fn uniform_chord_matches_analytical_oracle() {
    let field = build_field(|_, _, _| 0.25);
    let rays = hit_and_miss_rays();
    let out = ray_line_integrals(&field.view(), ORIGIN, SPACING, &rays.view(), 0.5).unwrap();
    let got = out.storage().as_slice();
    assert!(
        (got[0] - 4.0).abs() < SUM_BOUND,
        "uniform chord integral {} != 4.0",
        got[0]
    );
    assert_eq!(got[1], 0.0, "a missing ray must integrate to 0");
}

/// Affine field f(x) = 0.01*ix + 0.02 along the chord: the midpoint rule is
/// exact for affine integrands, so only summation error remains.
#[test]
fn affine_field_matches_analytical_oracle() {
    let field = build_field(|ix, _, _| 0.01 * ix as f32 + 0.02);
    let rays = Array::from_shape_vec([1, 6], vec![-10.0, 4.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
    let out = ray_line_integrals(&field.view(), ORIGIN, SPACING, &rays.view(), 1.0).unwrap();
    let got = out.storage().as_slice();
    assert!(
        (got[0] - 0.96).abs() < SUM_BOUND,
        "affine midpoint integral {} != 0.96",
        got[0]
    );
}

/// A uniform integral is independent of the step size.
#[test]
fn uniform_integral_is_step_independent() {
    let field = build_field(|_, _, _| 0.25);
    let rays = Array::from_shape_vec([1, 6], vec![-10.0, 4.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
    let coarse = ray_line_integrals(&field.view(), ORIGIN, SPACING, &rays.view(), 8.0).unwrap();
    let fine = ray_line_integrals(&field.view(), ORIGIN, SPACING, &rays.view(), 0.125).unwrap();
    let (coarse, fine) = (coarse.storage().as_slice()[0], fine.storage().as_slice()[0]);
    assert!(
        (coarse - fine).abs() < SUM_BOUND,
        "step dependence: {coarse} vs {fine}"
    );
}

/// An output shorter than the ray count is rejected and untouched.
#[test]
fn short_output_is_rejected_untouched() {
    let field = build_field(|_, _, _| 0.25);
    let rays = Array::from_shape_vec(
        [2, 6],
        vec![
            -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, //
            -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, //
        ],
    )
    .unwrap();
    let mut short = Array::from_shape_vec([1], vec![9.0f32]).unwrap();
    let err = ray_line_integrals_into(
        &field.view(),
        ORIGIN,
        SPACING,
        &rays.view(),
        0.5,
        &mut short.view_mut(),
    )
    .expect_err("output shorter than the ray count must be rejected");
    assert_eq!(
        err,
        LetoError::ShapeMismatch {
            lhs: vec![2],
            rhs: vec![1],
        }
    );
    assert_eq!(
        short.storage().as_slice(),
        &[9.0],
        "a rejected dispatch must not touch the output"
    );
}

#[test]
fn invalid_parameters_are_typed_rejections() {
    let field = build_field(|_, _, _| 0.25);
    let rays = Array::from_shape_vec([1, 6], vec![-10.0, 4.0, 4.0, 1.0, 0.0, 0.0]).unwrap();
    let layout = Layout::c_contiguous([1]).unwrap();
    let mut out = Array::new(layout, VecStorage::fill(1, 0.0)).unwrap();
    for step in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            ray_line_integrals_into(
                &field.view(),
                ORIGIN,
                SPACING,
                &rays.view(),
                step,
                &mut out.view_mut()
            )
            .is_err(),
            "step {step} must be rejected"
        );
    }
    for spacing in [[0.0, 2.0, 2.0], [2.0, -2.0, 2.0], [f32::NAN, 2.0, 2.0]] {
        assert!(
            ray_line_integrals_into(
                &field.view(),
                ORIGIN,
                spacing,
                &rays.view(),
                0.5,
                &mut out.view_mut()
            )
            .is_err(),
            "spacing {spacing:?} must be rejected"
        );
    }
    assert!(
        ray_line_integrals_into(
            &field.view(),
            [f32::INFINITY, 0.0, 0.0],
            SPACING,
            &rays.view(),
            0.5,
            &mut out.view_mut()
        )
        .is_err(),
        "non-finite origin must be rejected"
    );
    let bad_rays = Array::from_shape_vec([1, 5], vec![0.0f32; 5]).unwrap();
    assert_eq!(
        ray_line_integrals_into(
            &field.view(),
            ORIGIN,
            SPACING,
            &bad_rays.view(),
            0.5,
            &mut out.view_mut()
        )
        .expect_err("ray stride must be 6"),
        LetoError::ShapeMismatch {
            lhs: vec![1, 6],
            rhs: vec![1, 5],
        }
    );
}

#[test]
fn march_serves_strided_views() {
    let field = build_field(|_, _, _| 0.25);
    // Even rows carry hit/miss; odd rows are decoys that must never march.
    let rays = Array::from_shape_vec(
        [4, 6],
        vec![
            -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            -10.0, 100.0, 4.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        ],
    )
    .unwrap();
    let stepped = rays
        .view()
        .slice_with::<2>(&[SliceArg::range(Some(0), None, 2), SliceArg::All])
        .unwrap();
    assert_eq!(stepped.shape(), [2, 6]);
    let mut backing = Array::from_shape_vec([4], vec![9.0f32; 4]).unwrap();
    let mut out = backing
        .slice_with_mut::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();
    ray_line_integrals_into(&field.view(), ORIGIN, SPACING, &stepped, 0.5, &mut out).unwrap();
    let got = backing.storage().as_slice();
    assert!(
        (got[0] - 4.0).abs() < SUM_BOUND,
        "strided uniform chord integral {} != 4.0",
        got[0]
    );
    assert_eq!(got[1], 9.0, "output gaps must stay untouched");
    assert_eq!(got[2], 0.0, "a missing ray must integrate to 0");
    assert_eq!(got[3], 9.0, "output gaps must stay untouched");
}
