//! Contract tests for the arbitrary-order staggered pair.
//!
//! The oracles are analytical throughout: published stencil coefficients, the
//! measured order of accuracy against an exactly differentiable field, the
//! adjoint identity that makes the leapfrog conservative, and the wall closure
//! the reflection is supposed to impose.

use leto::Array3;

use super::super::coefficients::{
    central_first_derivative_coefficients, staggered_first_derivative_coefficients, MAX_HALF_ORDER,
};
use super::super::{FiniteDifference3D, FiniteDifference3DScheme};
use super::{Axis, StaggeredLeapfrog3D};

const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// Deterministic, non-separable test field: no axis is a constant multiple of
/// another, so an adjointness failure on one axis cannot hide behind another.
fn seeded(shape: [usize; 3], salt: f64) -> Array3<f64> {
    let mut field = Array3::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let x = i as f64 * 0.37 + salt;
                let y = j as f64 * 0.53 - salt * 0.5;
                let z = k as f64 * 0.71 + salt * 0.25;
                field[[i, j, k]] = (x.sin() * y.cos() + z.sin() * 0.75) * (1.0 + 0.1 * salt);
            }
        }
    }
    field
}

fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.as_slice()
        .unwrap()
        .iter()
        .zip(b.as_slice().unwrap())
        .fold(0.0, |sum, (&a, &b)| sum + a * b)
}

// ── Coefficients ─────────────────────────────────────────────────────────────

#[test]
fn staggered_coefficients_match_the_published_rationals() {
    // Orders 2, 4, 6, 8 from Fornberg (1988) Table 1 / Levander (1988).
    let published: [&[f64]; 4] = [
        &[1.0],
        &[9.0 / 8.0, -1.0 / 24.0],
        &[75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0],
        &[
            1225.0 / 1024.0,
            -245.0 / 3072.0,
            49.0 / 5120.0,
            -5.0 / 7168.0,
        ],
    ];
    for (index, expected) in published.iter().enumerate() {
        let half_order = index + 1;
        let derived = staggered_first_derivative_coefficients::<f64>(half_order).unwrap();
        assert_eq!(derived.half_order(), half_order);
        assert_eq!(derived.order(), 2 * half_order);
        for (&derived, &expected) in derived.taps().iter().zip(expected.iter()) {
            let relative = (derived - expected).abs() / expected.abs();
            assert!(
                relative < 1e-13,
                "half-order {half_order}: derived {derived} vs published {expected} \
                 (relative {relative:e})"
            );
        }
    }
}

#[test]
fn collocated_coefficients_match_the_published_rationals() {
    let published: [&[f64]; 3] = [
        &[0.5],
        &[2.0 / 3.0, -1.0 / 12.0],
        &[3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0],
    ];
    for (index, expected) in published.iter().enumerate() {
        let derived = central_first_derivative_coefficients::<f64>(index + 1).unwrap();
        for (&derived, &expected) in derived.taps().iter().zip(expected.iter()) {
            let relative = (derived - expected).abs() / expected.abs();
            assert!(
                relative < 1e-13,
                "derived {derived} vs published {expected}"
            );
        }
    }
}

#[test]
fn coefficients_reject_orders_outside_the_verified_range() {
    assert!(staggered_first_derivative_coefficients::<f64>(0).is_err());
    assert!(staggered_first_derivative_coefficients::<f64>(MAX_HALF_ORDER + 1).is_err());
    assert!(central_first_derivative_coefficients::<f64>(0).is_err());
    assert!(central_first_derivative_coefficients::<f64>(MAX_HALF_ORDER + 1).is_err());
    // Every order inside the range derives.
    for half_order in 1..=MAX_HALF_ORDER {
        assert!(staggered_first_derivative_coefficients::<f64>(half_order).is_ok());
        assert!(central_first_derivative_coefficients::<f64>(half_order).is_ok());
    }
}

// ── Construction ─────────────────────────────────────────────────────────────

#[test]
fn rejects_invalid_orders_and_spacings() {
    assert!(StaggeredLeapfrog3D::<f64>::new(0, 1.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(3, 1.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2 * MAX_HALF_ORDER + 2, 1.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 0.0, 1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 1.0, -1.0, 1.0).is_err());
    assert!(StaggeredLeapfrog3D::<f64>::new(2, 1.0, 1.0, 0.0).is_err());
    let op = StaggeredLeapfrog3D::<f64>::new(8, 1e-3, 2e-3, 3e-3).unwrap();
    assert_eq!(op.order(), 8);
    assert_eq!(op.halo_width(), 4);
    assert_eq!(op.spacing(), (1e-3, 2e-3, 3e-3));
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

#[test]
fn a_uniform_field_has_no_gradient_on_any_axis_or_order() {
    let shape = [7, 6, 8];
    let field = Array3::from_elem(shape, -3.25_f64);
    for order in [2, 4, 6, 8] {
        let op = StaggeredLeapfrog3D::<f64>::new(order, 1e-3, 1e-3, 1e-3).unwrap();
        for axis in AXES {
            let mut dst = Array3::zeros(shape);
            op.gradient_into(axis, field.view(), &mut dst.view_mut())
                .unwrap();
            for &value in dst.as_slice().unwrap() {
                assert_eq!(value, 0.0, "order {order} axis {axis:?}");
            }
        }
    }
}

#[test]
fn the_far_wall_is_rigid_without_being_forced() {
    // Reflection is what imposes the zero-normal-derivative wall; nothing
    // clamps the boundary cell afterwards.
    let shape = [9, 3, 3];
    let field = seeded(shape, 2.0);
    let op = StaggeredLeapfrog3D::<f64>::new(4, 1.0, 1.0, 1.0).unwrap();
    let mut dst = Array3::zeros(shape);
    op.gradient_into(Axis::X, field.view(), &mut dst.view_mut())
        .unwrap();
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            assert_eq!(dst[[shape[0] - 1, j, k]], 0.0);
        }
    }
}

// ── Order of accuracy ────────────────────────────────────────────────────────

#[test]
fn each_order_converges_at_its_claimed_rate() {
    // On a sinusoid of fixed wavelength, refining the grid must reduce the
    // face-derivative error at the nominal rate. Measured on the interior only:
    // the wall closure is first order by construction and is not what this
    // test is about.
    fn error_at(order: usize, points_per_wavelength: f64) -> f64 {
        let n = points_per_wavelength as usize;
        let shape = [n, 1, 1];
        let dx = 1.0 / points_per_wavelength;
        let k = std::f64::consts::TAU;
        let mut field = Array3::zeros(shape);
        for i in 0..n {
            field[[i, 0, 0]] = (k * i as f64 * dx).sin();
        }
        let op = StaggeredLeapfrog3D::<f64>::new(order, dx, dx, dx).unwrap();
        let mut dst = Array3::zeros(shape);
        op.gradient_into(Axis::X, field.view(), &mut dst.view_mut())
            .unwrap();

        let halo = op.halo_width();
        let mut worst: f64 = 0.0;
        for i in halo..n - halo - 1 {
            // The face i+1/2 sits at (i + 1/2) * dx.
            let exact = k * (k * (i as f64 + 0.5) * dx).cos();
            worst = worst.max((dst[[i, 0, 0]] - exact).abs());
        }
        worst
    }

    for order in [2, 4, 6, 8] {
        let coarse = error_at(order, 32.0);
        let fine = error_at(order, 64.0);
        let measured = (coarse / fine).log2();
        assert!(
            (measured - order as f64).abs() < 0.35,
            "order {order}: measured rate {measured:.3} from {coarse:e} -> {fine:e}"
        );
    }
}

// ── The adjoint identity ─────────────────────────────────────────────────────

#[test]
fn gradient_and_divergence_are_negative_adjoints() {
    // <G p, u> = -<p, D u> is the identity a conservative leapfrog rests on.
    let shape = [6, 5, 7];
    for order in [2, 4, 6, 8] {
        let op = StaggeredLeapfrog3D::<f64>::new(order, 1.3e-3, 0.7e-3, 2.1e-3).unwrap();
        for (index, axis) in AXES.into_iter().enumerate() {
            let p = seeded(shape, 0.4 + index as f64);
            let u = seeded(shape, 1.9 - index as f64);

            let mut grad_p = Array3::zeros(shape);
            op.gradient_into(axis, p.view(), &mut grad_p.view_mut())
                .unwrap();
            let mut div_u = Array3::zeros(shape);
            op.divergence_into(axis, u.view(), &mut div_u.view_mut())
                .unwrap();

            let left = dot(&grad_p, &u);
            let right = -dot(&p, &div_u);
            // Both sides are sums of the same products in different orders, so
            // the tolerance is the accumulated rounding of a length-N sum:
            // O(N eps) relative, with N the cell count.
            let scale = left.abs().max(right.abs());
            let bound = 64.0 * f64::EPSILON * scale * shape.iter().product::<usize>() as f64;
            assert!(
                (left - right).abs() <= bound,
                "order {order} axis {axis:?}: {left:e} vs {right:e} (bound {bound:e})"
            );
        }
    }
}

#[test]
fn the_adjointness_fields_are_non_degenerate() {
    // Guards the test above: a pair of fields whose inner products are near
    // zero would satisfy the identity trivially.
    let shape = [6, 5, 7];
    let p = seeded(shape, 0.4);
    let u = seeded(shape, 1.9);
    assert!(dot(&p, &p) > 1.0);
    assert!(dot(&u, &u) > 1.0);
    let op = StaggeredLeapfrog3D::<f64>::new(4, 1.0, 1.0, 1.0).unwrap();
    let mut grad_p = Array3::zeros(shape);
    op.gradient_into(Axis::Y, p.view(), &mut grad_p.view_mut())
        .unwrap();
    assert!(dot(&grad_p, &u).abs() > 1e-3);
}

// ── Traversal agreement ──────────────────────────────────────────────────────

#[test]
fn the_block_kernels_agree_with_the_contiguous_one() {
    // The three axes take three different traversals through the same
    // coefficients; a cubic grid makes their results directly comparable after
    // transposing the field.
    let n = 6;
    let shape = [n, n, n];
    let field = seeded(shape, 0.9);
    let op = StaggeredLeapfrog3D::<f64>::new(6, 1.0, 1.0, 1.0).unwrap();

    let mut along_z = Array3::zeros(shape);
    op.gradient_into(Axis::Z, field.view(), &mut along_z.view_mut())
        .unwrap();

    // Transpose x <-> z, differentiate along x, transpose back.
    let mut transposed = Array3::zeros(shape);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                transposed[[k, j, i]] = field[[i, j, k]];
            }
        }
    }
    let mut along_x = Array3::zeros(shape);
    op.gradient_into(Axis::X, transposed.view(), &mut along_x.view_mut())
        .unwrap();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert_eq!(along_z[[i, j, k]], along_x[[k, j, i]], "({i}, {j}, {k})");
            }
        }
    }
}

// ── The Courant limit ────────────────────────────────────────────────────────

#[test]
fn cfl_limit_matches_its_derivation() {
    // At order 2 the tap sum is 1, so the limit is the familiar 1/sqrt(D).
    let second = StaggeredLeapfrog3D::<f64>::new(2, 1.0, 1.0, 1.0).unwrap();
    for dimensions in 1..=3 {
        let expected = 1.0 / (dimensions as f64).sqrt();
        assert!((second.cfl_limit(dimensions) - expected).abs() < 1e-15);
    }
    // Higher orders shrink it by exactly the tap sum, and stay above the
    // collocated limit they are often confused with.
    for order in [4, 6, 8] {
        let op = StaggeredLeapfrog3D::<f64>::new(order, 1.0, 1.0, 1.0).unwrap();
        let sum: f64 = op.coefficients().taps().iter().map(|c| c.abs()).sum();
        let expected = 1.0 / (3.0_f64.sqrt() * sum);
        assert!((op.cfl_limit(3) - expected).abs() < 1e-15);
        assert!(op.cfl_limit(3) < second.cfl_limit(3));
    }
    let fourth = StaggeredLeapfrog3D::<f64>::new(4, 1.0, 1.0, 1.0).unwrap();
    assert!(
        fourth.cfl_limit(3) > 0.49 && fourth.cfl_limit(3) < 0.50,
        "fourth-order staggered limit is 0.495, not the collocated 0.258: {}",
        fourth.cfl_limit(3)
    );
}

// ── Generic instantiation ────────────────────────────────────────────────────

#[test]
fn the_operator_runs_at_every_supported_scalar() {
    // The kernels are generic; every scalar a consumer can instantiate is one
    // that was exercised. f32 carries the same contract at its own precision.
    let shape = [6, 4, 5];
    let op32 = StaggeredLeapfrog3D::<f32>::new(4, 1.0, 1.0, 1.0).unwrap();
    let op64 = StaggeredLeapfrog3D::<f64>::new(4, 1.0, 1.0, 1.0).unwrap();
    let field64 = seeded(shape, 0.6);
    let mut field32 = Array3::<f32>::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                field32[[i, j, k]] = field64[[i, j, k]] as f32;
            }
        }
    }

    let mut dst32 = Array3::<f32>::zeros(shape);
    op32.gradient_into(Axis::Y, field32.view(), &mut dst32.view_mut())
        .unwrap();
    let mut dst64 = Array3::<f64>::zeros(shape);
    op64.gradient_into(Axis::Y, field64.view(), &mut dst64.view_mut())
        .unwrap();

    // f32 carries about 2^-24 relative; the stencil sums 2N taps, so the
    // difference is bounded by the input rounding plus the sum's growth.
    let bound = 32.0 * f32::EPSILON as f64;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let narrow = f64::from(dst32[[i, j, k]]);
                let wide = dst64[[i, j, k]];
                let scale = wide.abs().max(1.0);
                assert!(
                    (narrow - wide).abs() <= bound * scale,
                    "({i}, {j}, {k}): f32 {narrow} vs f64 {wide}"
                );
            }
        }
    }
}

/// The leapfrog pair writes into a mutable view over storage this crate does
/// not own, matching the owned path bitwise on both operators.
///
/// The pair takes the contiguous fast path through `as_mut_slice`, which is the
/// one that would silently fall back to indexed addressing — or fail to see the
/// buffer at all — if the view's slice access did not carry through.
#[test]
fn the_pair_writes_through_a_view_over_a_foreign_slice() {
    use leto::{ArrayViewMut3, Layout};

    let shape = [6usize, 5, 7];
    let count = shape[0] * shape[1] * shape[2];
    let strides = [(shape[1] * shape[2]) as isize, shape[2] as isize, 1_isize];
    let field = seeded(shape, 0.8);
    let op = StaggeredLeapfrog3D::<f64>::new(4, 1.5e-3, 2.5e-3, 0.5e-3).unwrap();

    for axis in AXES {
        let mut owned_gradient = Array3::zeros(shape);
        op.gradient_into(axis, field.view(), &mut owned_gradient.view_mut())
            .unwrap();
        let mut owned_divergence = Array3::zeros(shape);
        op.divergence_into(axis, field.view(), &mut owned_divergence.view_mut())
            .unwrap();

        let mut foreign_gradient = vec![f64::NAN; count];
        let layout = Layout::<3>::try_new(shape, strides, 0).unwrap();
        let mut view = ArrayViewMut3::try_new(layout, foreign_gradient.as_mut_slice()).unwrap();
        op.gradient_into(axis, field.view(), &mut view).unwrap();

        let mut foreign_divergence = vec![f64::NAN; count];
        let layout = Layout::<3>::try_new(shape, strides, 0).unwrap();
        let mut view = ArrayViewMut3::try_new(layout, foreign_divergence.as_mut_slice()).unwrap();
        op.divergence_into(axis, field.view(), &mut view).unwrap();

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let index = (i * shape[1] + j) * shape[2] + k;
                    assert_eq!(
                        foreign_gradient[index],
                        owned_gradient[[i, j, k]],
                        "gradient {axis:?} ({i}, {j}, {k})"
                    );
                    assert_eq!(
                        foreign_divergence[index],
                        owned_divergence[[i, j, k]],
                        "divergence {axis:?} ({i}, {j}, {k})"
                    );
                }
            }
        }
    }
}
