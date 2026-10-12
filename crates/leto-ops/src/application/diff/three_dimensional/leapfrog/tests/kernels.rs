//! Gradient, wall and kernel-agreement checks.

use super::*;

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
    // transposing the field. The 41³ cube moves more than the parallel floor,
    // so both traversals also spread over tasks.
    for n in [6, 41] {
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
                    assert_eq!(
                        along_z[[i, j, k]],
                        along_x[[k, j, i]],
                        "n {n} ({i}, {j}, {k})"
                    );
                }
            }
        }
    }
}
