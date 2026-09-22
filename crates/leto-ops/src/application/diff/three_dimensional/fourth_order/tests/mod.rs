use leto::{Array3, LetoError};

use super::super::leapfrog::Axis;
use super::super::FiniteDifference3D;

const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
const SPACING: [f64; 3] = [0.3, 0.7, 1.1];

/// Values with no structure a stencil could cancel, so bit comparisons mean
/// something.
fn seeded(shape: [usize; 3]) -> Array3<f64> {
    let mut field = Array3::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let t = (i as f64).mul_add(1.3, (j as f64).mul_add(2.9, k as f64 * 0.7));
                field[[i, j, k]] = t.sin() * (t * 0.37).cos() + 0.25 * t;
            }
        }
    }
    field
}

/// `field` stored so that its `[2, 1, 0]` transpose reads as `field`: a view
/// that is not C-dense, which sends the operator down the logical walk.
fn reversed_storage(field: &Array3<f64>) -> Array3<f64> {
    let [nx, ny, nz] = field.shape();
    let mut stored = Array3::zeros([nz, ny, nx]);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                stored[[k, j, i]] = field[[i, j, k]];
            }
        }
    }
    stored
}

fn apply(field: leto::ArrayView3<'_, f64>, axis: Axis) -> Array3<f64> {
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let mut out = Array3::from_elem(field.shape(), f64::NAN);
    let mut view = out.view_mut();
    match axis {
        Axis::X => op.apply_x_into(field, &mut view),
        Axis::Y => op.apply_y_into(field, &mut view),
        Axis::Z => op.apply_z_into(field, &mut view),
    }
    .expect("matching shapes");
    out
}

/// The documented closure, restated per point from the stencil table.
fn reference(field: &Array3<f64>, axis: Axis, index: [usize; 3]) -> f64 {
    let d = axis.index();
    let (n, c, h) = (field.shape()[d], index[d], SPACING[d]);
    let at = |offset: isize| {
        let mut neighbour = index;
        neighbour[d] = c
            .checked_add_signed(offset)
            .expect("stencil stays on the axis");
        field[neighbour]
    };
    if n == 1 {
        0.0
    } else if c == 0 {
        (at(1) - at(0)) * (1.0 / h)
    } else if c == n - 1 {
        (at(0) - at(-1)) * (1.0 / h)
    } else if c < 2 || c >= n - 2 {
        (at(1) - at(-1)) * (1.0 / (2.0 * h))
    } else {
        ((-8.0 * at(-1)) + (8.0 * at(1)) + (-at(2)) + at(-2)) * (1.0 / (12.0 * h))
    }
}

fn assert_bitwise(actual: &Array3<f64>, expected: impl Fn([usize; 3]) -> f64, context: &str) {
    let [nx, ny, nz] = actual.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (a, e) = (actual[[i, j, k]], expected([i, j, k]));
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{context} at [{i}, {j}, {k}]: {a} vs {e}"
                );
            }
        }
    }
}

/// Every axis length from a singleton up past the interior stencil, on each
/// axis, on both traversals.
#[test]
fn every_axis_length_takes_the_documented_closure_on_both_paths() {
    for axis in AXES {
        for n in 1..=8 {
            let mut shape = [5, 4, 6];
            shape[axis.index()] = n;
            let field = seeded(shape);
            let dense = apply(field.view(), axis);
            assert_bitwise(
                &dense,
                |index| reference(&field, axis, index),
                &format!("dense {axis:?} n={n}"),
            );

            let stored = reversed_storage(&field);
            let transposed = stored.transpose([2, 1, 0]).expect("a permutation");
            assert!(
                transposed.as_slice().is_none() || shape.iter().filter(|&&e| e > 1).count() < 2
            );
            let strided = apply(transposed, axis);
            assert_bitwise(
                &strided,
                |index| dense[index],
                &format!("logical walk {axis:?} n={n}"),
            );
        }
    }
}

/// A volume large enough to spread over tasks gives the value each point
/// takes alone.
#[cfg(feature = "parallel")]
#[test]
fn a_parallel_sweep_matches_the_pointwise_closure() {
    let shape = [32, 30, 28];
    assert!(
        shape.iter().product::<usize>() * super::ELEMENTS_PER_UNIT * size_of::<f64>()
            >= crate::infrastructure::parallel::PARALLEL_MIN_BYTES,
        "the volume must spread over tasks"
    );
    let field = seeded(shape);
    for axis in AXES {
        let dense = apply(field.view(), axis);
        assert_bitwise(
            &dense,
            |index| reference(&field, axis, index),
            &format!("{axis:?}"),
        );
    }
}

/// The closure is exact on the polynomials its orders promise: a quartic in
/// the interior, a quadratic next to the walls, a line at the walls.
#[test]
fn each_order_is_exact_on_its_polynomial() {
    let n = 9;
    let h = SPACING[0];
    let mut field = Array3::zeros([n, 3, 3]);
    for i in 0..n {
        let x = i as f64 * h;
        field[[i, 1, 1]] = x.powi(4) - 2.0 * x.powi(3) + x;
    }
    let dense = apply(field.view(), Axis::X);
    let derivative = |x: f64| 4.0 * x.powi(3) - 6.0 * x.powi(2) + 1.0;
    // Interior: a quartic's derivative is recovered to rounding. The bound is
    // a few ULPs of the largest term, |f| * 16 / (12 h) at x = 2.4.
    for i in 2..n - 2 {
        let x = i as f64 * h;
        let bound = 64.0 * f64::EPSILON * (x.powi(4) + 2.0 * x.powi(3) + x) / h;
        assert!(
            (dense[[i, 1, 1]] - derivative(x)).abs() <= bound,
            "interior i={i}: {} vs {}",
            dense[[i, 1, 1]],
            derivative(x)
        );
    }
    // Walls: first order on a line, second order next to it on a quadratic.
    let mut line = Array3::zeros([n, 3, 3]);
    let mut quadratic = Array3::zeros([n, 3, 3]);
    for i in 0..n {
        let x = i as f64 * h;
        line[[i, 1, 1]] = 3.0 * x - 1.0;
        quadratic[[i, 1, 1]] = x * x;
    }
    let line_dense = apply(line.view(), Axis::X);
    let quadratic_dense = apply(quadratic.view(), Axis::X);
    for i in [0, n - 1] {
        assert!((line_dense[[i, 1, 1]] - 3.0).abs() <= 16.0 * f64::EPSILON * 3.0 * n as f64 / h);
    }
    for i in [1, n - 2] {
        let x = i as f64 * h;
        assert!(
            (quadratic_dense[[i, 1, 1]] - 2.0 * x).abs()
                <= 16.0 * f64::EPSILON * (n as f64 * h).powi(2) / h
        );
    }
}

/// `seeded` shifted, so the three divergence inputs differ everywhere.
fn seeded_offset(shape: [usize; 3], offset: f64) -> Array3<f64> {
    let mut field = seeded(shape);
    for value in field.iter_mut() {
        *value = *value * 1.37 + offset;
    }
    field
}

/// The composed form the fused kernel replaces: one buffer per axis, summed
/// in x, y, z order.
fn composed_divergence(fields: [&Array3<f64>; 3]) -> Array3<f64> {
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let shape = fields[0].shape();
    let mut parts = [
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
    ];
    for (axis, (field, part)) in fields.iter().zip(parts.iter_mut()).enumerate() {
        let mut view = part.view_mut();
        match axis {
            0 => op.apply_x_into(field.view(), &mut view),
            1 => op.apply_y_into(field.view(), &mut view),
            _ => op.apply_z_into(field.view(), &mut view),
        }
        .expect("matching shapes");
    }
    let [nx, ny, nz] = shape;
    let mut sum = Array3::zeros(shape);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = [i, j, k];
                sum[p] = (parts[0][p] + parts[1][p]) + parts[2][p];
            }
        }
    }
    sum
}

fn fused_divergence(fields: [&Array3<f64>; 3]) -> Array3<f64> {
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let mut out = Array3::from_elem(fields[0].shape(), f64::NAN);
    let mut view = out.view_mut();
    op.divergence_into(
        [fields[0].view(), fields[1].view(), fields[2].view()],
        &mut view,
    )
    .expect("matching shapes");
    out
}

/// Shapes below and past the sweep's parallel floor, with short and singleton
/// axes that take only the wall closures.
#[test]
fn the_fused_divergence_is_the_composed_one_bit_for_bit() {
    for shape in [
        [5usize, 4, 6],
        [9, 7, 8],
        [32, 30, 28],
        [4, 3, 1],
        [1, 6, 5],
        [2, 2, 2],
    ] {
        let fields = [
            seeded(shape),
            seeded_offset(shape, 3.5),
            seeded_offset(shape, -7.25),
        ];
        let borrowed = [&fields[0], &fields[1], &fields[2]];
        let fused = fused_divergence(borrowed);
        let composed = composed_divergence(borrowed);
        let [nx, ny, nz] = shape;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let p = [i, j, k];
                    assert_eq!(
                        fused[p].to_bits(),
                        composed[p].to_bits(),
                        "shape {shape:?} at {p:?}: {} vs {}",
                        fused[p],
                        composed[p]
                    );
                }
            }
        }
    }
}

/// A field the caller stores transposed takes the logical walk; the values
/// are the same ones the dense path produces.
#[test]
fn a_transposed_field_gives_the_dense_values() {
    let shape = [6usize, 6, 6];
    let fields = [
        seeded(shape),
        seeded_offset(shape, 3.5),
        seeded_offset(shape, -7.25),
    ];
    let dense = fused_divergence([&fields[0], &fields[1], &fields[2]]);

    let stored = reversed_storage(&fields[1]);
    let transposed = stored.transpose([2, 1, 0]).expect("a permutation");
    assert!(
        transposed.as_slice().is_none(),
        "the case needs a non-C-dense field"
    );
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let mut out = Array3::from_elem(shape, f64::NAN);
    let mut view = out.view_mut();
    op.divergence_into([fields[0].view(), transposed, fields[2].view()], &mut view)
        .expect("matching shapes");
    assert_bitwise(&out, |index| dense[index], "transposed field");
}

#[test]
fn a_mismatched_shape_and_an_unfused_scheme_are_refused() {
    let shape = [6usize, 5, 7];
    let fields = [seeded(shape), seeded(shape), seeded(shape)];
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");

    let mut wrong = Array3::<f64>::zeros([6, 5, 6]);
    assert!(op
        .divergence_into(
            [fields[0].view(), fields[1].view(), fields[2].view()],
            &mut wrong.view_mut()
        )
        .is_err());

    let mut out = Array3::<f64>::zeros(shape);
    let second_order = FiniteDifference3D::central_second_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let refusal = second_order.divergence_into(
        [fields[0].view(), fields[1].view(), fields[2].view()],
        &mut out.view_mut(),
    );
    // Exact text of the `other` branch in `FiniteDifference3D::divergence_into`
    // (operator.rs): the refusal must name the scheme it has no fused kernel
    // for, not merely fail.
    assert_eq!(
        refusal,
        Err(LetoError::InvalidInput(
            "divergence_into has no fused kernel for CentralSecondOrder; apply each \
             axis separately and sum"
                .to_string()
        )),
        "an unfused scheme must name itself in the refusal"
    );
}

mod map;
mod map_triple;
