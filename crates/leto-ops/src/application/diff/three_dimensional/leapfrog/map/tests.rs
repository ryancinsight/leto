//! The fused gradient against the separate sweep and pass it replaces.

use leto::{Array3, LetoError};

use super::super::{Axis, StaggeredLeapfrog3D};

const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// Shapes under, at, and past the order-8 halo on every axis, none cubic,
/// so a swapped axis cannot pass.
const SHAPES: [[usize; 3]; 3] = [[3, 5, 4], [9, 7, 8], [12, 10, 11]];

fn seeded(shape: [usize; 3], salt: f64) -> Array3<f64> {
    Array3::from_shape_fn(shape, |[i, j, k]| {
        ((i as f64).mul_add(0.37, salt).sin() * (j as f64).mul_add(0.53, -salt).cos())
            + (k as f64).mul_add(0.71, 0.25 * salt).sin()
    })
}

fn assert_bitwise(actual: &Array3<f64>, expected: &Array3<f64>, context: &str) {
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}: flat {index}: {a} against {e}"
        );
    }
}

/// A leapfrog velocity update, `v − (Δt/ρ) g`, fused, equals the swept
/// gradient followed by the same arithmetic in a separate pass, to the bit,
/// for every order, axis and shape; and a combine that returns the gradient
/// alone is the plain sweep.
#[test]
fn a_fused_update_is_the_sweep_then_the_pass_to_the_bit() {
    const DT: f64 = 3.7e-3;
    for order in [2, 4, 8] {
        let op = StaggeredLeapfrog3D::new(order, 0.7, 1.1, 1.3).expect("valid operator");
        for shape in SHAPES {
            let pressure = seeded(shape, 0.4);
            let density = seeded(shape, 1.9).mapv(|value| 1_000.0 + 100.0 * value);
            let velocity = seeded(shape, -2.3);
            for axis in AXES {
                let context = format!("order {order}, {shape:?}, {axis:?}");
                let mut swept = Array3::zeros(shape);
                op.gradient_into(axis, pressure.view(), &mut swept.view_mut())
                    .expect("matching shapes");
                let mut expected = velocity.clone();
                for ((v, &g), &rho) in expected.iter_mut().zip(swept.iter()).zip(density.iter()) {
                    *v -= DT / rho * g;
                }

                let mut fused = velocity.clone();
                op.map_gradient_into(
                    axis,
                    pressure.view(),
                    [density.view()],
                    &mut fused.view_mut(),
                    |g, [rho], v| v - DT / rho * g,
                )
                .expect("matching shapes");
                assert_bitwise(&fused, &expected, &context);

                let mut gradient = velocity.clone();
                op.map_gradient_into(
                    axis,
                    pressure.view(),
                    [],
                    &mut gradient.view_mut(),
                    |g, [], _| g,
                )
                .expect("matching shapes");
                assert_bitwise(&gradient, &swept, &format!("{context}, gradient alone"));
            }
        }
    }
}

/// A transposed field takes the logical walk and gives the dense values.
#[test]
fn a_transposed_field_takes_the_logical_walk() {
    let op = StaggeredLeapfrog3D::new(4, 0.7, 1.1, 1.3).expect("valid operator");
    let shape = [9, 7, 8];
    let pressure = seeded(shape, 0.4);
    let transposed_source = Array3::from_shape_fn([8, 7, 9], |[k, j, i]| pressure[[i, j, k]]);
    let transposed = transposed_source
        .view()
        .transpose([2, 1, 0])
        .expect("a permutation");
    assert!(
        transposed.as_slice().is_none(),
        "the case needs a strided field"
    );
    for axis in AXES {
        let mut dense = Array3::zeros(shape);
        op.map_gradient_into(
            axis,
            pressure.view(),
            [],
            &mut dense.view_mut(),
            |g, [], v| v + g,
        )
        .expect("matching shapes");
        let mut logical = Array3::zeros(shape);
        op.map_gradient_into(axis, transposed, [], &mut logical.view_mut(), |g, [], v| {
            v + g
        })
        .expect("matching shapes");
        assert_bitwise(&logical, &dense, &format!("{axis:?}"));
    }
}

/// A destination or pointwise input of another shape is refused before
/// anything is written.
#[test]
fn a_mismatched_shape_is_refused_untouched() {
    let op = StaggeredLeapfrog3D::new(2, 1.0, 1.0, 1.0).expect("valid operator");
    let field = seeded([4, 5, 6], 0.4);
    let mut dst = Array3::from_elem([4, 5, 6], 7.0);
    let short = Array3::<f64>::zeros([4, 5, 5]);
    let refused = |outcome: leto::Result<()>| matches!(outcome, Err(LetoError::InvalidInput(message)) if message.contains("[4, 5, 5]"));
    assert!(refused(op.map_gradient_into(
        Axis::X,
        field.view(),
        [short.view()],
        &mut dst.view_mut(),
        |g, [_], _| g,
    )));
    let mut wrong = Array3::from_elem([4, 5, 5], 7.0);
    assert!(refused(op.map_gradient_into(
        Axis::X,
        field.view(),
        [],
        &mut wrong.view_mut(),
        |g, [], _| g,
    )));
    assert!(dst.iter().chain(wrong.iter()).all(|&value| value == 7.0));
}
