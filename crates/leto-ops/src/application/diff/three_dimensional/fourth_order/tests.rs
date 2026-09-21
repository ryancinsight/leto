use leto::Array3;

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
    assert!(refusal.is_err(), "an unfused scheme must say so");
}

/// The shear shape: a scale times the sum of two axis derivatives of two
/// different fields, which is what `map_axis_derivatives` exists to fuse.
fn composed_scaled_pair(
    first: (Axis, &Array3<f64>),
    second: (Axis, &Array3<f64>),
    scale: &Array3<f64>,
) -> Array3<f64> {
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let shape = scale.shape();
    let sweep = |axis: Axis, field: &Array3<f64>| {
        let mut out = Array3::from_elem(shape, f64::NAN);
        let mut view = out.view_mut();
        match axis {
            Axis::X => op.apply_x_into(field.view(), &mut view),
            Axis::Y => op.apply_y_into(field.view(), &mut view),
            Axis::Z => op.apply_z_into(field.view(), &mut view),
        }
        .expect("matching shapes");
        out
    };
    let a = sweep(first.0, first.1);
    let b = sweep(second.0, second.1);
    let [nx, ny, nz] = shape;
    let mut out = Array3::from_elem(shape, f64::NAN);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = [i, j, k];
                out[p] = scale[p] * (a[p] + b[p]);
            }
        }
    }
    out
}

fn mapped_scaled_pair(
    first: (Axis, &Array3<f64>),
    second: (Axis, &Array3<f64>),
    scale: &Array3<f64>,
) -> Array3<f64> {
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let mut out = Array3::from_elem(scale.shape(), f64::NAN);
    let mut view = out.view_mut();
    op.map_axis_derivatives(
        [(first.0, first.1.view()), (second.0, second.1.view())],
        [scale.view()],
        &mut view,
        |[a, b], [m]| m * (a + b),
    )
    .expect("matching shapes");
    out
}

#[test]
fn a_scaled_axis_pair_matches_its_composed_sweeps_bit_for_bit() {
    for shape in [
        [5usize, 4, 6],
        [9, 7, 8],
        [32, 30, 28],
        [4, 3, 1],
        [1, 6, 5],
        [2, 2, 2],
    ] {
        let first = seeded(shape);
        let second = seeded_offset(shape, 3.5);
        let scale = seeded_offset(shape, -7.25);
        // Every axis pairing the elastic shear stresses use.
        for (a, b) in [
            (Axis::Y, Axis::X),
            (Axis::Z, Axis::X),
            (Axis::Z, Axis::Y),
            (Axis::X, Axis::Z),
        ] {
            let composed = composed_scaled_pair((a, &first), (b, &second), &scale);
            let mapped = mapped_scaled_pair((a, &first), (b, &second), &scale);
            assert_bitwise(
                &mapped,
                |index| composed[index],
                &format!("{shape:?} {a:?}{b:?}"),
            );
        }
    }
}

#[test]
fn a_mapped_sum_is_the_fused_divergence_bit_for_bit() {
    for shape in [[9usize, 7, 8], [32, 30, 28], [2, 2, 2], [1, 6, 5]] {
        let fields = [
            seeded(shape),
            seeded_offset(shape, 3.5),
            seeded_offset(shape, -7.25),
        ];
        let fused = fused_divergence([&fields[0], &fields[1], &fields[2]]);
        let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
            .expect("positive spacing");
        let mut out = Array3::from_elem(shape, f64::NAN);
        let mut view = out.view_mut();
        op.map_axis_derivatives(
            [
                (Axis::X, fields[0].view()),
                (Axis::Y, fields[1].view()),
                (Axis::Z, fields[2].view()),
            ],
            [],
            &mut view,
            |[a, b, c], []| (a + b) + c,
        )
        .expect("matching shapes");
        assert_bitwise(&out, |index| fused[index], "mapped sum");
    }
}

#[test]
fn a_transposed_field_takes_the_logical_walk_in_a_mapped_pair() {
    let shape = [6usize, 6, 6];
    let first = seeded(shape);
    let second = seeded_offset(shape, 3.5);
    let scale = seeded_offset(shape, -7.25);
    let dense = mapped_scaled_pair((Axis::Z, &first), (Axis::X, &second), &scale);

    let stored = reversed_storage(&second);
    let transposed = stored.transpose([2, 1, 0]).expect("a permutation");
    assert!(
        transposed.as_slice().is_none(),
        "the case needs a non-C-dense field"
    );
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let mut out = Array3::from_elem(shape, f64::NAN);
    let mut view = out.view_mut();
    op.map_axis_derivatives(
        [(Axis::Z, first.view()), (Axis::X, transposed)],
        [scale.view()],
        &mut view,
        |[a, b], [m]| m * (a + b),
    )
    .expect("matching shapes");
    assert_bitwise(&out, |index| dense[index], "transposed mapped pair");
}

#[test]
fn a_mapped_mismatch_and_an_unfused_scheme_are_refused() {
    let shape = [6usize, 5, 7];
    let first = seeded(shape);
    let second = seeded(shape);
    let scale = seeded(shape);
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");

    let mut wrong = Array3::<f64>::zeros([6, 5, 6]);
    assert!(op
        .map_axis_derivatives(
            [(Axis::Y, first.view()), (Axis::X, second.view())],
            [scale.view()],
            &mut wrong.view_mut(),
            |[a, b], [m]| m * (a + b),
        )
        .is_err());

    let mut out = Array3::<f64>::zeros(shape);
    let short = Array3::<f64>::zeros([6, 5, 6]);
    assert!(
        op.map_axis_derivatives(
            [(Axis::Y, first.view()), (Axis::X, second.view())],
            [short.view()],
            &mut out.view_mut(),
            |[a, b], [m]| m * (a + b),
        )
        .is_err(),
        "a pointwise input off the grid must be refused"
    );

    let second_order = FiniteDifference3D::central_second_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    assert!(
        second_order
            .map_axis_derivatives(
                [(Axis::Y, first.view()), (Axis::X, second.view())],
                [scale.view()],
                &mut out.view_mut(),
                |[a, b], [m]| m * (a + b),
            )
            .is_err(),
        "an unfused scheme must say so"
    );
}

/// Three results from the same three axis derivatives and two pointwise
/// fields -- the diagonal of an elastic stress tensor, which is why the
/// three-destination kernel exists.
fn diagonal_from(exx: f64, eyy: f64, ezz: f64, lambda: f64, mu: f64) -> [f64; 3] {
    let la2mu = 2.0f64.mul_add(mu, lambda);
    [
        la2mu.mul_add(exx, lambda * (eyy + ezz)),
        la2mu.mul_add(eyy, lambda * (exx + ezz)),
        la2mu.mul_add(ezz, lambda * (exx + eyy)),
    ]
}

fn composed_diagonal(
    fields: [&Array3<f64>; 3],
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
) -> [Array3<f64>; 3] {
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let shape = lambda.shape();
    let mut strains = [
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
    ];
    for (axis, (field, strain)) in AXES.into_iter().zip(fields.into_iter().zip(&mut strains)) {
        let mut view = strain.view_mut();
        match axis {
            Axis::X => op.apply_x_into(field.view(), &mut view),
            Axis::Y => op.apply_y_into(field.view(), &mut view),
            Axis::Z => op.apply_z_into(field.view(), &mut view),
        }
        .expect("matching shapes");
    }
    let [nx, ny, nz] = shape;
    let mut out = [
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
    ];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = [i, j, k];
                let values = diagonal_from(
                    strains[0][p],
                    strains[1][p],
                    strains[2][p],
                    lambda[p],
                    mu[p],
                );
                for (component, value) in out.iter_mut().zip(values) {
                    component[p] = value;
                }
            }
        }
    }
    out
}

#[test]
fn a_mapped_triple_matches_its_composed_sweeps_bit_for_bit() {
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
        let lambda = seeded_offset(shape, 11.0);
        let mu = seeded_offset(shape, -2.5);
        let composed = composed_diagonal([&fields[0], &fields[1], &fields[2]], &lambda, &mu);

        let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
            .expect("positive spacing");
        let mut out = [
            Array3::from_elem(shape, f64::NAN),
            Array3::from_elem(shape, f64::NAN),
            Array3::from_elem(shape, f64::NAN),
        ];
        let [one, two, three] = &mut out;
        let (mut first, mut second, mut third) = (one.view_mut(), two.view_mut(), three.view_mut());
        op.map_axis_derivatives_triple(
            [
                (Axis::X, fields[0].view()),
                (Axis::Y, fields[1].view()),
                (Axis::Z, fields[2].view()),
            ],
            [lambda.view(), mu.view()],
            [&mut first, &mut second, &mut third],
            |[exx, eyy, ezz], [la, mv]| diagonal_from(exx, eyy, ezz, la, mv),
        )
        .expect("matching shapes");

        for (component, expected) in out.iter().zip(&composed) {
            assert_bitwise(component, |index| expected[index], &format!("{shape:?}"));
        }
    }
}

#[test]
fn a_transposed_field_takes_the_logical_walk_in_a_mapped_triple() {
    let shape = [6usize, 6, 6];
    let fields = [
        seeded(shape),
        seeded_offset(shape, 3.5),
        seeded_offset(shape, -7.25),
    ];
    let lambda = seeded_offset(shape, 11.0);
    let mu = seeded_offset(shape, -2.5);
    let dense = composed_diagonal([&fields[0], &fields[1], &fields[2]], &lambda, &mu);

    let stored = reversed_storage(&fields[1]);
    let transposed = stored.transpose([2, 1, 0]).expect("a permutation");
    assert!(
        transposed.as_slice().is_none(),
        "the case needs a non-C-dense field"
    );
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let mut out = [
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
        Array3::from_elem(shape, f64::NAN),
    ];
    let [one, two, three] = &mut out;
    let (mut first, mut second, mut third) = (one.view_mut(), two.view_mut(), three.view_mut());
    op.map_axis_derivatives_triple(
        [
            (Axis::X, fields[0].view()),
            (Axis::Y, transposed),
            (Axis::Z, fields[2].view()),
        ],
        [lambda.view(), mu.view()],
        [&mut first, &mut second, &mut third],
        |[exx, eyy, ezz], [la, mv]| diagonal_from(exx, eyy, ezz, la, mv),
    )
    .expect("matching shapes");

    for (component, expected) in out.iter().zip(&dense) {
        assert_bitwise(
            component,
            |index| expected[index],
            "transposed mapped triple",
        );
    }
}

#[test]
fn a_mapped_triple_mismatch_and_an_unfused_scheme_are_refused() {
    let shape = [6usize, 5, 7];
    let field = seeded(shape);
    let scale = seeded(shape);
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    let terms = || {
        [
            (Axis::X, field.view()),
            (Axis::Y, field.view()),
            (Axis::Z, field.view()),
        ]
    };
    let combine = |[a, b, c]: [f64; 3], [m]: [f64; 1]| [m * a, m * b, m * c];

    let mut good = Array3::<f64>::zeros(shape);
    let mut other = Array3::<f64>::zeros(shape);
    let mut wrong = Array3::<f64>::zeros([6, 5, 6]);
    assert!(
        op.map_axis_derivatives_triple(
            terms(),
            [scale.view()],
            [
                &mut good.view_mut(),
                &mut other.view_mut(),
                &mut wrong.view_mut()
            ],
            combine,
        )
        .is_err(),
        "a destination off the grid must be refused"
    );

    let mut third = Array3::<f64>::zeros(shape);
    let second_order = FiniteDifference3D::central_second_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    assert!(
        second_order
            .map_axis_derivatives_triple(
                terms(),
                [scale.view()],
                [
                    &mut good.view_mut(),
                    &mut other.view_mut(),
                    &mut third.view_mut()
                ],
                combine,
            )
            .is_err(),
        "an unfused scheme must say so"
    );
}
