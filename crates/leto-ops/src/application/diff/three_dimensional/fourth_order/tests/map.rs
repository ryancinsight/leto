use super::*;

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
