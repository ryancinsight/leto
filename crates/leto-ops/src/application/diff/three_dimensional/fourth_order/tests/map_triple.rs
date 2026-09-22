use super::*;

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
