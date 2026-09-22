use super::*;

/// Three results from the same three axis derivatives and two pointwise
/// fields -- the diagonal of an elastic stress tensor, which is why the
/// three-destination kernel exists.
pub(super) fn diagonal_from(exx: f64, eyy: f64, ezz: f64, lambda: f64, mu: f64) -> [f64; 3] {
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
        op.map_axis_derivatives_many(
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
    op.map_axis_derivatives_many(
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
        op.map_axis_derivatives_many(
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
            .map_axis_derivatives_many(
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

/// The six stresses of an isotropic elastic solid from its displacement
/// gradients `g[a][b] = ∂u_a/∂x_b`: the diagonal as [`diagonal_from`], the
/// shears as `μ (∂u_a/∂x_b + ∂u_b/∂x_a)`.
fn six_stresses(g: [[f64; 3]; 3], lambda: f64, mu: f64) -> [f64; 6] {
    let [xx, yy, zz] = diagonal_from(g[0][0], g[1][1], g[2][2], lambda, mu);
    [
        xx,
        yy,
        zz,
        mu * (g[0][1] + g[1][0]),
        mu * (g[0][2] + g[2][0]),
        mu * (g[1][2] + g[2][1]),
    ]
}

/// All six stresses from one pass over the nine displacement gradients are
/// the diagonal pass and the three shear passes to the bit -- each value is
/// the same arithmetic on the same derivatives -- on the dense walk and on
/// the logical walk a transposed field forces.
#[test]
fn six_destinations_from_one_pass_match_four_passes_bit_for_bit() {
    let shape = [7usize, 6, 5];
    let u = [
        seeded(shape),
        seeded_offset(shape, 3.5),
        seeded_offset(shape, -7.25),
    ];
    let (lambda, mu) = (seeded_offset(shape, 11.0), seeded_offset(shape, -2.5));
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");

    let mut four = [(); 6].map(|()| Array3::from_elem(shape, f64::NAN));
    {
        let [xx, yy, zz, xy, xz, yz] = &mut four;
        let (mut a, mut b, mut c) = (xx.view_mut(), yy.view_mut(), zz.view_mut());
        op.map_axis_derivatives_many(
            [
                (Axis::X, u[0].view()),
                (Axis::Y, u[1].view()),
                (Axis::Z, u[2].view()),
            ],
            [lambda.view(), mu.view()],
            [&mut a, &mut b, &mut c],
            |[exx, eyy, ezz], [la, mv]| diagonal_from(exx, eyy, ezz, la, mv),
        )
        .expect("matching shapes");
        for ((first, second), out) in [
            (((Axis::Y, &u[0]), (Axis::X, &u[1])), xy),
            (((Axis::Z, &u[0]), (Axis::X, &u[2])), xz),
            (((Axis::Z, &u[1]), (Axis::Y, &u[2])), yz),
        ] {
            op.map_axis_derivatives(
                [(first.0, first.1.view()), (second.0, second.1.view())],
                [mu.view()],
                &mut out.view_mut(),
                |[p, q], [m]| m * (p + q),
            )
            .expect("matching shapes");
        }
    }

    let stored = reversed_storage(&u[1]);
    let transposed = stored.transpose([2, 1, 0]).expect("a permutation");
    for (walk, uy) in [("dense", u[1].view()), ("logical", transposed)] {
        let mut one = [(); 6].map(|()| Array3::from_elem(shape, f64::NAN));
        let mut views = one.each_mut().map(Array3::view_mut);
        let [a, b, c, d, e, f] = &mut views;
        op.map_axis_derivatives_many(
            [
                (Axis::X, u[0].view()),
                (Axis::Y, u[0].view()),
                (Axis::Z, u[0].view()),
                (Axis::X, uy),
                (Axis::Y, uy),
                (Axis::Z, uy),
                (Axis::X, u[2].view()),
                (Axis::Y, u[2].view()),
                (Axis::Z, u[2].view()),
            ],
            [lambda.view(), mu.view()],
            [a, b, c, d, e, f],
            |[a0, a1, a2, b0, b1, b2, c0, c1, c2], [la, mv]| {
                six_stresses([[a0, a1, a2], [b0, b1, b2], [c0, c1, c2]], la, mv)
            },
        )
        .expect("matching shapes");
        for (label, (fused, composed)) in ["xx", "yy", "zz", "xy", "xz", "yz"]
            .into_iter()
            .zip(one.iter().zip(&four))
        {
            assert_bitwise(fused, |index| composed[index], &format!("{walk} {label}"));
        }
    }
}

/// A fused pass with nowhere to write is refused rather than run for nothing.
#[test]
fn a_pass_with_no_destination_is_refused() {
    let shape = [4usize, 3, 3];
    let field = seeded(shape);
    let op = FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing");
    match op.map_axis_derivatives_many([(Axis::X, field.view())], [], [], |[_], []| []) {
        Err(LetoError::InvalidInput(message)) => assert!(
            message.contains("at least one destination"),
            "unexpected message {message}"
        ),
        other => panic!("a pass with no destination was accepted: {other:?}"),
    }
}
