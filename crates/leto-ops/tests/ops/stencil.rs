use aequitas::systems::si::{quantities::Length, units::Meter};
use leto::{Array1, BoundaryCondition, Laplacian2D, LaplacianPolarity};
use leto_ops::{laplacian_2d_into, RealScalar};

fn negative_neumann_quadratic_is_exact<T>()
where
    T: RealScalar,
{
    let nx = 4;
    let ny = 4;
    let dx = T::from_f32(0.5);
    let dy = T::from_f32(0.25);
    let three = T::from_f32(3.0);
    let values = (0..ny)
        .flat_map(|y| {
            (0..nx).map(move |x| {
                let x = T::from_usize(x) * dx;
                let y = T::from_usize(y) * dy;
                x * x + three * y * y
            })
        })
        .collect();
    let input = Array1::from_shape_vec([nx * ny], values).expect("valid input shape");
    let mut output = Array1::zeros([nx * ny]);
    let stencil = Laplacian2D::new(
        nx,
        ny,
        Length::from_unit::<Meter>(dx),
        Length::from_unit::<Meter>(dy),
        BoundaryCondition::Neumann,
    )
    .expect("valid stencil")
    .with_polarity(LaplacianPolarity::NegativeLaplacian);

    laplacian_2d_into(&stencil, &input.view(), &mut output.view_mut())
        .expect("matching array lengths");

    assert!(output.iter().all(|&value| value == T::from_f32(-8.0)));
}

#[test]
fn negative_neumann_quadratic_matches_closed_form() {
    negative_neumann_quadratic_is_exact::<f32>();
    negative_neumann_quadratic_is_exact::<f64>();
}
