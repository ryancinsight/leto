//! Parallel-sweep parity, CFL limit, and scalar-genericity tests.

use super::*;

#[cfg(feature = "parallel")]
#[test]
fn a_parallel_sweep_matches_its_planes_swept_alone() {
    let shape = [41, 38, 44];
    let [nx, ny, nz] = shape;
    assert!(
        nx * ny * nz * 2 * size_of::<f64>() >= crate::infrastructure::parallel::PARALLEL_MIN_BYTES,
        "the volume must spread over tasks"
    );
    let plane = ny * nz;
    let field = seeded(shape, 0.4);
    let values = field.as_slice().unwrap();
    for order in [2, 6] {
        let op = StaggeredLeapfrog3D::<f64>::new(order, 1.0e-3, 2.0e-3, 1.5e-3).unwrap();
        for axis in [Axis::Y, Axis::Z] {
            for divergence in [false, true] {
                let sweep = |input: &Array3<f64>, initial: f64| {
                    let mut out = Array3::from_elem(input.shape(), initial);
                    if divergence {
                        op.divergence_into(axis, input.view(), &mut out.view_mut())
                            .unwrap();
                    } else {
                        op.gradient_into(axis, input.view(), &mut out.view_mut())
                            .unwrap();
                    }
                    out
                };
                let whole = sweep(&field, f64::NAN);
                let whole_values = whole.as_slice().unwrap();
                for x in 0..nx {
                    let slab = Array3::from_shape_vec(
                        [1, ny, nz],
                        values[x * plane..(x + 1) * plane].to_vec(),
                    )
                    .unwrap();
                    let alone = sweep(&slab, 0.0);
                    let same = alone
                        .as_slice()
                        .unwrap()
                        .iter()
                        .zip(&whole_values[x * plane..(x + 1) * plane])
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        same,
                        "order {order} axis {axis:?} divergence {divergence} plane {x}"
                    );
                }
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
