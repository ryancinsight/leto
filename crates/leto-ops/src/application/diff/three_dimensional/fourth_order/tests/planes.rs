use core::ops::Range;

use super::super::super::window::{PlaneWindow, PlaneWindowMut};
use super::map_many::diagonal_from;
use super::*;

/// Plane counts spanning a singleton, the one-sided and second-order closures,
/// and an interior deep enough that a slab boundary can fall on every stencil.
const EXTENTS: [usize; 5] = [1, 2, 3, 5, 9];

/// Every way to cut `0..nx` once, plus plane-at-a-time: each plane is written
/// by a call whose range starts or ends at it, including at both closures.
fn partitions(nx: usize) -> Vec<Vec<Range<usize>>> {
    let mut all: Vec<Vec<Range<usize>>> = (0..=nx).map(|cut| vec![0..cut, cut..nx]).collect();
    all.push((0..nx).map(|x| x..x + 1).collect());
    all
}

fn operator() -> FiniteDifference3D<f64> {
    FiniteDifference3D::central_fourth_order(SPACING[0], SPACING[1], SPACING[2])
        .expect("positive spacing")
}

/// A scaled divergence: three axes, one pointwise field, one destination.
fn divergence_on(
    planes: Range<usize>,
    fields: [leto::ArrayView3<'_, f64>; 3],
    scale: leto::ArrayView3<'_, f64>,
    dst: &mut Array3<f64>,
) -> leto::Result<()> {
    let [fx, fy, fz] = fields;
    let grid_planes = dst.shape()[0];
    operator().map_axis_derivatives_in_windows(
        grid_planes,
        planes,
        [
            (Axis::X, PlaneWindow::whole(fx)),
            (Axis::Y, PlaneWindow::whole(fy)),
            (Axis::Z, PlaneWindow::whole(fz)),
        ],
        [PlaneWindow::whole(scale)],
        [PlaneWindowMut::whole(&mut dst.view_mut())],
        |[a, b, c], [s]| [(a + b + c) * s],
    )
}

/// The elastic diagonal: three axes, two pointwise fields, three destinations.
fn diagonal_on(
    planes: Range<usize>,
    fields: [leto::ArrayView3<'_, f64>; 3],
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    dst: &mut [Array3<f64>; 3],
) -> leto::Result<()> {
    let [fx, fy, fz] = fields;
    let [one, two, three] = dst;
    let (mut first, mut second, mut third) = (one.view_mut(), two.view_mut(), three.view_mut());
    let grid_planes = first.shape()[0];
    operator().map_axis_derivatives_in_windows(
        grid_planes,
        planes,
        [
            (Axis::X, PlaneWindow::whole(fx)),
            (Axis::Y, PlaneWindow::whole(fy)),
            (Axis::Z, PlaneWindow::whole(fz)),
        ],
        [
            PlaneWindow::whole(lambda.view()),
            PlaneWindow::whole(mu.view()),
        ],
        [
            PlaneWindowMut::whole(&mut first),
            PlaneWindowMut::whole(&mut second),
            PlaneWindowMut::whole(&mut third),
        ],
        |[exx, eyy, ezz], [la, mv]| diagonal_from(exx, eyy, ezz, la, mv),
    )
}

fn nan_field(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_elem(shape, f64::NAN)
}

fn nan_triple(shape: [usize; 3]) -> [Array3<f64>; 3] {
    [nan_field(shape), nan_field(shape), nan_field(shape)]
}

/// Slabbed calls take the stencil the whole-grid call takes at each plane, so
/// any partition of the planes rebuilds the whole-grid result to the bit --
/// on the dense walk, and on the logical walk a transposed field forces.
#[test]
fn every_partition_of_the_planes_rebuilds_the_whole_grid_bit_for_bit() {
    for nx in EXTENTS {
        let shape = [nx, 4, 5];
        let fields = [
            seeded(shape),
            seeded_offset(shape, 3.5),
            seeded_offset(shape, -7.25),
        ];
        let (lambda, mu) = (seeded_offset(shape, 11.0), seeded_offset(shape, -2.5));
        let stored = reversed_storage(&fields[1]);
        let transposed = stored.transpose([2, 1, 0]).expect("a permutation");
        assert!(
            transposed.as_slice().is_none() || nx == 1,
            "the logical walk needs a non-C-dense field"
        );
        for (walk, second) in [("dense", fields[1].view()), ("logical", transposed)] {
            let views = || [fields[0].view(), second, fields[2].view()];

            let mut whole = nan_field(shape);
            divergence_on(0..nx, views(), mu.view(), &mut whole).expect("valid planes");
            let mut whole_diagonal = nan_triple(shape);
            diagonal_on(0..nx, views(), &lambda, &mu, &mut whole_diagonal).expect("valid planes");

            for partition in partitions(nx) {
                let context = format!("{walk} walk, {nx} planes, slabs {partition:?}");
                let mut slabbed = nan_field(shape);
                let mut slabbed_diagonal = nan_triple(shape);
                for planes in &partition {
                    divergence_on(planes.clone(), views(), mu.view(), &mut slabbed)
                        .expect("valid planes");
                    diagonal_on(planes.clone(), views(), &lambda, &mu, &mut slabbed_diagonal)
                        .expect("valid planes");
                }
                assert_bitwise(&slabbed, |index| whole[index], &context);
                for (slabbed, whole) in slabbed_diagonal.iter().zip(&whole_diagonal) {
                    assert_bitwise(slabbed, |index| whole[index], &context);
                }
            }
        }
    }
}

/// A call writes its planes and nothing else, so a consumer can interleave
/// slabs of different passes over one field.
#[test]
fn planes_outside_the_range_are_left_as_they_were() {
    let shape = [7usize, 4, 5];
    let fields = [
        seeded(shape),
        seeded_offset(shape, 1.5),
        seeded_offset(shape, -2.0),
    ];
    let views = || [fields[0].view(), fields[1].view(), fields[2].view()];
    let scale = seeded_offset(shape, 4.0);
    let mut dst = nan_field(shape);
    divergence_on(2..5, views(), scale.view(), &mut dst).expect("valid planes");
    for i in 0..shape[0] {
        let written = (2..5).contains(&i);
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                assert_eq!(
                    dst[[i, j, k]].is_nan(),
                    !written,
                    "plane {i}: written {written} disagrees with the range 2..5"
                );
            }
        }
    }
}

/// A range that runs backwards or past the planes is refused before anything
/// is written, and an empty range writes nothing.
#[test]
fn a_backward_or_overlong_range_is_refused_and_an_empty_one_writes_nothing() {
    let shape = [4usize, 3, 3];
    let fields = [
        seeded(shape),
        seeded_offset(shape, 2.0),
        seeded_offset(shape, 5.0),
    ];
    let views = || [fields[0].view(), fields[1].view(), fields[2].view()];
    let (lambda, mu) = (seeded_offset(shape, 9.0), seeded_offset(shape, -1.0));
    let backward = Range { start: 3, end: 1 };
    for planes in [backward, 2..5, 5..5] {
        let mut dst = nan_field(shape);
        let mut diagonal = nan_triple(shape);
        let single = divergence_on(planes.clone(), views(), mu.view(), &mut dst);
        let triple = diagonal_on(planes.clone(), views(), &lambda, &mu, &mut diagonal);
        for (form, outcome) in [("single", single), ("triple", triple)] {
            match outcome {
                Err(LetoError::InvalidInput(message)) => assert!(
                    message.contains("do not lie within 0..4"),
                    "{form} {planes:?}: unexpected message {message}"
                ),
                other => panic!("{form} {planes:?} was accepted: {other:?}"),
            }
        }
        assert!(dst.iter().all(|value| value.is_nan()), "{planes:?} wrote");
        assert!(
            diagonal
                .iter()
                .all(|field| field.iter().all(|value| value.is_nan())),
            "{planes:?} wrote a diagonal"
        );
    }
    let mut dst = nan_field(shape);
    divergence_on(2..2, views(), mu.view(), &mut dst).expect("an empty range is valid");
    assert!(
        dst.iter().all(|value| value.is_nan()),
        "an empty range wrote"
    );
}

/// The same rebuild with slabs large enough to split into moirai tasks, so a
/// task's first plane is an offset into the range rather than into the grid.
#[cfg(feature = "parallel")]
#[test]
fn slabs_split_across_tasks_rebuild_the_whole_grid_bit_for_bit() {
    let shape = [32usize, 30, 28];
    let partition = [0..5, 5..19, 19..32];
    // One output, four neighbours for each of three terms, one pointwise
    // field: the single-destination kernel's bytes per element.
    let element_bytes = (1 + 3 * (super::super::ELEMENTS_PER_UNIT - 1) + 1) * size_of::<f64>();
    let plane_bytes = shape[1] * shape[2] * element_bytes;
    assert!(
        partition[1..]
            .iter()
            .all(|planes| planes.len() * plane_bytes
                >= crate::infrastructure::parallel::PARALLEL_MIN_BYTES),
        "the inner slabs must spread over tasks"
    );
    let fields = [
        seeded(shape),
        seeded_offset(shape, 3.5),
        seeded_offset(shape, -7.25),
    ];
    let views = || [fields[0].view(), fields[1].view(), fields[2].view()];
    let (lambda, mu) = (seeded_offset(shape, 11.0), seeded_offset(shape, -2.5));

    let mut whole = nan_field(shape);
    divergence_on(0..shape[0], views(), mu.view(), &mut whole).expect("valid planes");
    let mut whole_diagonal = nan_triple(shape);
    diagonal_on(0..shape[0], views(), &lambda, &mu, &mut whole_diagonal).expect("valid planes");

    let mut slabbed = nan_field(shape);
    let mut slabbed_diagonal = nan_triple(shape);
    for planes in partition {
        divergence_on(planes.clone(), views(), mu.view(), &mut slabbed).expect("valid planes");
        diagonal_on(planes, views(), &lambda, &mu, &mut slabbed_diagonal).expect("valid planes");
    }
    assert_bitwise(&slabbed, |index| whole[index], "parallel slabs");
    for (slabbed, whole) in slabbed_diagonal.iter().zip(&whole_diagonal) {
        assert_bitwise(slabbed, |index| whole[index], "parallel diagonal slabs");
    }
}

/// The planes `planes` of `field`, as a buffer holding only them.
fn planes_of(field: &Array3<f64>, planes: Range<usize>) -> Array3<f64> {
    let [_, ny, nz] = field.shape();
    let mut held = Array3::zeros([planes.len(), ny, nz]);
    for (local, i) in planes.enumerate() {
        for j in 0..ny {
            for k in 0..nz {
                held[[local, j, k]] = field[[i, j, k]];
            }
        }
    }
    held
}

/// A slab computed from windows holding only the planes it reads, into a
/// buffer holding only the planes it writes, is the whole-grid result on
/// those planes to the bit -- the stencil follows the grid plane, not the
/// window's edge -- on the dense walk and on the logical walk.
#[test]
fn windows_holding_only_what_a_slab_reads_give_the_whole_grid_planes() {
    let nx = 9;
    let shape = [nx, 4, 5];
    let fields = [
        seeded(shape),
        seeded_offset(shape, 3.5),
        seeded_offset(shape, -7.25),
    ];
    let (lambda, mu) = (seeded_offset(shape, 11.0), seeded_offset(shape, -2.5));
    let whole_views = || [fields[0].view(), fields[1].view(), fields[2].view()];
    let mut whole = nan_field(shape);
    divergence_on(0..nx, whole_views(), mu.view(), &mut whole).expect("valid planes");
    let mut whole_diagonal = nan_triple(shape);
    diagonal_on(0..nx, whole_views(), &lambda, &mu, &mut whole_diagonal).expect("valid planes");

    for (start, end) in [(0_usize, 3_usize), (2, 5), (3, 4), (4, 9), (0, 9), (8, 9)] {
        let reached = start.saturating_sub(2)..(end + 2).min(nx);
        let x_field = planes_of(&fields[0], reached.clone());
        let y_field = planes_of(&fields[1], start..end);
        let z_field = planes_of(&fields[2], start..end);
        let (lambda_held, mu_held) = (planes_of(&lambda, start..end), planes_of(&mu, start..end));
        let y_reversed = reversed_storage(&y_field);
        let y_logical = y_reversed.transpose([2, 1, 0]).expect("a permutation");
        for (walk, y_view) in [("dense", y_field.view()), ("logical", y_logical)] {
            let context = format!("{walk} walk, slab {start}..{end}");
            let terms = [
                (Axis::X, PlaneWindow::new(x_field.view(), reached.start)),
                (Axis::Y, PlaneWindow::new(y_view, start)),
                (Axis::Z, PlaneWindow::new(z_field.view(), start)),
            ];
            let mut slab = nan_field([end - start, 4, 5]);
            operator()
                .map_axis_derivatives_in_windows(
                    nx,
                    start..end,
                    terms,
                    [PlaneWindow::new(mu_held.view(), start)],
                    [PlaneWindowMut::new(&mut slab.view_mut(), start)],
                    |[a, b, c], [s]| [(a + b + c) * s],
                )
                .expect("windows covering the slab");
            assert_bitwise(&slab, |[i, j, k]| whole[[i + start, j, k]], &context);

            let mut slab_diagonal = nan_triple([end - start, 4, 5]);
            let [one, two, three] = &mut slab_diagonal;
            let (mut a, mut b, mut c) = (one.view_mut(), two.view_mut(), three.view_mut());
            operator()
                .map_axis_derivatives_in_windows(
                    nx,
                    start..end,
                    terms,
                    [
                        PlaneWindow::new(lambda_held.view(), start),
                        PlaneWindow::new(mu_held.view(), start),
                    ],
                    [
                        PlaneWindowMut::new(&mut a, start),
                        PlaneWindowMut::new(&mut b, start),
                        PlaneWindowMut::new(&mut c, start),
                    ],
                    |[exx, eyy, ezz], [la, mv]| diagonal_from(exx, eyy, ezz, la, mv),
                )
                .expect("windows covering the slab");
            for (slab, whole) in slab_diagonal.iter().zip(&whole_diagonal) {
                assert_bitwise(slab, |[i, j, k]| whole[[i + start, j, k]], &context);
            }
        }
    }
}

/// Each way a window can fail a pass is refused before anything is written,
/// naming what was wrong.
#[test]
fn windows_that_cannot_serve_the_pass_are_refused() {
    let nx = 9;
    let full = seeded([nx, 4, 5]);
    let held = |planes: Range<usize>| planes_of(&full, planes);
    let (x_reach, exact, short) = (held(1..8), held(3..6), held(4..6));
    let narrow = Array3::from_elem([3, 4, 4], 1.0);
    let past = Array3::from_elem([3, 4, 5], 1.0);
    let x_ok = || (Axis::X, PlaneWindow::new(x_reach.view(), 1));
    let y_ok = || (Axis::Y, PlaneWindow::new(exact.view(), 3));
    let scale_ok = || PlaneWindow::new(exact.view(), 3);
    let cases = [
        (
            "an x-field short of the stencil's reach",
            [(Axis::X, PlaneWindow::new(exact.view(), 3)), y_ok()],
            scale_ok(),
            "needs 1..8",
        ),
        (
            "a y-field short of the planes written",
            [x_ok(), (Axis::Y, PlaneWindow::new(short.view(), 4))],
            scale_ok(),
            "needs 3..6",
        ),
        (
            "a pointwise input short of the planes written",
            [x_ok(), y_ok()],
            PlaneWindow::new(short.view(), 4),
            "needs 3..6",
        ),
        (
            "a field with other lanes",
            [x_ok(), (Axis::Y, PlaneWindow::new(narrow.view(), 3))],
            scale_ok(),
            "has lanes [4, 4]",
        ),
        (
            "a window reaching past the grid",
            [x_ok(), (Axis::Y, PlaneWindow::new(past.view(), 7))],
            scale_ok(),
            "past the grid",
        ),
    ];
    for (case, terms, scale, expected) in cases {
        let mut slab = nan_field([3, 4, 5]);
        let outcome = operator().map_axis_derivatives_in_windows(
            nx,
            3..6,
            terms,
            [scale],
            [PlaneWindowMut::new(&mut slab.view_mut(), 3)],
            |[a, b], [s]| [(a + b) * s],
        );
        match outcome {
            Err(LetoError::InvalidInput(message)) => {
                assert!(
                    message.contains(expected),
                    "{case}: unexpected message {message}"
                );
            }
            other => panic!("{case} was accepted: {other:?}"),
        }
        assert!(
            slab.iter().all(|value| value.is_nan()),
            "{case} wrote before refusing"
        );
    }

    let (mut a, mut b, mut c) = (
        nan_field([3, 4, 5]),
        nan_field([3, 4, 5]),
        nan_field([4, 4, 5]),
    );
    let (mut va, mut vb, mut vc) = (a.view_mut(), b.view_mut(), c.view_mut());
    let outcome = operator().map_axis_derivatives_in_windows(
        nx,
        3..6,
        [x_ok()],
        [],
        [
            PlaneWindowMut::new(&mut va, 3),
            PlaneWindowMut::new(&mut vb, 3),
            PlaneWindowMut::new(&mut vc, 3),
        ],
        |[d], []| [d, d, d],
    );
    match outcome {
        Err(LetoError::InvalidInput(message)) => assert!(
            message.contains("must hold the same planes"),
            "unexpected message {message}"
        ),
        other => panic!("destinations holding different planes were accepted: {other:?}"),
    }
}
