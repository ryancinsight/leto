//! `min`/`max` select the least or greatest non-NaN element, whatever the
//! element order or memory layout. `Complex` orders by real part, then
//! imaginary part; an element with any NaN component is a NaN lane, ignored as
//! a real NaN is; an empty or all-NaN lane reduces to the fold identity.
#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use eunomia::{Bf16, Complex, F16};
use leto::{Array, ArrayView, SliceArg, Storage, VecStorage};
use leto_ops::{max, max_axis, min, min_axis, Scalar};

/// Runs `$check::<T>(lift)` for every shipped component type; `lift` is exact.
macro_rules! over_components {
    ($check:ident) => {{
        $check::<f32>(|x| x);
        $check::<f64>(f64::from);
        $check::<F16>(F16::from_f32);
        $check::<Bf16>(Bf16::from_f32);
    }};
}

type Lift<T> = fn(f32) -> T;

fn array<T: Copy>(shape: [usize; 2], data: Vec<T>) -> Array<T, VecStorage<T>, 2> {
    Array::from_shape_vec(shape, data).unwrap()
}

/// Whole-array and per-lane `(minima, maxima)` of a rank-2 view, asserted by
/// value: rows (contiguous axis) and columns (strided axis).
fn assert_extrema<T>(
    view: &ArrayView<'_, T, 2>,
    whole: (T, T),
    rows: (&[T], &[T]),
    columns: (&[T], &[T]),
) where
    T: Scalar + Copy + PartialEq + core::fmt::Debug,
{
    assert_eq!(min(view).unwrap(), whole.0);
    assert_eq!(max(view).unwrap(), whole.1);
    let lanes = |reduced: Array<T, VecStorage<T>, 2>| reduced.storage().as_slice().to_vec();
    assert_eq!(lanes(min_axis(view, 1).unwrap()), rows.0);
    assert_eq!(lanes(max_axis(view, 1).unwrap()), rows.1);
    assert_eq!(lanes(min_axis(view, 0).unwrap()), columns.0);
    assert_eq!(lanes(max_axis(view, 0).unwrap()), columns.1);
}

/// Real parts tie in rows 1 and 2 and in column 2, so the imaginary part
/// decides there.
fn complex_extrema_are_lexicographic_elements<T>(lift: Lift<T>)
where
    T: Copy + core::fmt::Debug,
    Complex<T>: Scalar + PartialEq,
{
    let c = |re: f32, im: f32| Complex::new(lift(re), lift(im));
    let mut slice = [
        c(2.0, 1.0),
        c(2.0, -3.0),
        c(2.0, 7.0),
        c(-1.0, 4.0),
        c(-1.0, -5.0),
        c(0.0, 0.0),
    ];
    for _ in 0..2 {
        assert_eq!(Scalar::min_slice(&slice), c(-1.0, -5.0));
        assert_eq!(Scalar::max_slice(&slice), c(2.0, 7.0));
        slice.reverse();
    }
    assert_eq!(Scalar::min_slice(&[c(5.0, -6.0)]), c(5.0, -6.0));
    assert_eq!(Scalar::max_slice(&[c(5.0, -6.0)]), c(5.0, -6.0));
    // Elements at or beside a fold identity, and the empty identity.
    let inf = f32::INFINITY;
    assert_eq!(Scalar::min_slice(&[c(inf, inf), c(inf, 1.0)]), c(inf, 1.0));
    assert_eq!(Scalar::max_slice(&[c(inf, inf), c(inf, 1.0)]), c(inf, inf));
    assert_eq!(Scalar::max_slice(&[c(-inf, -1.0)]), c(-inf, -1.0));
    assert_eq!(<Complex<T> as Scalar>::min_slice(&[]), c(inf, inf));
    assert_eq!(<Complex<T> as Scalar>::max_slice(&[]), c(-inf, -inf));

    let rows = [
        [c(2.0, 1.0), c(2.0, -3.0), c(2.0, 7.0), c(-1.0, 4.0)],
        [c(-1.0, -5.0), c(0.0, 0.0), c(3.0, 2.0), c(3.0, 2.0)],
        [c(0.0, 9.0), c(0.0, -9.0), c(3.0, -8.0), c(-4.0, 1.0)],
    ];
    let input = array([3, 4], rows.concat());
    assert_extrema(
        &input.view(),
        (c(-4.0, 1.0), c(3.0, 2.0)),
        (
            &[c(-1.0, 4.0), c(-1.0, -5.0), c(-4.0, 1.0)],
            &[c(2.0, 7.0), c(3.0, 2.0), c(3.0, -8.0)],
        ),
        (
            &[c(-1.0, -5.0), c(0.0, -9.0), c(2.0, 7.0), c(-4.0, 1.0)],
            &[c(2.0, 1.0), c(2.0, -3.0), c(3.0, 2.0), c(3.0, 2.0)],
        ),
    );
}

#[test]
fn complex_extrema_are_lexicographic_elements_across_component_types() {
    over_components!(complex_extrema_are_lexicographic_elements);
}

/// An element with a NaN component never replaces the accumulator, so the
/// result is a selection over the non-NaN elements: independent of element
/// order and of memory layout, and the identity when none exists.
fn complex_nan_component_is_ignored<T>(lift: Lift<T>)
where
    T: Copy + core::fmt::Debug,
    Complex<T>: Scalar + PartialEq,
{
    let c = |re: f32, im: f32| Complex::new(lift(re), lift(im));
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    let (top, bottom) = (c(inf, inf), c(-inf, -inf));

    for lone in [c(inf, nan), c(nan, 1.0)] {
        assert_eq!(Scalar::min_slice(&[lone]), top);
        assert_eq!(Scalar::max_slice(&[lone]), bottom);
    }
    let elements = [c(2.0, 0.0), c(1.0, nan), c(1.0, -1.0)];
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let slice = order.map(|index| elements[index]);
        assert_eq!(Scalar::min_slice(&slice), c(1.0, -1.0));
        assert_eq!(Scalar::max_slice(&slice), c(2.0, 0.0));
    }

    // One logical array: transposed column-major buffer, C-order copy, strided.
    let (a, b, d, e) = (c(2.0, 0.0), c(1.0, nan), c(1.0, -1.0), c(5.0, 5.0));
    let column_major = array([2, 2], vec![a, b, d, e]);
    let transposed = column_major.view().transpose([1, 0]).unwrap();
    let c_order = array([2, 2], vec![a, d, b, e]);
    for view in [&transposed, &c_order.view()] {
        assert_extrema(view, (d, e), (&[d, e], &[a, e]), (&[a, d], &[a, e]));
    }
    let (f, g) = (c(-7.0, -7.0), c(7.0, 7.0));
    let (h, i) = (c(1.0, 0.0), c(9.0, 9.0));
    let source = array([4, 2], [[h, i], [f, g], [b, d], [g, f]].concat());
    let strided = source
        .view()
        .slice_with::<2>(&[SliceArg::range(Some(0), None, 2), SliceArg::All])
        .unwrap();
    assert_extrema(&strided, (d, i), (&[h, d], &[i, d]), (&[h, d], &[h, i]));

    // An all-NaN row and an all-NaN column reduce to the identity.
    let (j, k) = (c(2.0, 5.0), c(nan, nan));
    let mixed = array([2, 2], vec![c(nan, 1.0), c(3.0, nan), j, k]);
    let (row, column) = (
        (&[top, j][..], &[bottom, j][..]),
        (&[j, top][..], &[j, bottom][..]),
    );
    assert_extrema(&mixed.view(), (j, j), row, column);
}

#[test]
fn complex_nan_component_is_ignored_across_component_types() {
    over_components!(complex_nan_component_is_ignored);
}

/// Real scalars keep the IEEE contract: a NaN lane is ignored wherever it
/// sits, infinities are elements, and an all-NaN lane reduces to the identity,
/// on the slice, contiguous axis and strided axis routes.
fn real_extrema_keep_the_ieee_contract<T>(lift: Lift<T>)
where
    T: Scalar + Copy + PartialEq + core::fmt::Debug,
{
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    #[rustfmt::skip]
    let rows = [
        [nan, 3.0, -1.0, 2.0],
        [3.0, nan, -1.0, 2.0],
        [3.0, -1.0, 2.0, nan],
        [nan, nan, nan, nan],
        [inf, 1.0, 2.0, 3.0],
        [-inf, 1.0, 2.0, 3.0],
    ];
    let lanes = |expected: [f32; 6]| expected.map(lift);
    let (row_min, row_max) = (
        lanes([-1.0, -1.0, -1.0, inf, 1.0, -inf]),
        lanes([3.0, 3.0, 3.0, -inf, inf, 3.0]),
    );
    let input = array([6, 4], rows.concat().into_iter().map(lift).collect());
    let reduced = |lanes: Array<T, VecStorage<T>, 2>| lanes.storage().as_slice().to_vec();
    assert_eq!(reduced(min_axis(&input.view(), 1).unwrap()), row_min);
    assert_eq!(reduced(max_axis(&input.view(), 1).unwrap()), row_max);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(T::min_slice(&row.map(lift)), row_min[index]);
        assert_eq!(T::max_slice(&row.map(lift)), row_max[index]);
    }
    assert_eq!(
        reduced(min_axis(&input.view(), 0).unwrap()),
        [-inf, -1.0, -1.0, 2.0].map(lift)
    );
    assert_eq!(
        reduced(max_axis(&input.view(), 0).unwrap()),
        [inf, 3.0, 2.0, 3.0].map(lift)
    );
}

#[test]
fn real_extrema_keep_the_ieee_contract_across_shipped_floats() {
    over_components!(real_extrema_keep_the_ieee_contract);
}

/// An extremum that differs from the fold seed is selected over it (the pair
/// `[MAX_VALUE, MIN_VALUE]` answers each fold with the other seed).
#[test]
fn extrema_differing_from_the_seed_are_selected_across_shipped_scalars() {
    fn check<T: Scalar + Copy + PartialEq + core::fmt::Debug>() {
        let pair = [T::MAX_VALUE, T::MIN_VALUE];
        assert_eq!(T::min_slice(&pair), T::MIN_VALUE);
        assert_eq!(T::max_slice(&pair), T::MAX_VALUE);
        assert_eq!(T::min_slice(&[]), T::MAX_VALUE);
        assert_eq!(T::max_slice(&[]), T::MIN_VALUE);
    }
    macro_rules! over_scalars {
        ($($t:ty),*) => {$(check::<$t>();)*};
    }
    over_scalars!(f32, f64, F16, Bf16, i8, i32, i64, u8, u32, u64, usize);
}
