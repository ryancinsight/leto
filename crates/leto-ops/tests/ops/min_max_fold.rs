//! `min`/`max` fold contract: a reduction returns an element of its input.
//!
//! `Complex` orders lexicographically by real part, then imaginary part (the
//! order of `NumericElement::min_scalar`/`max_scalar`), so a fold over complex
//! input returns the lexicographic extremum, never the seed. Real scalars keep
//! the IEEE contract: a NaN lane is ignored and an all-NaN input reduces to the
//! identity.
#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use eunomia::{Bf16, Complex, F16};
use leto::{Array, Storage, VecStorage};
use leto_ops::{max, max_axis, min, min_axis, Scalar};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

/// Real parts tie at `2` and at `-1`, so the imaginary part decides both
/// extrema: the minimum is `(-1, -5)` and the maximum is `(2, 7)`.
fn tied_real_parts() -> Vec<C> {
    vec![
        c(2.0, 1.0),
        c(2.0, -3.0),
        c(2.0, 7.0),
        c(-1.0, 4.0),
        c(-1.0, -5.0),
        c(0.0, 0.0),
    ]
}

#[test]
fn complex_slice_extrema_are_lexicographic_elements() {
    let data = tied_real_parts();
    assert_eq!(<C as Scalar>::min_slice(&data), c(-1.0, -5.0));
    assert_eq!(<C as Scalar>::max_slice(&data), c(2.0, 7.0));
}

#[test]
fn complex_slice_extrema_of_a_single_element_is_that_element() {
    let data = [c(5.0, -6.0)];
    assert_eq!(<C as Scalar>::min_slice(&data), c(5.0, -6.0));
    assert_eq!(<C as Scalar>::max_slice(&data), c(5.0, -6.0));
}

/// An element equal to a fold identity, or sharing one component with it, is
/// still returned: the identity is the least or greatest pair, not a sentinel
/// that absorbs input.
#[test]
fn complex_slice_extrema_keep_elements_at_the_identity() {
    let inf = f64::INFINITY;
    let top = [c(inf, inf), c(inf, 1.0)];
    assert_eq!(<C as Scalar>::min_slice(&top), c(inf, 1.0));
    assert_eq!(<C as Scalar>::max_slice(&top), c(inf, inf));
    let bottom = [c(-inf, -inf), c(-inf, 2.0)];
    assert_eq!(<C as Scalar>::min_slice(&bottom), c(-inf, -inf));
    assert_eq!(<C as Scalar>::max_slice(&bottom), c(-inf, 2.0));
    // `(re = -inf, im < 0)` is the element a `(-inf, 0)` max identity dropped.
    let below_zero = [c(-inf, -1.0)];
    assert_eq!(<C as Scalar>::max_slice(&below_zero), c(-inf, -1.0));
    let above_zero = [c(inf, 1.0)];
    assert_eq!(<C as Scalar>::min_slice(&above_zero), c(inf, 1.0));
}

/// Real parts tie in row 1 (`3`), row 2 (`0`) and column 2 (`3`), so the
/// imaginary part decides there. Axis 1 takes the contiguous slice route and
/// axis 0 the strided fold.
fn tied_matrix() -> Array<C, VecStorage<C>, 2> {
    let rows = [
        [c(2.0, 1.0), c(2.0, -3.0), c(2.0, 7.0), c(-1.0, 4.0)],
        [c(-1.0, -5.0), c(0.0, 0.0), c(3.0, 2.0), c(3.0, 2.0)],
        [c(0.0, 9.0), c(0.0, -9.0), c(3.0, -8.0), c(-4.0, 1.0)],
    ];
    Array::from_shape_vec([3, 4], rows.iter().flatten().copied().collect()).unwrap()
}

#[test]
fn complex_axis_extrema_are_lexicographic_elements_on_both_routes() {
    let input = tied_matrix();

    let row_min = min_axis(&input.view(), 1).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();
    assert_eq!(
        row_min.storage().as_slice(),
        &[c(-1.0, 4.0), c(-1.0, -5.0), c(-4.0, 1.0)]
    );
    assert_eq!(
        row_max.storage().as_slice(),
        &[c(2.0, 7.0), c(3.0, 2.0), c(3.0, -8.0)]
    );

    let column_min = min_axis(&input.view(), 0).unwrap();
    let column_max = max_axis(&input.view(), 0).unwrap();
    assert_eq!(
        column_min.storage().as_slice(),
        &[c(-1.0, -5.0), c(0.0, -9.0), c(2.0, 7.0), c(-4.0, 1.0)]
    );
    assert_eq!(
        column_max.storage().as_slice(),
        &[c(2.0, 1.0), c(2.0, -3.0), c(3.0, 2.0), c(3.0, 2.0)]
    );
}

#[test]
fn complex_whole_array_extrema_are_lexicographic_elements() {
    let input = tied_matrix();
    assert_eq!(min(&input.view()).unwrap(), c(-4.0, 1.0));
    assert_eq!(max(&input.view()).unwrap(), c(3.0, 2.0));
}

#[test]
fn complex_single_element_axis_extrema_are_that_element() {
    let input = Array::from_shape_vec([1, 1], vec![c(5.0, -6.0)]).unwrap();
    for axis in 0..2 {
        assert_eq!(
            min_axis(&input.view(), axis).unwrap().storage().as_slice(),
            &[c(5.0, -6.0)]
        );
        assert_eq!(
            max_axis(&input.view(), axis).unwrap().storage().as_slice(),
            &[c(5.0, -6.0)]
        );
    }
}

/// An element whose real part is NaN never replaces the accumulator, wherever
/// it sits, and an input of such elements reduces to the fold identity.
#[test]
fn complex_nan_real_part_is_ignored_on_every_route() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    let leading = [c(nan, 1.0), c(2.0, 3.0), c(-1.0, 0.0)];
    let trailing = [c(2.0, 3.0), c(-1.0, 0.0), c(nan, 1.0)];
    for data in [leading, trailing] {
        assert_eq!(<C as Scalar>::min_slice(&data), c(-1.0, 0.0));
        assert_eq!(<C as Scalar>::max_slice(&data), c(2.0, 3.0));
        let column = Array::from_shape_vec([3, 1], data.to_vec()).unwrap();
        assert_eq!(min(&column.view()).unwrap(), c(-1.0, 0.0));
        assert_eq!(max(&column.view()).unwrap(), c(2.0, 3.0));
        assert_eq!(
            min_axis(&column.view(), 0).unwrap().storage().as_slice(),
            &[c(-1.0, 0.0)]
        );
        assert_eq!(
            max_axis(&column.view(), 0).unwrap().storage().as_slice(),
            &[c(2.0, 3.0)]
        );
    }
    let all_nan = [c(nan, 1.0), c(nan, -1.0)];
    assert_eq!(<C as Scalar>::min_slice(&all_nan), c(inf, inf));
    assert_eq!(<C as Scalar>::max_slice(&all_nan), c(-inf, -inf));
}

/// Real scalars keep the IEEE contract under the order-based fold: the result
/// is the numeric extremum for every position of a NaN lane, infinities are
/// ordinary elements, and an all-NaN input reduces to the identity, on the
/// slice route and on the contiguous and strided axis routes alike.
fn real_extrema_ignore_nan_and_keep_infinities<T>(lift: fn(f32) -> T)
where
    T: Scalar + Copy + PartialEq + core::fmt::Debug,
{
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    #[rustfmt::skip]
    let rows = [
        [nan, 3.0, -1.0, 2.0],
        [3.0, nan, -1.0, 2.0],
        [3.0, -1.0, 2.0, nan],
        [nan, nan, nan, nan],
        [inf, 1.0, 2.0, 3.0],
        [-inf, 1.0, 2.0, 3.0],
        [inf, inf, inf, inf],
        [-inf, -inf, -inf, -inf],
    ];
    let expected_min = [-1.0, -1.0, -1.0, inf, 1.0, -inf, inf, -inf];
    let expected_max = [3.0, 3.0, 3.0, -inf, inf, 3.0, inf, -inf];

    let data = rows.iter().flatten().map(|&x| lift(x)).collect();
    let input = Array::from_shape_vec([rows.len(), 4], data).unwrap();
    let row_min = min_axis(&input.view(), 1).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();
    for (index, row) in rows.iter().enumerate() {
        let lifted: Vec<T> = row.iter().map(|&x| lift(x)).collect();
        let (least, greatest) = (lift(expected_min[index]), lift(expected_max[index]));
        assert_eq!(T::min_slice(&lifted), least, "min_slice row {index}");
        assert_eq!(T::max_slice(&lifted), greatest, "max_slice row {index}");
        assert_eq!(
            row_min.storage().as_slice()[index],
            least,
            "min_axis row {index}"
        );
        assert_eq!(
            row_max.storage().as_slice()[index],
            greatest,
            "max_axis row {index}"
        );
    }

    // Strided axis: each column folds one element per row, in row order.
    let column_min = min_axis(&input.view(), 0).unwrap();
    let column_max = max_axis(&input.view(), 0).unwrap();
    assert_eq!(
        column_min.storage().as_slice(),
        &[lift(-inf), lift(-inf), lift(-inf), lift(-inf)]
    );
    assert_eq!(
        column_max.storage().as_slice(),
        &[lift(inf), lift(inf), lift(inf), lift(inf)]
    );
}

#[test]
fn real_extrema_ignore_nan_and_keep_infinities_across_shipped_floats() {
    real_extrema_ignore_nan_and_keep_infinities::<f32>(|x| x);
    real_extrema_ignore_nan_and_keep_infinities::<f64>(f64::from);
    real_extrema_ignore_nan_and_keep_infinities::<F16>(F16::from_f32);
    real_extrema_ignore_nan_and_keep_infinities::<Bf16>(Bf16::from_f32);
}

/// An element at the type's own fold identity is returned, not dropped: the
/// identity is `MAX_VALUE` for a minimum and `MIN_VALUE` for a maximum.
fn extrema_keep_an_element_at_the_identity<T>()
where
    T: Scalar + Copy + PartialEq + core::fmt::Debug,
{
    let at_max = [T::MAX_VALUE];
    let at_min = [T::MIN_VALUE];
    assert_eq!(T::min_slice(&at_max), T::MAX_VALUE);
    assert_eq!(T::max_slice(&at_min), T::MIN_VALUE);
    let pair = [T::MAX_VALUE, T::MIN_VALUE];
    assert_eq!(T::min_slice(&pair), T::MIN_VALUE);
    assert_eq!(T::max_slice(&pair), T::MAX_VALUE);
    let column = Array::from_shape_vec([2, 1], pair.to_vec()).unwrap();
    assert_eq!(min(&column.view()).unwrap(), T::MIN_VALUE);
    assert_eq!(max(&column.view()).unwrap(), T::MAX_VALUE);
}

#[test]
fn extrema_keep_an_element_at_the_identity_across_shipped_scalars() {
    extrema_keep_an_element_at_the_identity::<f32>();
    extrema_keep_an_element_at_the_identity::<f64>();
    extrema_keep_an_element_at_the_identity::<F16>();
    extrema_keep_an_element_at_the_identity::<Bf16>();
    extrema_keep_an_element_at_the_identity::<i8>();
    extrema_keep_an_element_at_the_identity::<i32>();
    extrema_keep_an_element_at_the_identity::<i64>();
    extrema_keep_an_element_at_the_identity::<u8>();
    extrema_keep_an_element_at_the_identity::<u32>();
    extrema_keep_an_element_at_the_identity::<u64>();
    extrema_keep_an_element_at_the_identity::<usize>();
}
